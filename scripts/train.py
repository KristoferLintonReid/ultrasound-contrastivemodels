# Additional imports needed to load src.model and src.dataset
import sys
sys.path.append("..")

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import torch
import pandas as pd
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from src.model import CLIPModel, RNAEncoder, ImageEncoder, ContrastiveLoss
from src.dataset import RNACustomDataset
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch  # For logging PyTorch models
import torch.nn as nn 
import numpy as np
import albumentations as albu # For image augmentations
import cv2
from albumentations.pytorch import ToTensorV2

# Set the MLflow tracking URI to point to the correct folder
mlflow.set_tracking_uri("/home/kryan24/MRes_Ultrasound/CLIPRNA/scripts/mlruns")

# Step 1: Load RNA-Seq data
df_rna = pd.read_csv("/home/kryan24/MRes_Ultrasound/data/rna_counts/RNA_counts_141024.csv", header=None, low_memory=False)
df_rna_transposed = df_rna.transpose()
new_header = df_rna_transposed.iloc[1]
df_rna_transposed = df_rna_transposed[2:]
df_rna_transposed.columns = new_header

# Step 2: Filter RNA data for 'ENSG' columns
rna_cols = df_rna_transposed.filter(like='ENSG')

# Step 3: Load and merge image data
df_images = pd.read_excel("/home/kryan24/MRes_Ultrasound/data/metadata/RNA_Sample_Linking2.xlsx")
df_images['Sample ID'] = df_images['Sample ID'].str.replace(' ', '_')
df_merged = pd.merge(df_images, df_rna_transposed, left_on='Sample ID', right_on='Geneid')

# Step 4: Prepare image filenames and filter RNA data
image_directory = "/home/kryan24/MRes_Ultrasound/CLIPRNA/images_matched"
image_filenames = []
rna_data_filtered = []
max_pairs = 100
pair_count = 0

for index, row in df_merged.iterrows():
    if pair_count >= max_pairs:
        break
    
    case_id = str(row['CASE ID ']).strip() if pd.notna(row['CASE ID ']) else ''
    case_id2 = str(row['CASE ID2']).strip() if pd.notna(row['CASE ID2']) else ''
    
    for file_name in os.listdir(image_directory):
        if file_name.endswith('.nii.gz') and 'seg' not in file_name:
            if case_id in file_name or case_id2 in file_name:
                image_filenames.append(file_name)
                rna_data_filtered.append(row[rna_cols.columns].apply(pd.to_numeric, errors='coerce').fillna(0))
                pair_count += 1
                break

print(f"Number of Pairs: ", pair_count)

# Convert to DataFrame
rna_data_filtered = pd.DataFrame(rna_data_filtered)

# Normalise RNA sample counts
rna_data_normalised = np.log2(rna_data_filtered+1)

# Step 5: Split into train and test sets
rna_train, rna_test, img_train, img_test = train_test_split(rna_data_normalised, image_filenames, test_size=0.2)

# Define transforms 
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),  # Resize images to a consistent size using nearest neighbour interpolation
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # Normalise pixel values
])

transform_albu = albu.Compose([
    albu.Resize(224, 224, interpolation=cv2.INTER_NEAREST),
    albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

train_transform_albu = albu.Compose([
    albu.HorizontalFlip(p=0.5),
    albu.RandomRotate90(p=1),
    albu.GaussNoise(p=0.5),

    albu.OneOf(
        [
           albu.CLAHE(p=1),
           albu.RandomGamma(p=1)
        ],
        p=0.9
    ),

    albu.OneOf(
        [
           albu.Sharpen(p=1),
           albu.Blur(blur_limit=3, p=1),
           albu.MotionBlur(blur_limit=3, p=1)
        ],
        p=0.9
    ),

    albu.OneOf(
        [
           albu.RandomBrightnessContrast(p=1),
           albu.HueSaturationValue(p=1)
        ],
        p=0.9
    ),

    albu.Resize(224, 224, interpolation=cv2.INTER_NEAREST),
    albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()

])

# Create datasets with image_directory passed in
train_dataset = RNACustomDataset(rna_train, img_train, image_directory, transform=train_transform_albu, transform_type="albu")
test_dataset = RNACustomDataset(rna_test, img_test, image_directory, transform=transform_albu, transform_type="albu")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 2: Initialize the Model, Optimizer, and Criterion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: ", device)

rna_encoder = RNAEncoder(input_dim=len(rna_cols.columns), embedding_dim=512)
image_encoder = ImageEncoder(embedding_dim=512)

model = CLIPModel(rna_encoder, image_encoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = ContrastiveLoss(temperature=0.5)

# MLflow setup
mlflow.set_experiment("RNA-Image CLIP Model: Early Stopping and Reduce LR on Plateau")

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs):
    # Set run name
    with mlflow.start_run(run_name="earlyStopping_test_19122024"):
        mlflow.log_param("learning_rate", 1e-4)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("embedding_dim", 512)

        # Initialise variables for early stopping
        best_loss = None
        patience = 20

        for epoch in range(epochs):
            # Training Phase
            model.train()  # Set both encoders to training mode
            total_train_loss = 0
            for rna_batch, image_batch in train_loader:
                rna_batch, image_batch = rna_batch.to(device), image_batch.to(device)
                
                # Forward pass through both encoders
                rna_embeddings, image_embeddings = model(rna_batch, image_batch)
                
                # Compute contrastive loss
                loss = criterion(rna_embeddings, image_embeddings)
                total_train_loss += loss.item()
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Log training loss to MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            # Validation Phase (no gradient calculation)
            model.eval()
            total_val_loss = 0
            with torch.no_grad():  # Disable gradient calculation
                for rna_batch, image_batch in val_loader:
                    rna_batch, image_batch = rna_batch.to(device), image_batch.to(device)
                    
                    # Forward pass through both encoders
                    rna_embeddings, image_embeddings = model(rna_batch, image_batch)
                    
                    # Compute contrastive loss
                    loss = criterion(rna_embeddings, image_embeddings)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)

            # Log validation loss to MLflow
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            
            # Print both training and validation loss for each epoch
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

            # Early stopping
            if best_loss is None:
                best_loss = avg_val_loss
            elif avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience = 10
            else:
                patience -=1
                if patience == 0:
                    print(f"Early Stopping")
                    mlflow.log_metric("early_stopping_epoch", epoch)
                    break

        # Log the model at the end of the run
        mlflow.pytorch.log_model(model, "clip_model")

# Step 3: Training Loop
epochs = 1000
train(model, train_loader, test_loader, optimizer, criterion, device, epochs)
