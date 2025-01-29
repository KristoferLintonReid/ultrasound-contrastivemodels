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
from src.dataset import RNACustomDataset, ImgCustomDataset
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch  # For logging PyTorch models
import torch.nn as nn 
import numpy as np
import albumentations as albu # For image augmentations
import cv2
from albumentations.pytorch import ToTensorV2
import optuna
from src.optimise import set_seed, calc_mean_std, generate_model_name
from sklearn.preprocessing import StandardScaler

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

# Step 5: Split into train and test sets
rna_train, rna_test, img_train, img_test = train_test_split(rna_data_filtered, image_filenames, test_size=0.2, random_state=4)
# Convert back to DataFrame
rna_train = pd.DataFrame(rna_train, columns=rna_data_filtered.columns)
rna_test = pd.DataFrame(rna_test, columns=rna_data_filtered.columns)

# Scale RNA sample counts
scaler = StandardScaler()
rna_train = pd.DataFrame(scaler.fit_transform(rna_train), columns=rna_data_filtered.columns)
rna_test = pd.DataFrame(scaler.transform(rna_test), columns=rna_data_filtered.columns)

#### Calculate mean and standard deviation for normalisation ####

# Define pre-transforms 
pre_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),  # Resize images to a consistent size using nearest neighbour interpolation
    transforms.ToTensor()  # Convert to tensor
])

pre_transform_albu = albu.Compose([
    albu.Resize(224, 224, interpolation=cv2.INTER_NEAREST),
    albu.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    ToTensorV2()
])

# Create dataset with image_directory passed in
train_img_dataset = ImgCustomDataset(img_train, image_directory, transform=pre_transform_albu, transform_type="albu")

# Calculate mean and standard deviation 
num_imgs = len(img_train)
calc_mean, calc_std = calc_mean_std(train_img_dataset, num_imgs)

print(f"Mean: {calc_mean}")
print(f"Standard Deviation: {calc_std}")

#### END ####

# Define transforms 
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),  # Resize images to a consistent size using nearest neighbour interpolation
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=calc_mean, std=calc_std) # Normalise pixel values
])

transform_albu = albu.Compose([
    albu.Resize(224, 224, interpolation=cv2.INTER_NEAREST),
    albu.Normalize(mean=calc_mean, std=calc_std),
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
    albu.Normalize(mean=calc_mean, std=calc_std),
    ToTensorV2()

])

# Create datasets with image_directory passed in
train_dataset = RNACustomDataset(rna_train, img_train, image_directory, transform=train_transform_albu, transform_type="albu")
test_dataset = RNACustomDataset(rna_test, img_test, image_directory, transform=transform_albu, transform_type="albu")

# Set random seed for weights
random_seed = 2
set_seed(random_seed)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Step 6: Initialize the Model, Optimizer, and Criterion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: ", device)

rna_encoder = RNAEncoder(input_dim=len(rna_cols.columns), embedding_dim=128)
image_encoder = ImageEncoder(embedding_dim=128)

model = CLIPModel(rna_encoder, image_encoder).to(device)
learning_rate = 1e-4 # Initialise
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = ContrastiveLoss()

# MLflow setup
mlflow.set_experiment("RNA-Image CLIP Model: Inference 29012025")

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs):
    with mlflow.start_run():
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("learning_rate_sci", "{:.2e}".format(learning_rate))
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("embedding_dim", 128)

        for epoch in range(epochs):
            # Training Phase
            model.train() # Set both encoders to training mode
            total_train_loss = 0
            for rna_batch, image_batch, labels in train_loader:
                rna_batch, image_batch, labels = rna_batch.to(device), image_batch.to(device), labels.to(device)

                # Forward pass through both encoders
                rna_embeddings, image_embeddings = model(rna_batch, image_batch)

                # Compute contrastive loss
                loss = criterion(rna_embeddings, image_embeddings, labels)
                total_train_loss += loss.item()
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            avg_train_loss = total_train_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            # Validation Phase
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for rna_batch, image_batch, labels in val_loader:
                    rna_batch, image_batch, labels = rna_batch.to(device), image_batch.to(device), labels.to(device)

                    rna_embeddings, image_embeddings = model(rna_batch, image_batch)
                    loss = criterion(rna_embeddings, image_embeddings, labels)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        mlflow.pytorch.log_model(model, "clip_model")

# Step 7: Training Loop
epochs = 100
train(model, train_loader, test_loader, optimizer, criterion, device, epochs)

# Step 8: Save the model checkpoint (weights, hyperparameters, architectures)
model_directory = "/home/kryan24/MRes_Ultrasound/CLIPRNA/saved_models/"
model_name = generate_model_name()
model_path = model_directory + model_name

torch.save({
    "rna_encoder_state_dict": rna_encoder.state_dict(),
    "image_encoder_state_dict": image_encoder.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "clip_model_state_dict": model.state_dict(),
    "contrastive_loss_state_dict": criterion.state_dict(),
    "epoch": epochs,
}, model_path)