import os
import torch
import pandas as pd
import torch.optim as optim
from torchvision import transforms  # Add this import for image transforms
from torch.utils.data import DataLoader
from src.model import CLIPModel, RNAEncoder, ImageEncoder, ContrastiveLoss
from src.dataset import RNACustomDataset
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn 

# Step 1: Load RNA-Seq data
df_rna = pd.read_csv('../RNA_counts_141024.csv', header=None, low_memory=False)
df_rna_transposed = df_rna.transpose()
new_header = df_rna_transposed.iloc[1]
df_rna_transposed = df_rna_transposed[2:]
df_rna_transposed.columns = new_header

# Step 2: Filter RNA data for 'ENSG' columns
rna_cols = df_rna_transposed.filter(like='ENSG')

# Step 3: Load and merge image data
df_images = pd.read_excel('../RNA_Sample_Linking2.xlsx')
df_images['Sample ID'] = df_images['Sample ID'].str.replace(' ', '_')
df_merged = pd.merge(df_images, df_rna_transposed, left_on='Sample ID', right_on='Geneid')

# Step 4: Prepare image filenames and filter RNA data
image_directory = '../images_matched'
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

# Convert to DataFrame
rna_data_filtered = pd.DataFrame(rna_data_filtered)

# Step 5: Split into train and test sets
rna_train, rna_test, img_train, img_test = train_test_split(rna_data_filtered, image_filenames, test_size=0.2)

# Continue with the rest of the training process...


# Split into train and test (you can change the data paths as needed)
rna_train, rna_test, img_train, img_test = train_test_split(rna_data_filtered, image_filenames, test_size=0.2)

# define transforms 
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a consistent size
    transforms.ToTensor(),  # Convert to tensor
])



# Create datasets with image_directory passed in
train_dataset = RNACustomDataset(rna_train, img_train, image_directory, transform=transform)
test_dataset = RNACustomDataset(rna_test, img_test, image_directory, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Step 2: Initialize the Model, Optimizer, and Criterion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rna_encoder = RNAEncoder(input_dim=len(rna_cols.columns), embedding_dim=512)
image_encoder = ImageEncoder(embedding_dim=512)

model = CLIPModel(rna_encoder, image_encoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = ContrastiveLoss(temperature=0.5)

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs):
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
        
        # Print both training and validation loss for each epoch
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')


# Step 3: Training Loop
epochs = 10
train(model, train_loader, test_loader, optimizer, criterion, device, epochs)
