#!/usr/bin/env python3
"""
train.py

Trains a CLIP-like model (RNA + Image) across multiple sets of hyperparameters.
We do the following:
    - Experiment with different image encoder backbones (CNN, ResNet, ViT).
    - Train an MLP-based encoder for RNA data.
    - Use a contrastive loss to align RNA embeddings with image embeddings.
    - Track experiments in MLflow (parameters, metrics, models).
    - Implement early stopping and ReduceLROnPlateau for scheduling.

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import sys
# Ensure we can import from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# MLflow imports
import mlflow
import mlflow.pytorch

# Data augmentation
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

# Utility: For train/val splits, scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Optional GPU usage logging
try:
    import GPUtil
    GPUtil_available = True
except ImportError:
    GPUtil_available = False

# Local imports from 'src' directory
from src.dataset import RNACustomDataset
from src.model import RNAEncoder, create_image_encoder, CLIPModel
from src.losses import ContrastiveLoss
from src.utils import (
    set_seed,
    calc_mean_std,
    train_one_epoch,
    validate_one_epoch
)

def main(args):
    """
    Main training function that:
    1. Loads and processes RNA data.
    2. Matches RNA rows with images.
    3. Splits data into train/val, scales RNA.
    4. Creates datasets/dataloaders with augmentations.
    5. Loops through hyperparameters and trains a CLIP-like model.
    """
    # -------------------------
    # 1. Set seed and device
    # -------------------------
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -------------------------
    # 2. Load RNA Data
    # -------------------------
    print("[INFO] Loading RNA data...")
    df_rna = pd.read_csv(args.rna_csv, header=None, low_memory=False)
    
    # Transpose and set columns
    df_rna_t = df_rna.transpose()
    new_header = df_rna_t.iloc[1]  # second row has the gene IDs
    df_rna_t = df_rna_t[2:]       # skip the first two lines of metadata
    df_rna_t.columns = new_header

    # Filter only columns with "ENSG"
    rna_cols = df_rna_t.filter(like='ENSG')
    print(f"[INFO] Found {rna_cols.shape[1]} gene columns (ENSG).")

    # -------------------------
    # 3. Load & Merge Image Metadata
    # -------------------------
    print("[INFO] Loading image metadata...")
    df_images = pd.read_excel(args.metadata_xlsx)
    # Replace spaces with underscores in the 'Sample ID' column
    df_images['Sample ID'] = df_images['Sample ID'].str.replace(' ', '_')

    # Merge to link image info to the RNA data
    df_merged = pd.merge(df_images, df_rna_t, left_on='Sample ID', right_on='Geneid')
    print(f"[INFO] Merged dataframe has shape: {df_merged.shape}")

    # -------------------------
    # 4. Match images with RNA (up to max_pairs)
    # -------------------------
    def capitalize_j(s):
        if not isinstance(s, str):
            return s
        return s.replace('j', 'J')

    def format_case_id(case):
        return str(case).replace('_', '').replace(' ', '')

    image_filenames = []
    rna_data_filtered = []
    pair_count = 0

    print("[INFO] Matching RNA rows to image files...")
    for _, row in df_merged.iterrows():
        if pair_count >= args.max_pairs:
            break

        # Format the case IDs for matching
        case_id = capitalize_j(format_case_id(row['CASE ID ']))
        case_id2 = capitalize_j(format_case_id(row['CASE ID2']))

        # Look for an image file that contains the case_id or case_id2
        for file_name in os.listdir(args.image_dir):
            if file_name.endswith('.nii.gz') and 'seg' not in file_name:
                file_name_corrected = capitalize_j(file_name)
                if (case_id and case_id in file_name_corrected) or (case_id2 and case_id2 in file_name_corrected):
                    image_filenames.append(file_name)
                    # Convert row to numeric and fill NaN with 0
                    rna_data_filtered.append(
                        row[rna_cols.columns].apply(pd.to_numeric, errors='coerce').fillna(0)
                    )
                    pair_count += 1
                    break

    print(f"[INFO] Number of matched pairs: {pair_count}")
    rna_data_filtered = pd.DataFrame(rna_data_filtered)

    # -------------------------
    # 5. Train/Val Split + Scaling
    # -------------------------
    print("[INFO] Splitting data into train/val...")
    rna_train, rna_val, img_train, img_val = train_test_split(
        rna_data_filtered, image_filenames, test_size=0.2, random_state=args.seed
    )
    # Convert back to DataFrame to keep columns
    rna_train = pd.DataFrame(rna_train, columns=rna_data_filtered.columns)
    rna_val = pd.DataFrame(rna_val, columns=rna_data_filtered.columns)

    print("[INFO] Scaling RNA features...")
    scaler = StandardScaler()
    rna_train_scaled = pd.DataFrame(scaler.fit_transform(rna_train), columns=rna_data_filtered.columns)
    rna_val_scaled = pd.DataFrame(scaler.transform(rna_val), columns=rna_data_filtered.columns)

    # -------------------------
    # 6. Calculate mean/std for image normalization
    # -------------------------
    print("[INFO] Calculating mean/std for images (sample-based)...")
    pre_transform_albu = A.Compose([
        A.Resize(224, 224, interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=(0,0,0), std=(1,1,1)),
        ToTensorV2()
    ])
    temp_dataset = RNACustomDataset(
        rna_train_scaled, img_train, args.image_dir,
        transform=pre_transform_albu, transform_type="albu"
    )
    calc_mean, calc_std = calc_mean_std(temp_dataset, num_samples=min(50, len(temp_dataset)))
    print(f"[INFO] Computed mean: {calc_mean}, std: {calc_std}")

    # -------------------------
    # 7. Define Augmentations
    # -------------------------
    print("[INFO] Defining train/val transformations...")
    train_transform_albu = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=1.0),
        A.GaussNoise(p=0.5),
        A.OneOf([A.CLAHE(p=1.0), A.RandomGamma(p=1.0)], p=0.9),
        A.OneOf([A.Sharpen(p=1.0), A.Blur(3, p=1.0), A.MotionBlur(3, p=1.0)], p=0.9),
        A.OneOf([A.RandomBrightnessContrast(p=1.0), A.HueSaturationValue(p=1.0)], p=0.9),
        A.Resize(224, 224, interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=calc_mean, std=calc_std),
        ToTensorV2()
    ])

    val_transform_albu = A.Compose([
        A.Resize(224, 224, interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=calc_mean, std=calc_std),
        ToTensorV2()
    ])

    # -------------------------
    # 8. Create Datasets & Loaders
    # -------------------------
    print("[INFO] Creating train/val datasets and dataloaders...")
    train_dataset = RNACustomDataset(
        rna_train_scaled, img_train, args.image_dir,
        transform=train_transform_albu, transform_type="albu",
        negative_ratio=args.negative_ratio
    )
    val_dataset = RNACustomDataset(
        rna_val_scaled, img_val, args.image_dir,
        transform=val_transform_albu, transform_type="albu",
        negative_ratio=args.negative_ratio
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # -------------------------
    # 9. Set up MLflow
    # -------------------------
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    # -------------------------
    # 10. Define Hyperparam Grid
    # -------------------------
    image_model_choices = [
        "cnn2", "cnn3", "cnn4",
        "resnet18", "resnet34", "resnet50",
        "vit_small_a", "vit_small_b"
    ]
    embedding_dims = [128]  # example list, could add more
    margins = [1.0]         # e.g., could try [0.5, 1.0, 1.5]
    learning_rates = [1e-4] # e.g., could try [1e-4, 1e-3]

    # Best across all hyperparam combos
    global_best_val_loss = float("inf")
    global_best_config = None

    # -------------------------
    # 11. Nested loops over hyperparams
    # -------------------------
    for model_choice in image_model_choices:
        for emb_dim in embedding_dims:
            for margin in margins:
                for lr in learning_rates:
                    print("=" * 70)
                    print(f"[INFO] Training with: "
                          f"model={model_choice}, emb_dim={emb_dim}, margin={margin}, lr={lr}")
                    print("=" * 70)

                    # Create RNA & image encoder
                    rna_encoder = RNAEncoder(input_dim=len(rna_cols.columns), embedding_dim=emb_dim)

                    try:
                        image_encoder = create_image_encoder(model_choice, embedding_dim=emb_dim)
                    except ValueError as e:
                        print(f"[WARNING] Skipping {model_choice} due to: {e}")
                        continue

                    # Combine into CLIPModel
                    model = CLIPModel(rna_encoder, image_encoder).to(device)

                    # Define optimizer & contrastive loss
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    criterion = ContrastiveLoss(margin=margin)

                    # Learning rate scheduler (ReduceLROnPlateau)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        factor=0.5,
                        patience=2,
                        verbose=True,
                        min_lr=1e-7
                    )

                    # Early Stopping
                    early_stopping_patience = 5
                    no_improvement_count = 0
                    best_val_loss = float("inf")

                    # ---------------------
                    # 12. MLflow Run
                    # ---------------------
                    run_name = f"{model_choice}-{emb_dim}-{margin}-{lr}"
                    with mlflow.start_run(run_name=run_name):
                        mlflow.log_params({
                            "image_model_choice": model_choice,
                            "embedding_dim": emb_dim,
                            "margin": margin,
                            "learning_rate": lr,
                            "epochs": args.epochs,
                            "early_stopping_patience": early_stopping_patience,
                            "reduce_on_plateau_patience": 2,
                            "reduce_on_plateau_factor": 0.5,
                            "reduce_on_plateau_min_lr": 1e-7
                        })

                        # Epoch loop
                        for epoch in range(args.epochs):
                            # ---- Train One Epoch ----
                            train_loss = train_one_epoch(
                                model, train_loader, criterion, optimizer, device
                            )
                            # ---- Validate ----
                            val_loss = validate_one_epoch(
                                model, val_loader, criterion, device
                            )

                            # Log to MLflow
                            mlflow.log_metric("train_loss", train_loss, step=epoch)
                            mlflow.log_metric("val_loss", val_loss, step=epoch)

                            # Log GPU usage if available
                            if torch.cuda.is_available() and GPUtil_available:
                                for i, gpu in enumerate(GPUtil.getGPUs()):
                                    mlflow.log_metric(f"gpu_load_{i}", gpu.load, step=epoch)
                                    mlflow.log_metric(f"gpu_mem_used_{i}", gpu.memoryUsed, step=epoch)
                                    mlflow.log_metric(f"gpu_mem_total_{i}", gpu.memoryTotal, step=epoch)

                            # Update LR scheduler using val loss
                            scheduler.step(val_loss)

                            print(f"Epoch [{epoch+1}/{args.epochs}] -> "
                                  f"Train Loss: {train_loss:.4f}, "
                                  f"Val Loss: {val_loss:.4f}")

                            # Early stopping check
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                no_improvement_count = 0
                            else:
                                no_improvement_count += 1

                            if no_improvement_count >= early_stopping_patience:
                                print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
                                break

                        # Compare with global best
                        if best_val_loss < global_best_val_loss:
                            global_best_val_loss = best_val_loss
                            global_best_config = (model_choice, emb_dim, margin, lr)

                        # Log final model as an artifact
                        mlflow.pytorch.log_model(model, artifact_path="model")

    # -------------------------
    # 13. Print Final Results
    # -------------------------
    print("\n================ FINAL RESULTS ================")
    print(f"Lowest val_loss = {global_best_val_loss:.4f} with config:")
    print(f"  model={global_best_config[0]}")
    print(f"  emb_dim={global_best_config[1]}")
    print(f"  margin={global_best_config[2]}")
    print(f"  learning_rate={global_best_config[3]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CLIP-like RNA+Image model with manual hyperparam loops."
    )
    parser.add_argument("--rna_csv", type=str, default="RNA_counts.csv",
                        help="Path to the RNA CSV file.")
    parser.add_argument("--metadata_xlsx", type=str, default="metadata.xlsx",
                        help="Path to the Excel file containing metadata.")
    parser.add_argument("--image_dir", type=str, default="path/to/images",
                        help="Directory containing .nii.gz image files.")
    parser.add_argument("--mlflow_uri", type=str, default="mlruns",
                        help="MLflow tracking URI.")
    parser.add_argument("--mlflow_experiment", type=str, default="RNA-Image-CLIP-ManualLoops",
                        help="Name of the MLflow experiment.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size.")
    parser.add_argument("--max_pairs", type=int, default=100,
                        help="Max number of RNA-image pairs to match.")
    parser.add_argument("--negative_ratio", type=float, default=1.0,
                        help="Negative ratio for RNACustomDataset.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs for each hyperparam combination.")
    args = parser.parse_args()

    main(args)
