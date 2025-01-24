import os
import torch  # Add this import
import pandas as pd
import nibabel as nib
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

class RNACustomDataset(Dataset):
    def __init__(self, rna_data, image_filenames, image_directory, transform=None, transform_type="", negative_ratio=1.0):
        self.rna_data = rna_data
        self.image_filenames = image_filenames
        self.image_directory = image_directory  # Add image_directory as a class attribute
        self.transform = transform
        self.transform_type = transform_type
        self.negative_ratio = negative_ratio

        # Generate pairs: positive and negative
        self.pairs = self.generate_pairs()

    def generate_pairs(self):
        pairs = []
        num_samples = len(self.rna_data)
        
        # Create positive pairs
        for idx in range(num_samples):
            pairs.append((idx, idx, 1))  # (RNA index, Image index, label 1 for positive)
        
        # Create negative pairs
        num_negatives = int(len(pairs) * self.negative_ratio)
        for _ in range(num_negatives):
            rna_idx = random.randint(0, num_samples - 1)
            img_idx = random.randint(0, num_samples - 1)
            while img_idx == rna_idx:  # Ensure it's a negative pair
                img_idx = random.randint(0, num_samples - 1)
            pairs.append((rna_idx, img_idx, 0))  # (RNA index, Image index, label 0 for negative)

        random.shuffle(pairs)  # Shuffle pairs
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        rna_idx, img_idx, label = self.pairs[idx]

        # RNA Data
        rna_sample = torch.tensor(self.rna_data.iloc[rna_idx].values.astype(float)).float()
        
        # Image Data
        img_path = os.path.join(self.image_directory, self.image_filenames[img_idx])  # Use self.image_directory
        nii_image = nib.load(img_path)
        image_slice = nii_image.get_fdata()
        image_slice = np.stack([image_slice] * 3, axis=-1)  # Convert to 3-channel image
        image = Image.fromarray(np.uint8(image_slice))
        
        if self.transform:
            if self.transform_type == "albu":
                # Convert image from PIL to openCV format
                image_rgb = image.convert("RGB")
                image_open_cv = np.array(image_rgb)
                image = image_open_cv
                image = self.transform(image=image)["image"]
            else:
                image = self.transform(image)
        
        return rna_sample, image, torch.tensor(label, dtype=torch.float)
    
class ImgCustomDataset(Dataset):
    def __init__(self, image_filenames, image_directory, transform=None, transform_type=""):
        self.image_filenames = image_filenames
        self.image_directory = image_directory  # Add image_directory as a class attribute
        self.transform = transform
        self.transform_type = transform_type

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Image Data
        img_path = os.path.join(self.image_directory, self.image_filenames[idx])  # Use self.image_directory
        nii_image = nib.load(img_path)
        image_slice = nii_image.get_fdata()
        image_slice = np.stack([image_slice] * 3, axis=-1)  # Convert to 3-channel image
        image = Image.fromarray(np.uint8(image_slice))
        
        if self.transform:
            if self.transform_type == "albu":
                # Convert image from PIL to openCV format
                image_rgb = image.convert("RGB")
                image_open_cv = np.array(image_rgb)
                image = image_open_cv
                image = self.transform(image=image)["image"]
            else:
                image = self.transform(image)
        
        return image
