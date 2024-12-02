import os
import torch  # Add this import
import pandas as pd
import nibabel as nib
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class RNACustomDataset(Dataset):
    def __init__(self, rna_data, image_filenames, image_directory, transform=None, transform_type=""):
        self.rna_data = rna_data
        self.image_filenames = image_filenames
        self.image_directory = image_directory  # Add image_directory as a class attribute
        self.transform = transform
        self.transform_type = transform_type
    
    def __len__(self):
        return len(self.rna_data)
    
    def __getitem__(self, idx):
        # RNA Data
        rna_sample = torch.tensor(self.rna_data.iloc[idx].values.astype(float)).float()
        
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
        
        return rna_sample, image
