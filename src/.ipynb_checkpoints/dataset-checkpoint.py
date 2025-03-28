import os
import random
import torch
import pandas as pd
import nibabel as nib
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class RNACustomDataset(Dataset):
    """
    Flexible RNA+Image dataset.

    If `use_surgical_label=False` (default), it behaves like the original contrastive dataset:
      - Generates positive/negative pairs
      - Yields (rna_feats, image, contrastive_label) with label=1 for positive, 0 for negative

    If `use_surgical_label=True`, it assumes `rna_data` has a "SURGICAL" or "label" column
      (e.g. 0 => "B-S", 1 => "M-S"),
      bypasses negative pair generation,
      and yields (rna_feats, image, classification_label).
    """
    def __init__(
        self, 
        rna_data, 
        image_filenames, 
        image_directory, 
        transform=None, 
        transform_type="", 
        negative_ratio=1.0,
        use_surgical_label=False
    ):
        """
        :param rna_data: DataFrame of RNA features (#samples x #genes).
                         If `use_surgical_label=True`, `rna_data` should have 
                         a column for classification label (like "SURGICAL").
        :param image_filenames: List of image filenames (one per sample).
        :param image_directory: Path to directory containing .nii.gz images.
        :param transform: Transform pipeline for images (albumentations or torchvision).
        :param transform_type: "albu" for albumentations; otherwise Torchvision.
        :param negative_ratio: Ratio of negatives for contrastive mode.
        :param use_surgical_label: If True, ignore negative pairs and use 
                                   the "SURGICAL" or "label" column for classification.
        """
        super().__init__()
        self.rna_data = rna_data.reset_index(drop=True)  # ensure consistent indexing
        self.image_filenames = image_filenames
        self.image_directory = image_directory
        self.transform = transform
        self.transform_type = transform_type
        self.negative_ratio = negative_ratio
        self.use_surgical_label = use_surgical_label

        if self.use_surgical_label:
            # We assume there's a "SURGICAL" or some label column. 
            # Example: "SURGICAL" -> "B-S" or "M-S" 
            # or numeric 0/1 is stored under "label".
            # For example:
            # self.labels = self.rna_data["SURGICAL"].map({"B-S":0, "M-S":1}).values
            # But you can adapt as needed. We'll do something generic:
            self.labels = self.rna_data["SURGICAL "].values  # or "label" if you renamed it
            # Remove the label column from the actual rna features so only numeric columns remain
            self.rna_data = self.rna_data.drop(columns=["SURGICAL "], errors="ignore")

            # For classification mode, each row => 1 sample. 
            self.indices = np.arange(len(self.rna_data))
        else:
            # Original contrastive approach: build positive/negative pairs
            self.pairs = self._generate_pairs()

    def _generate_pairs(self):
        """
        Builds a list of tuples: (rna_index, img_index, label),
        where label=1 => same sample (positive), label=0 => different (negative).
        """
        pairs = []
        num_samples = len(self.rna_data)

        # Positive pairs
        for idx in range(num_samples):
            pairs.append((idx, idx, 1))

        # Negative pairs
        num_negatives = int(len(pairs) * self.negative_ratio)
        for _ in range(num_negatives):
            rna_idx = random.randint(0, num_samples - 1)
            img_idx = random.randint(0, num_samples - 1)
            while img_idx == rna_idx:
                img_idx = random.randint(0, num_samples - 1)
            pairs.append((rna_idx, img_idx, 0))

        random.shuffle(pairs)
        return pairs

    def __len__(self):
        if self.use_surgical_label:
            return len(self.indices)
        else:
            return len(self.pairs)

    def __getitem__(self, idx):
        """
        If use_surgical_label=True => returns (rna_feats, image, classification_label)
        Else => returns (rna_feats, image, contrastive_label)
        """
        if self.use_surgical_label:
            # Classification mode
            real_idx = self.indices[idx]
            label = self.labels[real_idx]

            # RNA feats
            rna_feats = torch.tensor(
                self.rna_data.iloc[real_idx].values.astype(float),
                dtype=torch.float
            )
            # Image
            img_path = os.path.join(self.image_directory, self.image_filenames[real_idx])
            image = self._load_image(img_path)
            image = self._apply_transforms(image)

            return rna_feats, image, torch.tensor(label, dtype=torch.float)

        else:
            # Contrastive mode
            rna_idx, img_idx, contrastive_label = self.pairs[idx]
            rna_feats = torch.tensor(
                self.rna_data.iloc[rna_idx].values.astype(float),
                dtype=torch.float
            )
            img_path = os.path.join(self.image_directory, self.image_filenames[img_idx])
            image = self._load_image(img_path)
            image = self._apply_transforms(image)

            return rna_feats, image, torch.tensor(contrastive_label, dtype=torch.float)

    def _load_image(self, img_path):
        """
        Loads the .nii.gz image from disk, converts it to a 3-channel array,
        and returns a PIL Image.
        """
        nii_image = nib.load(img_path)
        image_slice = nii_image.get_fdata()

        # Force 3 channels
        image_slice_3ch = np.stack([image_slice]*3, axis=-1).astype(np.uint8)
        pil_img = Image.fromarray(image_slice_3ch)
        return pil_img

    def _apply_transforms(self, pil_img):
        """
        Applies either Albumentations or Torchvision transforms, if specified.
        """
        if self.transform:
            if self.transform_type == "albu":
                # Albumentations
                image_rgb = pil_img.convert("RGB")
                image_np = np.array(image_rgb)
                transformed = self.transform(image=image_np)
                return transformed["image"]
            else:
                # Torchvision
                return self.transform(pil_img)
        else:
            return pil_img


class ImgCustomDataset(Dataset):
    """
    Dataset providing only image data, typically for
    calculating dataset stats (mean/std) or tasks where no pairs/labels needed.
    """
    def __init__(
        self, 
        image_filenames, 
        image_directory, 
        transform=None, 
        transform_type=""
    ):
        super().__init__()
        self.image_filenames = image_filenames
        self.image_directory = image_directory
        self.transform = transform
        self.transform_type = transform_type

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_directory, self.image_filenames[idx])
        nii_image = nib.load(img_path)
        image_slice = nii_image.get_fdata()
        # Force 3 channels
        image_slice_3ch = np.stack([image_slice]*3, axis=-1).astype(np.uint8)
        pil_img = Image.fromarray(image_slice_3ch)

        if self.transform:
            if self.transform_type == "albu":
                image_rgb = pil_img.convert("RGB")
                image_np = np.array(image_rgb)
                image = self.transform(image=image_np)["image"]
            else:
                image = self.transform(pil_img)
        else:
            image = pil_img

        return image
