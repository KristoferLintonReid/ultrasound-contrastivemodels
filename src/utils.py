# src/utils.py

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

def set_seed(seed: int):
    """
    Set Python, NumPy, and PyTorch seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calc_mean_std(dataset, num_samples=50):
    """
    Calculate mean and std for a subset of the dataset's images.
    ...
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    mean = 0.0
    std = 0.0
    count = 0

    for i, (_, img, _) in enumerate(loader):
        img = img[0]
        mean += img.mean(dim=(1, 2))
        std += img.std(dim=(1, 2))
        count += 1
        if count >= num_samples:
            break

    mean /= count
    std /= count
    return mean.tolist(), std.tolist()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    ...
    """
    model.train()
    running_loss = 0.0

    for rna_batch, image_batch, labels in dataloader:
        rna_batch = rna_batch.to(device)
        image_batch = image_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        rna_embeddings, image_embeddings = model(rna_batch, image_batch)
        loss = criterion(rna_embeddings, image_embeddings, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Validates the model for one epoch.
    ...
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for rna_batch, image_batch, labels in dataloader:
            rna_batch = rna_batch.to(device)
            image_batch = image_batch.to(device)
            labels = labels.to(device)

            rna_embeddings, image_embeddings = model(rna_batch, image_batch)
            loss = criterion(rna_embeddings, image_embeddings, labels)
            running_loss += loss.item()

    return running_loss / len(dataloader)
