import torch 
import numpy as np
import random
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import randomname
from random import randint
import pandas as pd

# MLflow with Optuna: Hyperparameter Optimization and Tracking
# https://medium.com/swlh/pytorch-mlflow-optuna-experiment-tracking-and-hyperparameter-optimization-132778d6defc

def suggest_hyperparameters(trial):
    
    # Batch size 
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    return batch_size

def set_seed(random_seed):
    
    # Set random seed for NumPy
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Set random seed for PyTorch
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(random_seed)

    print(f"Random seed set as {random_seed}")

# Compute Mean and Standard Deviation for Normalisation
# https://www.kaggle.com/code/subhajeetdas/find-mean-and-std-for-image-normalization-pytorch
# https://www.kaggle.com/code/kozodoi/mean-and-std-of-pet-photos
# https://gist.github.com/spirosdim/79fc88231fffec347f1ad5d14a36b5a8
# https://kozodoi.me/blog/20210308/compute-image-stats

def calc_mean_std(img_dataset, num_imgs):

    # Load images
    img_loader = DataLoader(img_dataset, batch_size=256, shuffle=False, num_workers=0)

    # Placeholders
    sum_pixels = torch.tensor([0.0, 0.0, 0.0])
    sum_pixels_sq = torch.tensor([0.0, 0.0, 0.0])

    # Loop through images
    for images in img_loader:
        sum_pixels += images.sum(axis=[0, 2, 3])
        sum_pixels_sq += (images**2).sum(axis=[0, 2, 3])

    # Calculate mean and standard deviation
    img_size = 224
    pixel_count = num_imgs * img_size * img_size
    mean = sum_pixels/pixel_count
    std = torch.sqrt((sum_pixels_sq/pixel_count)-(mean**2))
 
    return mean.tolist(), std.tolist()

# Random model name function similar to MLFlow
def generate_model_name():
    # Create random name 
    name = randomname.get_name(adj=("sound", "appearance", "emotions"), noun=("cats", "apex_predators", "dogs", "birds", "fish"))
    num = randint(100, 999)

    # Join name
    model_name = str(name) + "-" + str(num) + ".pt"

    return model_name

def capitalize_j(text):
    """Ensure 'j' is always capitalized in the string without affecting other characters."""
    return text.replace('j', 'J') if text else text

def format_case_id(case_val):
    """
    Convert the case ID value to a properly formatted string.
    If the value is numeric and represents an integer, convert it to int first
    to avoid a trailing '.0'. Otherwise, return the stripped string.
    """
    if pd.isna(case_val):
        return ''
    try:
        # Try converting to a float first
        num = float(case_val)
        # If the number is an integer value, return it as an integer string.
        if num.is_integer():
            return str(int(num))
        else:
            return str(case_val)
    except Exception:
        # If conversion fails, treat it as a string.
        return str(case_val).strip()