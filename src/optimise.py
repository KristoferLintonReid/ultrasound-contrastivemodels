# MLflow with Optuna: Hyperparameter Optimization and Tracking
# https://medium.com/swlh/pytorch-mlflow-optuna-experiment-tracking-and-hyperparameter-optimization-132778d6defc
import torch 
import numpy as np
import random
import os

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