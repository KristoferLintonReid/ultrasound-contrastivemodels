import mlflow
import mlflow.pytorch
import optuna
import torch
import torch.optim as optim
from .model import (
    RNAEncoder, 
    create_image_encoder, 
    CLIPModel, 
    ContrastiveLoss
)
from .utils import set_seed

def train_one_epoch(model, dataloader, criterion, optimizer, device):
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


def objective(trial, train_loader, val_loader, input_dim, device):
    """
    Optuna objective function that:
    1. Chooses hyperparams (learning rate, margin, embedding_dim, encoder type).
    2. Trains a small # epochs.
    3. Logs to MLflow.
    4. Returns validation loss for the trial.
    """
    # Hyperparam suggestions
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    margin = trial.suggest_float("margin", 0.5, 2.0, step=0.1)
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
    image_model_choice = trial.suggest_categorical(
        "image_model_choice", 
        ["cnn2", "cnn3", "cnn4", "resnet18", "resnet34", "resnet50", 
         "vit_small_a", "vit_small_b"]
    )

    # Build encoders
    rna_encoder = RNAEncoder(input_dim=input_dim, embedding_dim=embedding_dim)
    image_encoder = create_image_encoder(image_model_choice, embedding_dim=embedding_dim)
    model = CLIPModel(rna_encoder, image_encoder).to(device)

    # Loss & Optim
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 5  # fewer epochs per trial to speed up search

    # Nested run in MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "trial_number": trial.number,
            "learning_rate": learning_rate,
            "margin": margin,
            "embedding_dim": embedding_dim,
            "image_model_choice": image_model_choice
        })
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate_one_epoch(model, val_loader, criterion, device)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        # Return final val loss to guide Optuna
        return val_loss

def run_optuna_search(
    train_loader, 
    val_loader, 
    input_dim, 
    device, 
    n_trials=10, 
    study_name="clip_study"
):
    """
    Launches Optuna study to find best hyperparams.
    """
    def wrapped_objective(trial):
        return objective(trial, train_loader, val_loader, input_dim, device)

    study = optuna.create_study(direction="minimize", study_name=study_name)
    study.optimize(wrapped_objective, n_trials=n_trials)

    return study
