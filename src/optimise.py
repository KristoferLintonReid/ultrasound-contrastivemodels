# MLflow with Optuna: Hyperparameter Optimization and Tracking
# https://medium.com/swlh/pytorch-mlflow-optuna-experiment-tracking-and-hyperparameter-optimization-132778d6defc

def suggest_hyperparameters(trial):
    
    # Batch size 
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    return batch_size