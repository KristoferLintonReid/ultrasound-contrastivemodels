#!/usr/bin/env bash
# run_mlflow.sh
#
# Launch MLflow server pointing to the local `mlruns` directory.

# Set your desired server port
MLFLOW_PORT=5000

# The folder with your local mlruns. This will serve both as the backend store
# (where MLflow metadata is stored) and the default artifact root, since MLflow
# can automatically store artifacts under the same path.
MLFLOW_DIRECTORY="/home/kl2418/Documents/Barcroft/CLIPRNA/ultrasound-contrastivemodels/scripts/mlruns"

# Launch MLflow
mlflow server \
  --backend-store-uri "$MLFLOW_DIRECTORY" \
  --default-artifact-root "$MLFLOW_DIRECTORY" \
  --host 0.0.0.0 \
  --port $MLFLOW_PORT
