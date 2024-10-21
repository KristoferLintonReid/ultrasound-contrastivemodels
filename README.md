
# RNA and Image CLIP Model

This repository contains code for training a CLIP-like model that uses RNA-Seq data and medical image data in `.nii.gz` format. The model uses a contrastive learning approach to align embeddings of RNA data with corresponding medical images. 

## Repository Structure

- **data/**: Contains sample data files.
- **models/**: Contains the model definition files and the encoders for RNA and Image.
- **src/**: Contains the source code for the dataset preparation, training, and validation.
- **notebooks/**: Jupyter notebooks for experimentation and testing.
- **scripts/**: Scripts for running the training and evaluation.

## Usage

### 1. Set Up Environment
To install the necessary dependencies:
```
pip install -r requirements.txt
```

### 2. Training the Model
To train the CLIP model:
```
python scripts/train.py
```

### 3. Validation
To validate the model on the test set:
```
python scripts/validate.py
```

## Model Architecture
- **RNAEncoder**: A multi-layer perceptron for embedding RNA-Seq data.
- **ImageEncoder**: A CNN (ResNet-50) for embedding `.nii.gz` medical images.
- **CLIPModel**: A combined model that processes RNA and image data to generate embeddings for contrastive learning.
