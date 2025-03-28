# RNA & Image Contrastive Model

This repository contains code for training a **contrastive** that aligns **RNA-Seq data** with **medical images** (in `.nii.gz` format). The approach leverages contrastive learning to ensure that matching RNA and image embeddings are pulled closer together while non-matching pairs are pushed apart. Then the embeddings are used for downstream task. For example classification of benign or malignant lesions.

# Example of contrastive model

![Short description of the figure](https://raw.githubusercontent.com/KristoferLintonReid/ultrasound-contrastivemodels/klr/dev/figures/Slide2.jpg)


## Repository Structure
```
. ├── data/ │ └── ... # (Optional) Local data files or samples ├── etc/ │ └── requirements.txt # Pinned dependencies ├── figures/ │ └── Overiview.jpeg # Diagram or overview image ├── models/ │ └── ... # Model definition files (encoders, etc.) ├── notebooks/ │ └── ... # Jupyter notebooks for experimentation ├── scripts/ │ ├── train.py # Main training script │ └── validate.py # Validation/testing script └── src/ ├── dataset.py # Custom dataset & transformations ├── losses.py # Contrastive loss definitions ├── model.py # Encoders + CLIP model ├── utils.py # Utility functions (training loops, etc.) └── ...
```
   
## Setup & Installation
  
### 1. Clone the Repository
  
  bash
  git clone https://github.com/YourUsername/ultrasound-contrastivemodels.git
  cd ultrasound-contrastivemodels
 
### 2. Create (Optional) and Activate a Python Environment

You can use conda or virtualenv. For example, with conda:

   conda create -n cliprna-env python=3.9
   conda activate cliprna-env

### 3. Install Dependencies

We recommend using the pinned requirements in etc/requirements.txt:

   pip install -r etc/requirements.txt

This installs all necessary packages, including PyTorch, albumentations, nibabel, etc.
Data Preparation

    RNA-Seq CSV: A CSV file with raw counts (or normalized counts).

    Metadata Excel: Contains sample IDs linking RNA rows to image filenames.

    Image Folder: Contains .nii.gz medical images (e.g., ultrasound scans).

Make sure to update the paths in the scripts or command-line arguments as needed.
Training the Model

Run the main training script:
   
   python scripts/train.py \
     --rna_csv path/to/RNA_counts.csv \
     --metadata_xlsx path/to/metadata.xlsx \
     --image_dir path/to/nii_gz_images \
     --epochs 5 \
     --batch_size 32

Key Arguments:

       --rna_csv: Path to the CSV containing RNA data.
   
       --metadata_xlsx: Path to the Excel file linking sample IDs.
   
       --image_dir: Directory with .nii.gz image files.
   
       --epochs: Number of training epochs.
   
       --batch_size: Batch size (default 32).

Additional arguments are available (e.g., --max_pairs, --negative_ratio, etc.). See scripts/train.py --help for more details.
Validation

After training, you can validate (or test) the model using a similar approach:

   python scripts/validate.py \
     --rna_csv path/to/RNA_counts_val.csv \
     --metadata_xlsx path/to/metadata_val.xlsx \
     --image_dir path/to/nii_gz_images

Model Architecture

    RNAEncoder:

        A multi-layer perceptron (MLP) with optional BatchNorm and ReLU activation for embedding RNA-Seq data into a latent vector.

    ImageEncoder:

        Supports multiple backbones, such as small CNNs or ResNets/Vision Transformers, to embed .nii.gz images into a latent vector.

    CLIPModel:

        A dual-tower model that processes RNA and image embeddings separately and aligns them via a contrastive loss.

Additional Notes

    MLflow Integration: The training script logs metrics, parameters, and artifacts (trained models) to MLflow. By default, it uses a local mlruns/ folder as the tracking URI.

    GPU Usage Logging: If GPUtil is installed, the script logs GPU utilization metrics to MLflow at each epoch.

    Early Stopping & Scheduler: We use a ReduceLROnPlateau scheduler and early stopping to reduce the learning rate or terminate training if validation loss plateaus.

Contributing

    Fork this repo and clone your fork locally.

    Create a feature branch (e.g., git checkout -b feature/my-new-feature).

    Commit your changes and push to your fork.

    Open a Pull Request against main in the original repository.

License
Apache License 2.0
