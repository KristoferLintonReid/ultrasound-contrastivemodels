# RNA & Image CLIP Model

This repository contains code for training a **CLIP-like model** that aligns **RNA-Seq data** with **medical images** (in `.nii.gz` format). The approach leverages contrastive learning to ensure that matching RNA and image embeddings are pulled closer together while non-matching pairs are pushed apart. Then the embeddings are used for downstream tasks such as classification of benign or malignant lesions.

![Overview of the RNA+Image CLIP pipeline](figures/Overiview.jpeg)

## Repository Structure
```
. ├── data/ │ └── ... # (Optional) Local data files or samples ├── etc/ │ └── requirements.txt # Pinned dependencies ├── figures/ │ └── Overiview.jpeg # Diagram or overview image ├── models/ │ └── ... # Model definition files (encoders, etc.) ├── notebooks/ │ └── ... # Jupyter notebooks for experimentation ├── scripts/ │ ├── train.py # Main training script │ └── validate.py # Validation/testing script └── src/ ├── dataset.py # Custom dataset & transformations ├── losses.py # Contrastive loss definitions ├── model.py # Encoders + CLIP model ├── utils.py # Utility functions (training loops, etc.) └── ...
```
   
  ## Setup & Installation
  
  ### 1. Clone the Repository
  
  ```bash
  git clone https://github.com/YourUsername/ultrasound-contrastivemodels.git
  cd ultrasound-contrastivemodels
