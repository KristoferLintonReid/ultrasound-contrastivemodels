
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

class RNAEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=512):
        super(RNAEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
    def forward(self, x):
        return self.model(x)

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet.fc = nn.Linear(resnet.fc.in_features, embedding_dim)
        self.model = resnet
    
    def forward(self, x):
        return self.model(x)

class CLIPModel(nn.Module):
    def __init__(self, rna_encoder, image_encoder):
        super(CLIPModel, self).__init__()
        self.rna_encoder = rna_encoder
        self.image_encoder = image_encoder
    
    def forward(self, rna_input, image_input):
        rna_embeddings = self.rna_encoder(rna_input)
        image_embeddings = self.image_encoder(image_input)
        return rna_embeddings, image_embeddings

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, rna_embeddings, image_embeddings, labels):
        # Compute cosine similarity
        similarities = self.cosine_similarity(rna_embeddings, image_embeddings)
        
        # Contrastive loss
        positive_loss = labels * (1 - similarities)  # For positive pairs
        negative_loss = (1 - labels) * torch.relu(similarities - self.margin)  # For negative pairs

        return torch.mean(positive_loss + negative_loss)