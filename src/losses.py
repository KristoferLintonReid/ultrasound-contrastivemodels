import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss using Cosine Similarity. margin=1.0 typically works well.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, rna_embeddings, image_embeddings, labels):
        similarities = self.cosine_similarity(rna_embeddings, image_embeddings)
        # Positive pairs => (1 - similarity)
        positive_loss = labels * (1 - similarities)
        # Negative pairs => relu(similarity - margin)
        negative_loss = (1 - labels) * F.relu(similarities - self.margin)
        loss = torch.mean(positive_loss + negative_loss)
        return loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss using Cosine Similarity. margin=1.0 typically works well.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, rna_embeddings, image_embeddings, labels):
        similarities = self.cosine_similarity(rna_embeddings, image_embeddings)
        # Positive pairs => (1 - similarity)
        positive_loss = labels * (1 - similarities)
        # Negative pairs => relu(similarity - margin)
        negative_loss = (1 - labels) * F.relu(similarities - self.margin)
        loss = torch.mean(positive_loss + negative_loss)
        return loss
