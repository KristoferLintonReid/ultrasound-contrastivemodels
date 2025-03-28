import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# If you want default pretrained weights for ResNet:
from torchvision.models import (
    ResNet18_Weights, 
    ResNet34_Weights,
    ResNet50_Weights
)

###############################################################################
# 1. RNA ENCODER
###############################################################################
class RNAEncoder(nn.Module):
    """
    MLP-based encoder for RNA data.
    """
    def __init__(self, input_dim, hidden_dims=[1024, 512, 512], embedding_dim=128):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.BatchNorm1d(hd))
            layers.append(nn.ReLU())
            prev_dim = hd

        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, input_dim)
        returns: (B, embedding_dim)
        """
        return self.model(x)

###############################################################################
# 2. SIMPLE CNN
###############################################################################
class SimpleCNN(nn.Module):
    """
    A small custom CNN with a variable number of convolution layers.
    """
    def __init__(self, num_layers=2, embedding_dim=128):
        super().__init__()
        layers = []
        in_channels = 3
        out_channels = 32

        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels
            out_channels *= 2

        self.conv_stack = nn.Sequential(*layers)
        # After num_layers of max pool, each dimension (224) is halved num_layers times.
        final_hw = 224 // (2 ** num_layers)  
        self.fc = nn.Linear(in_channels * final_hw * final_hw, embedding_dim)

    def forward(self, x):
        """
        x: (B, 3, 224, 224)
        returns: (B, embedding_dim)
        """
        x = self.conv_stack(x)   # (B, out_channels, H, W)
        x = x.view(x.size(0), -1)
        x = self.fc(x)           # (B, embedding_dim)
        return x

###############################################################################
# 3. RESNET ENCODER
###############################################################################
class ResNetEncoder(nn.Module):
    """
    Wrapper for torchvision ResNet family with final layer mapped to embedding_dim.
    """
    def __init__(self, resnet_type="resnet18", embedding_dim=128):
        super().__init__()
        if resnet_type == "resnet18":
            base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif resnet_type == "resnet34":
            base_model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        elif resnet_type == "resnet50":
            base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        # Replace the final classification layer with a Linear -> embedding_dim
        base_model.fc = nn.Linear(base_model.fc.in_features, embedding_dim)
        self.model = base_model

    def forward(self, x):
        """
        x: (B, 3, 224, 224)
        returns: (B, embedding_dim)
        """
        return self.model(x)

###############################################################################
# 4. SIMPLE ViT
###############################################################################
class SimpleViT(nn.Module):
    """
    A placeholder for small ViT variants.
    """
    def __init__(self, variant="vit_small_a", embedding_dim=128):
        super().__init__()
        if variant == "vit_small_a":
            vit = models.vit_b_16(weights=None)  # or use pretrained if available
        elif variant == "vit_small_b":
            vit = models.vit_l_16(weights=None)  # or use pretrained if available
        else:
            raise ValueError(f"Unknown ViT variant: {variant}")

        # Replace final heads with a Linear -> embedding_dim
        vit.heads = nn.Linear(vit.hidden_dim, embedding_dim)
        self.model = vit

    def forward(self, x):
        """
        x: (B, 3, 224, 224)
        returns: (B, embedding_dim)
        """
        return self.model(x)

###############################################################################
# 5. IMAGE ENCODER FACTORY
###############################################################################
def create_image_encoder(model_choice, embedding_dim=128):
    """
    Factory function to instantiate an image encoder
    based on the specified model choice string.
    """
    if model_choice == "cnn2":
        return SimpleCNN(num_layers=2, embedding_dim=embedding_dim)
    elif model_choice == "cnn3":
        return SimpleCNN(num_layers=3, embedding_dim=embedding_dim)
    elif model_choice == "cnn4":
        return SimpleCNN(num_layers=4, embedding_dim=embedding_dim)
    elif model_choice in ["resnet18", "resnet34", "resnet50"]:
        return ResNetEncoder(resnet_type=model_choice, embedding_dim=embedding_dim)
    elif model_choice in ["vit_small_a", "vit_small_b"]:
        return SimpleViT(variant=model_choice, embedding_dim=embedding_dim)
    else:
        raise ValueError(f"Unsupported image model choice: {model_choice}")

###############################################################################
# 6. CLIP-LIKE MODEL WITH PROJECTION HEAD
###############################################################################
class CLIPModel(nn.Module):
    """
    A CLIP-like model with an RNA encoder + image encoder, plus optional
    projection heads and L2 normalization on the final embeddings.
    """
    def __init__(
        self,
        rna_encoder: nn.Module,
        image_encoder: nn.Module,
        projection_dim=128,
        use_batchnorm=True,
        use_l2norm=True
    ):
        """
        Args:
            rna_encoder (nn.Module): Encodes RNA data to some embedding_dim.
            image_encoder (nn.Module): Encodes images to the same embedding_dim.
            projection_dim (int): Output dimension of the projection head.
            use_batchnorm (bool): Whether to include BatchNorm1d in the projection head.
            use_l2norm (bool): Whether to L2-normalize final embeddings.
        """
        super().__init__()
        self.rna_encoder = rna_encoder
        self.image_encoder = image_encoder
        self.use_l2norm = use_l2norm

        # We assume both encoders output the same dimension, e.g. 128.
        # If you want to be more robust, you can pass that dimension explicitly.
        encoder_out_dim = None

        # Try reading the final linear layer out_features if present:
        # (Not strictly necessary if you know your encoder dimension.)
        if hasattr(rna_encoder, 'model') and isinstance(rna_encoder.model[-1], nn.Linear):
            encoder_out_dim = rna_encoder.model[-1].out_features
        else:
            # Fallback to 128 or a known number
            encoder_out_dim = 128

        # Simple 2-layer MLP projection
        def make_projection_head(in_dim, out_dim):
            layers = [nn.Linear(in_dim, out_dim)]
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(out_dim, out_dim))
            return nn.Sequential(*layers)

        self.rna_projection = make_projection_head(encoder_out_dim, projection_dim)
        self.img_projection = make_projection_head(encoder_out_dim, projection_dim)

    def forward(self, rna_input, image_input):
        """
        Args:
            rna_input: (B, input_dim) RNA features
            image_input: (B, 3, 224, 224) Image tensor
        Returns:
            rna_proj, img_proj: the final projected (and optionally L2-normalized) embeddings
        """
        # 1) Encoder embeddings
        rna_emb = self.rna_encoder(rna_input)    # (B, encoder_out_dim)
        img_emb = self.image_encoder(image_input)  # (B, encoder_out_dim)

        # 2) Projection heads
        rna_proj = self.rna_projection(rna_emb)  # (B, projection_dim)
        img_proj = self.img_projection(img_emb)  # (B, projection_dim)

        # 3) Optional L2 normalization
        if self.use_l2norm:
            rna_proj = F.normalize(rna_proj, dim=-1)
            img_proj = F.normalize(img_proj, dim=-1)

        return rna_proj, img_proj
