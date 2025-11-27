import torch
from PIL import Image
import torchvision
import torch.nn as nn
from torchvision.models import ResNet50_Weights

try:
    import sys
    sys.path.append("../archs")
    import archs.mlp as mlp
except:
    import mlp

class ResNet50_Regressor(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.2, freeze_backbone: bool = False, sex_emb_dim: int = 4):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = torchvision.models.resnet50(weights=weights)
        feat_dim = base.fc.in_features
        self.ftr_xtr = nn.Sequential(*list(base.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sex_embed = nn.Embedding(num_embeddings=2, embedding_dim=sex_emb_dim)
        self.fc = mlp.MLP(feat_dim + sex_emb_dim, feat_dim, dropout)
        if freeze_backbone:
            for p in self.ftr_xtr.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        emb = self.ftr_xtr(x)
        emb = self.avg_pool(emb).flatten(1)
        sex = sex.long().view(-1)
        sex_emb = self.sex_embed(sex)
        emb = torch.cat([emb, sex_emb], dim=1)
        return self.fc(emb).squeeze(1)
