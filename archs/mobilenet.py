#%%

import torch
import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

try:
    import sys
    sys.path.append("../archs")
    import archs.mlp as mlp
except:
    import mlp


class MobileNetV2_Regressor(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        sex_emb_dim: int = 4,
    ):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = mobilenet_v2(weights=weights)

        feat_dim = base.classifier[1].in_features
        self.ftr_xtr = base.features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.sex_embed = nn.Embedding(num_embeddings=2, embedding_dim=sex_emb_dim)
        self.fc = mlp.MLP(feat_dim + sex_emb_dim, feat_dim, dropout)

        if freeze_backbone:
            for p in self.ftr_xtr.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor, sex: torch.Tensor):
        emb = self.ftr_xtr(x)
        emb = self.avg_pool(emb).flatten(1)

        sex = sex.long().view(-1)
        sex_emb = self.sex_embed(sex)

        emb = torch.cat([emb, sex_emb], dim=1)
        return self.fc(emb).squeeze(1)
# %%