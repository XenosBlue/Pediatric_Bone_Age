#%% Imports

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


#%%  Model


class ResNet50_Regressor(nn.Module):

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()


        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = torchvision.models.resnet50(weights=weights)

        feat_dim = base.fc.in_features
        self.ftr_xtr =  nn.Sequential(*list(base.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = mlp.MLP(feat_dim, feat_dim, dropout)

    def forward(self, x):
        emb = self.ftr_xtr(x)
        emb = self.avg_pool(emb).flatten(1)
        out = self.fc(emb).squeeze(1)
        return out
    

#%% Local Test

if __name__ == "__main__": 
    from torchinfo import summary
    model = ResNet50_Regressor()
    print(summary(model, (1, 3, 224, 224)))




# %%
