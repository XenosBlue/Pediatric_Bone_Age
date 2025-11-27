#%%
# 
import torch
import torchvision
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)



#%% Local Test

if __name__ == "__main__": 
    from torchinfo import summary
    model = MLP(356)
    print(summary(model, (1, 356)))

#%%