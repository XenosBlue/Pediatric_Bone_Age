#Import


import torch
import torch.nn as nn
import torch.nn.functional as F

#%% Definitions



class WingLoss(nn.Module):
    def __init__(self, w: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.w = w
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        x = y_true - y_pred
        abs_x = x.abs()

        # make w, eps tensors on the same device/dtype
        w = abs_x.new_tensor(self.w)
        eps = abs_x.new_tensor(self.epsilon)

        C = w - w * torch.log1p(w / eps)
        small = w * torch.log1p(abs_x / eps)
        large = abs_x - C

        loss = torch.where(abs_x < w, small, large)
        return loss.mean()



class CombinedRegressionLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.7, w: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.mse = torch.nn.MSELoss()
        self.wing = WingLoss(w=w, epsilon=epsilon)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss_mse = self.mse(y_pred, y_true)
        loss_wing = self.wing(y_pred, y_true)
        return self.alpha * loss_mse + (1.0 - self.alpha) * loss_wing



#%% Local Test