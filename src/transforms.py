import random
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T


# ---------- Custom transforms ----------


class FixedCLAHE(nn.Module):
    """
    Apply CLAHE with fixed clip limit and tile size.
    Works on PIL.Image (grayscale or RGB).
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def forward(self, img: Image.Image) -> Image.Image:
        np_img = np.array(img)

        if np_img.ndim == 3:  # H,W,C
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_grid_size, self.tile_grid_size),
        )
        np_img = clahe.apply(np_img)

        return Image.fromarray(np_img)


class RandomCLAHE(nn.Module):
    """
    Apply CLAHE with random clip limit [1,4]
    and random tile size in {2,4,8,16,32}.
    Works on PIL.Image (grayscale or RGB).
    """
    def __init__(self):
        super().__init__()

    def forward(self, img: Image.Image) -> Image.Image:
        # PIL -> numpy (H,W) or (H,W,3)
        np_img = np.array(img)

        # ensure single channel for CLAHE
        if np_img.ndim == 3:  # H,W,C
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        # random params
        clip = random.uniform(1.0, 4.0)
        tile_power = random.randint(1, 5)   # 2^1..2^5 = 2..32
        tile = 2 ** tile_power

        clahe = cv2.createCLAHE(clipLimit=clip,
                                tileGridSize=(tile, tile))
        np_img = clahe.apply(np_img)

        # back to PIL (still grayscale)
        return Image.fromarray(np_img)


class AddGaussianNoise(nn.Module):
    """
    Additive Gaussian noise on tensor image (C,H,W).
    """
    def __init__(self, mean=0.0, std=0.02):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std == 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

#%%

IMG_SIZE = 224

TRAIN_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    RandomCLAHE(),
    T.Grayscale(num_output_channels=3),
    T.RandomAffine(
        degrees=10,                 # rotation
        translate=(0.05, 0.05),     # up to 5% shift
        scale=(0.9, 1.1),           # zoom in/out
    ),
    T.RandomApply([T.RandomPosterize(bits=4)], p=0.4),
    T.RandomApply([T.RandomEqualize()], p=0.3),
    T.RandomApply([T.ColorJitter(contrast=0.2)], p=0.7),
    T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2.0)], p=0.5),
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
    T.ToTensor(),
    AddGaussianNoise(mean=0.0, std=0.02),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

VAL_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    FixedCLAHE(clip_limit=2.0, tile_grid_size=8),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

