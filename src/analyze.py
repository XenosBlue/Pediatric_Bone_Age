#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import os

#%% Definitions
def plot_losses(train_losses, val_losses, save_path=None,  title="Loss vs. Epoch"):
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Val")

    # plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch vs. Loss")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

#%%

#%% Analysis 

def evaluate_model(model,
                   loader,
                   mean= 0.,
                   std = 1.,
                   save_path = None):
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    all_y, all_p = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            y = labels["boneage"].to(device, non_blocking=True).float().view(-1)
            p = model(imgs).view(-1)

            y = y.detach().cpu().numpy()
            p = p.detach().cpu().numpy()

            # unnormalize
            y = y * std + mean
            p = p * std + mean

            all_y.append(y)
            all_p.append(p)

    y = np.concatenate(all_y, axis=0)
    p = np.concatenate(all_p, axis=0)

    err = p - y
    total_abs_error = np.abs(err).sum()
    mse = np.mean(err ** 2)

    print(f"Samples: {len(y)}")
    print(f"Total Absolute Error: {total_abs_error:.3f}")
    print(f"MSE: {mse:.3f}")


    sns.set_theme(style="whitegrid", context="talk")

    df = pd.DataFrame({"age": y, "err": err}).sort_values("age").reset_index(drop=True)


    win = max(15, len(df) // 30)
    roll = df["err"].rolling(window=win, center=True, min_periods=max(5, win // 3))
    df["err_mean"] = roll.mean()
    df["err_std"] = roll.std()

    if len(df) >= 2:
        coef = np.polyfit(df["age"].values, df["err"].values, 1)
        line_x = np.linspace(df["age"].min(), df["age"].max(), 200)
        line_y = np.polyval(coef, line_x)
    else:
        line_x, line_y = np.array([]), np.array([])

    plt.figure(figsize=(11, 7))
    ax = sns.scatterplot(
        data=df, x="age", y="err",
        s=22, alpha=0.45, edgecolor=None
    )

    sns.lineplot(
        data=df, x="age", y="err_mean",
        linewidth=2.2, ax=ax, label="Rolling mean (error)"
    )

    mean_vals = df["err_mean"].to_numpy()
    std_vals = df["err_std"].to_numpy()
    ax.fill_between(
        df["age"].to_numpy(),
        (mean_vals - std_vals),
        (mean_vals + std_vals),
        alpha=0.18, label="±1 SD"
    )

    if line_x.size:
        ax.plot(line_x, line_y, linestyle="--", linewidth=1.6, label="Linear trend")

    ax.axhline(0.0, ls=":", lw=1.5, color="black", label="Zero error")
    ax.set_xlabel("True Bone Age")
    ax.set_ylabel("Prediction Error (Pred − True)")
    ax.set_title("Error Spread Across Age with Trend and ±1 SD")
    ax.legend(frameon=True)
    plt.tight_layout()

    if save_path is None:
        plt.show()
        plt.close()
        return

    out_path = save_path 
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")
    plt.close()

    return {"total_abs_error": total_abs_error, "mse": mse, "n": len(y)}



#%% Local Test

# from torch.utils.data import DataLoader
# import torchvision

# import sys
# sys.path.append("../src")
# import dataloader

# sys.path.append("..")
# from archs import *


# ROOT_DIR = ".."
# NAME = "Nikhil"
# EXP_NAME = "ResNet50_50_50_epochs_exp1"

# #Data 
# TRAIN_CSV = os.path.join(ROOT_DIR, "data", "train.csv")
# TRAIN_DIR = os.path.join(ROOT_DIR, "data", "boneage-training-dataset")

# VAL_CSV = os.path.join(ROOT_DIR, "data", "Bone Age Validation Set", "Validation Dataset.csv")
# VAL_DIR = os.path.join(ROOT_DIR, "data", "Bone Age Validation Set","boneage-validation-dataset-1")
# EXP_DIR = os.path.join(ROOT_DIR, "experiments", NAME, EXP_NAME)
# CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")

# BONEAGE_MEAN = 132.0
# BONEAGE_STD  = 41.182
# IMG_SIZE = 224
# BATCH_SIZE = 32
# NUM_WORKERS = 4
# PIN_MEM = True
# TRANSFORM = torchvision.models.ResNet50_Weights.DEFAULT.transforms()
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ckpt = "/mnt/c/Users/cnikh/Projects/dl_proj/Pediatric_Bone_Age/checkpoints/ResNet50_50_50_epochs_exp1/model_fc_epoch.pth"
# ckpt = "/mnt/c/Users/cnikh/Projects/dl_proj/Pediatric_Bone_Age/checkpoints/ResNet50_exp0/model_fc_epoch.pth"
# ckpt = "/mnt/c/Users/cnikh/Projects/dl_proj/Pediatric_Bone_Age/checkpoints/ResNet50_exp0/model_finetune_epoch.pth"

# model = torch.load(ckpt)

# val_dataset = dataloader.BoneAgeDataset(
#         csv_path=VAL_CSV,
#         img_dir = VAL_DIR,
#         transform = TRANSFORM,
#         image_size = IMG_SIZE,
#         drop_missing = True)

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=NUM_WORKERS,
#     pin_memory=PIN_MEM,
#     persistent_workers=(NUM_WORKERS > 0),
#     prefetch_factor=2 if NUM_WORKERS > 0 else None,
# )


# _ = evaluate_model(
#     model,
#     val_loader,
#     mean=BONEAGE_MEAN,
#     std=BONEAGE_STD,
#     # save_path=os.path.join(EXP_DIR, "error_vs_age.png"),
# )


# %%
