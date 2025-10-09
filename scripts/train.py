#%% Imports

import os
import csv
import time
import glob
import pandas as pd
from typing import List, Dict, Optional, Callable

import torch
# from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision

os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
matplotlib.use("Agg")

import sys
sys.path.append("../src")
import dataloader
import analyze

sys.path.append("..")
from archs import *






#%% Parameters

ROOT_DIR = ".."
NAME = "Nikhil"
EXP_NAME = "ResNet50_exp0"

#Data 
TRAIN_CSV = os.path.join(ROOT_DIR, "data", "train.csv")
TRAIN_DIR = os.path.join(ROOT_DIR, "data", "boneage-training-dataset")

VAL_CSV = os.path.join(ROOT_DIR, "data", "Bone Age Validation Set", "Validation Dataset.csv")
VAL_DIR = os.path.join(ROOT_DIR, "data", "Bone Age Validation Set","boneage-validation-dataset-1")
EXP_DIR = os.path.join(ROOT_DIR, "experiments", NAME, EXP_NAME)
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")



os.makedirs(EXP_DIR, exist_ok=True)


#Dataloader
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEM = True
TRANSFORM = torchvision.models.ResNet50_Weights.DEFAULT.transforms()

#Model

MODEL = resnet50.ResNet50_Regressor()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Train

CRITERION = torch.nn.MSELoss()

## Phase 1 : Train FC
EPOCHS_P1 = 2
OPTIMIZER_P1 = torch.optim.AdamW
LERNING_RATE_P1 = 1e-3
WEIGHT_DECAY_P1 = 1e-4

## Phase 2 : Fine Tune

EPOCHS_P2 = 2
OPTIMIZER_P2 = torch.optim.SGD
LERNING_RATE_P2 = 1e-4
REGULARIZER_P2 = 1e-5

#Log

PLOT_FREQ = 2




#%% Data

TRAIN_DATASET = dataloader.BoneAgeDataset(
        csv_path=TRAIN_CSV,
        img_dir = TRAIN_DIR,
        transform = TRANSFORM,
        image_size = IMG_SIZE,
        drop_missing = True)

train_loader = DataLoader(
    TRAIN_DATASET,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEM,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
)

val_dataset = dataloader.BoneAgeDataset(
        csv_path=VAL_CSV,
        img_dir = VAL_DIR,
        transform = TRANSFORM,
        image_size = IMG_SIZE,
        drop_missing = True)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEM,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
)

#%% Optimizers

opt1 = OPTIMIZER_P1(MODEL.parameters(), lr=LERNING_RATE_P1, weight_decay = WEIGHT_DECAY_P1)
opt2 = OPTIMIZER_P2(MODEL.parameters(), lr=LERNING_RATE_P2, weight_decay = REGULARIZER_P2)

#%% Train FC

print("Strarting FC training")

MODEL.to(DEVICE)

for param in MODEL.parameters():
    param.requires_grad = False

print("Freezed all params")

for param in MODEL.fc.parameters():
    param.requires_grad = True

print("Unfreezed fc params")

best = 1e10
train_loss_list = []
val_loss_list = []

for epoch in range(1, EPOCHS_P1 + 1):

    start_time = time.perf_counter()

    MODEL.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        y = labels["boneage"].to(DEVICE)
        # sex = labels["male"].to(DEVICE)

        pred = MODEL(imgs)
        loss = CRITERION(pred, y)
        opt1.zero_grad()
        loss.backward()
        opt1.step()
        train_loss += loss.item() * imgs.size(0)

    print(f"epoch {epoch}: Loss = {train_loss/len(train_loader.dataset):.3f}", end = "\t")

    MODEL.eval()
    val_loss = 0.0
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        y = labels["boneage"].to(DEVICE)
        # sex = labels["male"].to(DEVICE)

        with torch.no_grad():
            pred = MODEL(imgs)
        loss = CRITERION(pred, y)
        val_loss += loss.item() * imgs.size(0)

    print(f"Val Loss = {val_loss/len(val_loader.dataset):.3f}\tTime = {time.perf_counter() - start_time:.3f}s")

    train_loss_list.append(train_loss/len(train_loader.dataset))
    val_loss_list.append(val_loss/len(val_loader.dataset))

    if val_loss < best:
        best = val_loss
        os.makedirs(os.path.join(CKPT_DIR, EXP_NAME), exist_ok=True)
        torch.save(MODEL, os.path.join(CKPT_DIR, EXP_NAME, f"model_fc_epoch.pth"))
        print(f"Saved best model with loss {best} o_O")

    if (epoch % PLOT_FREQ == 0) and epoch != 0:
        analyze.plot_losses(train_loss_list, val_loss_list, save_path=os.path.join(EXP_DIR, f"loss_fc_epoch{epoch}.png"))
        print(f"Saved plot at epoch {epoch}")


df = pd.concat(
    {
        "train_loss": pd.Series(train_loss_list),
        "val_loss":   pd.Series(val_loss_list),
    },
    axis=1
)
df.index += 1
df.index.name = "epoch"

df.to_csv(os.path.join(EXP_DIR, "fc_logs.csv"), index=True)



#%% Fine Tune Whole

print("Starting fine tune")

MODEL.to(DEVICE)

for param in MODEL.parameters():
    param.requires_grad = True

print("Unfreezed all params")

train_loss_list = []
val_loss_list = []
best = 1e10

for epoch in range(1, EPOCHS_P2 + 1):

    start_time = time.perf_counter()

    MODEL.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        y = labels["boneage"].to(DEVICE)

        pred = MODEL(imgs)
        loss = CRITERION(pred, y)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
        train_loss += loss.item() * imgs.size(0)

    print(f"epoch {epoch}: Loss = {train_loss/len(train_loader.dataset):.3f}", end = "\t")

    MODEL.eval()
    val_loss = 0.0
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        y = labels["boneage"].to(DEVICE)

        with torch.no_grad():
            pred = MODEL(imgs)
        loss = CRITERION(pred, y)
        val_loss += loss.item() * imgs.size(0)

    print(f"Val Loss = {val_loss/len(val_loader.dataset):.3f}\tTime = {time.perf_counter() - start_time:.3f}s")

    train_loss_list.append(train_loss/len(train_loader.dataset))
    val_loss_list.append(val_loss/len(val_loader.dataset))

    if val_loss < best:
        best = val_loss
        os.makedirs(os.path.join(CKPT_DIR, EXP_NAME), exist_ok=True)
        torch.save(MODEL, os.path.join(CKPT_DIR, EXP_NAME, f"model_finetune_epoch.pth"))
        print(f"Saved best model with loss {best} o_O")

    if (epoch % PLOT_FREQ == 0) and epoch != 0:
        analyze.plot_losses(train_loss_list, val_loss_list, save_path=os.path.join(EXP_DIR, f"loss_finetune_epoch{epoch}.png"))
        print(f"Saved plot at epoch {epoch}")


df = pd.concat(
    {
        "train_loss": pd.Series(train_loss_list),
        "val_loss":   pd.Series(val_loss_list),
    },
    axis=1
)
df.index += 1
df.index.name = "epoch"

df.to_csv(os.path.join(EXP_DIR, "finetune_logs.csv"), index=True)




#%% Analysis


#%% Save Exp Params

cfg_path = os.path.join(EXP_DIR, "exp_params.txt")

with open(cfg_path, "w", encoding="utf-8") as f:

    f.write("ROOT_DIR = " + repr(str(ROOT_DIR)) + "\n")
    f.write("NAME = " + repr(NAME) + "\n")
    f.write("EXP_NAME = " + repr(EXP_NAME) + "\n\n")

    f.write("# Data\n")
    f.write("TRAIN_CSV = " + repr(str(TRAIN_CSV)) + "\n")
    f.write("TRAIN_DIR = " + repr(str(TRAIN_DIR)) + "\n")
    f.write("VAL_CSV = " + repr(str(VAL_CSV)) + "\n")
    f.write("VAL_DIR = " + repr(str(VAL_DIR)) + "\n")
    f.write("EXP_DIR = " + repr(str(EXP_DIR)) + "\n")
    f.write("CKPT_DIR = " + repr(str(CKPT_DIR)) + "\n\n")

    f.write("# Dataloader\n")
    f.write("IMG_SIZE = " + str(IMG_SIZE) + "\n")
    f.write("BATCH_SIZE = " + str(BATCH_SIZE) + "\n")
    f.write("NUM_WORKERS = " + str(NUM_WORKERS) + "\n")
    f.write("PIN_MEM = " + str(PIN_MEM) + "\n")
    # f.write("TRANSFORM = " + repr(TRANSFORM) + "\n\n")

    f.write("# Model\n")
    f.write("MODEL = " + f"{MODEL.__class__.__module__}.{MODEL.__class__.__name__}" + "\n")
    f.write("DEVICE = " + repr(DEVICE) + "\n\n")

    f.write("# Train\n")
    f.write("CRITERION = " + f"{CRITERION.__class__.__module__}.{CRITERION.__class__.__name__}" + "\n\n")

    f.write("## Phase 1 : Train FC\n")
    f.write("EPOCHS_P1 = " + str(EPOCHS_P1) + "\n")
    f.write("OPTIMIZER_P1 = " + f"{OPTIMIZER_P1.__module__}.{OPTIMIZER_P1.__name__}" + "\n")
    f.write("LERNING_RATE_P1 = " + str(LERNING_RATE_P1) + "\n")
    f.write("WEIGHT_DECAY_P1 = " + str(WEIGHT_DECAY_P1) + "\n\n")

    f.write("## Phase 2 : Fine Tune\n")
    f.write("EPOCHS_P2 = " + str(EPOCHS_P2) + "\n")
    f.write("OPTIMIZER_P2 = " + f"{OPTIMIZER_P2.__module__}.{OPTIMIZER_P2.__name__}" + "\n")
    f.write("LERNING_RATE_P2 = " + str(LERNING_RATE_P2) + "\n")
    f.write("REGULARIZER_P2 = " + str(REGULARIZER_P2) + "\n\n")

    f.write("# Log\n")
    f.write("PLOT_FREQ = " + str(PLOT_FREQ) + "\n")

print(f"Saved config to: {cfg_path}")


model_summary_path = os.path.join(EXP_DIR, "model_summary.txt")
with open(model_summary_path, "w", encoding="utf-8") as f:
    f.write(str(MODEL))


print("Script finished Executing ;) ")

