#%% Imports

import os
import csv
import glob
from typing import List, Dict, Optional, Callable
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

#%% Definitions

def _index_images(img_dir: str):
    paths = []
    paths.extend(glob.glob(os.path.join(img_dir, "*.png")))
    id_to_path = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem.isdigit():
            id_to_path[int(stem)] = os.path.abspath(p)
    return id_to_path

def _default_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

class BoneAgeDataset(Dataset):
    def __init__(
        self,
        csv_path,
        img_dir,
        transform= None,
        image_size= 224,
        drop_missing = True,
    ):
        self.id_to_path = _index_images(img_dir)

        rows = []
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                rid  = int((r.get("id") or r.get("Image ID")))
                bone = float((r.get("boneage") or r.get("Bone Age (months)")))
                bone = (bone - 132) / 41.182                           # EDIT: normalize with median and std dev
                # rid = int(r["id"])
                # bone = int(r["boneage"])/228                            #EDIT: max age is set to 228
                # male = r["male"].strip().lower() is True
                male_str = (r.get("male") or r.get("Male") or "").strip().lower()
                male = male_str in ("1", "true", "TRUE", "True")
                # print(r["id"])
                # exit()
                path = self.id_to_path.get(rid)
                if path is None:
                    if drop_missing:
                        # print(f"image for id {rid} not found ;_;")
                        continue
                    else:
                        raise FileNotFoundError(f"No image for id {rid} ;_; Big Sad")
                rows.append({"id": rid, "boneage": bone, "male": male, "path": path})
        if len(rows) == 0:
            raise RuntimeError("Bruh no images found ;_;")
        self.samples = rows

        print(f"Found {len(rows)} samples")

        self.transform = transform or _default_transform(image_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        x = self.transform(img) if self.transform else img

        target = {
            "boneage": torch.tensor(float(s["boneage"]), dtype=torch.float32),
            "male": torch.tensor(1.0 if s["male"] is True else 0.0, dtype=torch.float32),
        }
        return x, target

#%% Local Test


# idx_dict = _index_images("/mnt/c/Users/cnikh/Projects/dl_proj/Pediatric_Bone_Age/data/boneage-training-dataset")
# csv_path = "/mnt/c/Users/cnikh/Projects/dl_proj/Pediatric_Bone_Age/data/train.csv"
# print(idx_dict)

# rows = []
# with open(csv_path, newline="") as f:
#     for r in csv.DictReader(f):
#         rid = int(r["id"])
#         bone = int(r["boneage"])/228                            #EDIT: max age is set to 228
#         male = r["male"].strip().lower() is True
#         # print(r["id"])
#         # exit()
#         path = idx_dict.get(rid)
#         if path is None:
#             if True:
#                 print(f"image for id {rid} not found ;_;")
#                 continue
#             else:
#                 raise FileNotFoundError(f"No image for id {rid} ;_; Big Sad")
#         rows.append({"id": rid, "boneage": bone, "male": male, "path": path})
# %%
