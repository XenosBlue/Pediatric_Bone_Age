#%% imports
import os, glob
import cv2
from typing import List
from tqdm import tqdm

#%% functions
def clahe_dir_list(dir_list: List[str],
                   exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff")):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for d in dir_list:
        files = [p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                 if p.lower().endswith(exts)]
        for p in tqdm(files, desc=d):
            img = cv2.imread(p)
            if img is None:
                continue
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = clahe.apply(l)
            lab2 = cv2.merge((l2, a, b))
            out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
            cv2.imwrite(p, out)

#%% example

dirs = [
    "../data/boneage-training-dataset",
    "../data/Bone Age Validation Set/boneage-validation-dataset-1",
    "../data/Bone Age Validation Set/boneage-validation-dataset-2",
    
]
clahe_dir_list(dirs)

#%%