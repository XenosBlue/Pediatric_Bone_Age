#%% Imports

import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2

#%% PARAMS

train_folder = "/mnt/c/Users/cnikh/Projects/dl_proj/Pediatric_Bone_Age/data/boneage-training-dataset/"
files = os.listdir(train_folder)

#%% EX


img = cv2.imread(os.path.join(train_folder, files[0]), cv2.IMREAD_GRAYSCALE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
img = clahe.apply(img)
plt.imshow(img, cmap='gray')

#%%

# %%
