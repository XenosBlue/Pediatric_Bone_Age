#%% Imports

import pandas as pd


#%% PARAMS

csv_file = "/mnt/c/Users/cnikh/Projects/dl_proj/Pediatric_Bone_Age/data/train.csv"


#%% EX



df = pd.read_csv(csv_file)
print(df.head())

#%%

print(df['boneage'])
print(df['boneage'].min())
print(df['boneage'].max())
print(df['boneage'].mean())
print(df['boneage'].std())

# %%
