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

import matplotlib.pyplot as plt

mean_age = df['boneage'].mean()
median_age = df['boneage'].median()
std_age = df['boneage'].std()

plt.hist(df['boneage'], bins=30)
plt.axvline(mean_age, color="red", linestyle="--", label=f"Mean = {mean_age:.1f}")
plt.axvline(mean_age-std_age, color="green", linestyle="--", label=f"std")
plt.axvline(mean_age+std_age, color="green", linestyle="--", label=f"std")
plt.axvline(median_age, color="blue", linestyle="--", label=f"Median= {median_age:.1f}")
plt.show()
# %%

print(mean_age, median_age, std_age)
# %%
