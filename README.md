# Pediatric_Bone_Age
Probabilistic prediction of bone age

# Setup

- Install the Repo
```
https://github.com/XenosBlue/Pediatric_Bone_Age.git
```

- Install the environment
```
conda env create -f environment.yml
```

- Dowload dataset
```
cd scripts

./download_dataset.bash

```

# Run Training

```
cd scripts
```

Open the train.py file and Edit parameters in under the #%% Parameters section

```
python train.py
```




# Explaination Folder Structure

- src contains classes and definitions 
- archs contain different architectures
- checkpoints contains trained checkpoints
- scripts contains train and test scripts 

# Results

| Arch         | Acc     |
|--------------|---------|
| ResNet50     | 0.0     |
| VIT(b)       | 0.0     |
| EfficientNet | 0.0     |


