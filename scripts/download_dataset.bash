#!/usr/bin/env bash

# https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Training+Set.zip
# https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Training+Set+Annotations.zip
# https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Validation+Set.zip


mkdir ../data || echo "folder data already exists -_-"

cd ../data

wget --show-progress -O training_annotations.zip https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Training+Set+Annotations.zip

unzip training_annotations.zip

wget --show-progress -O training_data.zip https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Training+Set.zip

unzip training_data.zip

wget --show-progress -O validation_data.zip https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Validation+Set.zip

unzip validation_data.zip

cd "Bone Age Validation Set"

unzip boneage-validation-dataset-1.zip

unzip boneage-validation-dataset-2.zip

cd ../..

echo "Downloaded Data o_o"



