# atmacup11
Solution of [atmaCup11](https://www.guruguru.science/competitions/17).

# Solution

## Rank

|LB     |Rank|RMSE  |
|:---:  |---:|---:  |
|Private|32  |0.6997|
|Public |29  |0.7113|

## Overview

The code is [experiments/exp035.ipynb](experiments/exp035.ipynb).
This code makes ensemble model of following 8 weak learners (7 DNN models and 1 SVM model) using `RidgeCV` of scikit-learn.

- DNN, architecture = ResNet18, number of epochs = 50
  - Notebook: [experiments/exp012.ipynb](experiments/exp012.ipynb)
  - CV = 0.841569, PublicLB = 0.8149, PrivateLB = 0.8155

- DNN, architecture = ResNet18, number of epochs = 200
  - Notebook: [experiments/exp021.ipynb](experiments/exp021.ipynb)
  - CV = 0.790173, PublicLB = 0.7541, PrivateLB = 0.7512

- DNN, architecture = ResNet18, number of epochs = 500
  - Notebook: [experiments/exp024.ipynb](experiments/exp024.ipynb)
  - CV = 0.789750, PublicLB = 0.7416, PrivateLB = 0.7311

- DNN, architecture = Inception V3, number of epochs = 100
  - Notebook: [experiments/exp017.ipynb](experiments/exp017.ipynb)
  - CV = 0.815681, PublicLB = 0.7677, PrivateLB = 0.7596

- DNN, architecture = Inception V3, number of epochs = 200
  - Notebook: [experiments/exp023.ipynb](experiments/exp023.ipynb)
  - CV = 0.803318, PublicLB = 0.7580, PrivateLB = 0.7483

- DNN, architecture = SqueezeNet, number of epochs = 200
  - Notebook: [experiments/exp022.ipynb](experiments/exp022.ipynb)
  - CV = 0.831849, PublicLB = 0.8180, PrivateLB = 0.8135

- DNN, architecture = EfficientNet-b0, number of epochs = 200
  - Notebook: [experiments/exp026.ipynb](experiments/exp026.ipynb)
  - CV = 0.776911, PublicLB = 0.7266, PrivateLB = 0.7186

- SVM regression (`SVR` of scikit-learn)
  - Notebook: [experiments/exp020.ipynb](experiments/exp020.ipynb)
  - CV = 0.843052, PublicLB = 0.8557, PrivateLB = 0.8314

These models predict `target` as continuous value, not discrete value.

DNN models use MSE (`torch.nn.MSELoss()`) as loss function and Adam (`torch.nn.Adam()`) as optimizer. 

SVM model predict `target` from __representative colors vector__ of each image. Following is how to extract representative colors vector from each image.
1. Load image.
1. Resize; height = width = 224.
1. Extract 8 representative colors from image. It is done by applying `KMeans` of scikit-learn to image after reshaping it into (224 * 224, 3) array (224 is image size and 3 is number of channel). Coordinates of cluster centers are representative RGB colors of that image. [This Qiita article](https://qiita.com/simonritchie/items/396112fb8a10702a3644) helps me a lot.
1. Separate image into 56 * 56 sub-image, and extract 8 representative colors from each sub-image. Image size is 224 * 224 thus there are 4 * 4 sub-images in 1 image (224 / 56 = 4).
1. Flatten representative RGB colors, concatenate them, then representative colors vector is obtained.

## CV
5-Fold StratifiedGroupKFold. 

The fold is stored in [fold/train_validation_object_ids.pkl](fold/train_validation_object_ids.pkl). Dictionary object is pickled and stored in this file. The dictionary has 2 keys; "training" and "validation". Both values are list of which length is 5, and each value of that list is list of `object_id`s. Following snipet shows hot to use this file for cross validation.

```
filepath = PATH_TO_train_validation_object_ids.pkl_FROM_CURRENT_DIR

# The file is pickled dictionary
import pickle
with open(filepath, 'rb') as f:
    fold_object_ids = pickle.load(f)

# There are 2 keys; "training" and "validation"
# Each value is List[str], length = 5
len(fold_object_ids['training'])  # 5
len(fold_object_ids['validation'])  # 5

# `train_object_ids` is list of `object_id` of training set.
# `valid_object_ids` is that of validation set.
for train_object_ids, valid_object_ids in zip(fold_object_ids['training'], fold_object_ids['validation']):
    # your cross validation code here
```

This file is output from [Save fold.ipynb](Save%20fold.ipynb).

# Environment

## Hardware

I used [Google Colab pro](https://colab.research.google.com/signup) and my laptop (OS: Microsoft Windows 10 Home, RAM: 16GB, CPU: Intel(R) Core(TM) i5-10300HCPU @ 2.5GHz (4-cores), No GPU).

## Python packages

Virtual environment manager is Miniconda 4.9.2. Here are 2 `conda list > ...` outputs.

- [conda_packages.txt](conda_packages.txt)  
  All of packages required for local work is listed in this file.

- [requerements_sklearn_dev.txt](requerements_sklearn_dev.txt)  
  Just use for `StratifiedGroupKFold` of scikit-learn. This class was implemented only in unstable development version, and that version conflicted with other packages such as lightgbm, thus I separated virtual environment only when I use `StratifiedGroupKFold`.
