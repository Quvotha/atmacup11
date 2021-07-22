# atmacup11
Solution of [atmaCup11](https://www.guruguru.science/competitions/17).

# Solution

My solution is ensembling following 

1. Resnet18
  - Number of epochs: 50
  - Loss function: `torch.nn.MSELoss()`
  - Optimaizer: `torch.nn.Adam`
  - Notebook: [exp012.ipynb](experiments/exp012.ipynb)
  - CV = 0.841569, PulicLB = 0.8149, PrivateLB = 0.8155

2. Inception V3
- Number of epochs: 100
- Loss function: `torch.nn.MSELoss()`
- Optimaizer: `torch.nn.Adam`
- Notebook: [exp017.ipynb](experiments/exp017.ipynb)
- CV = 0.815681, PulicLB = 0.7677, PrivateLB = 0.7596

## CV
5-Fold StratifiedGroupKFold. 
The fold is stored in [fold/train_validation_object_ids.pkl](fold/train_validation_object_ids.pkl)

[Data Description](#Data-description)
____
## Data description

### fold/train_validation_object_ids.pkl
- Description  
Use for StratifiedGroupKFold cross validation.  
Dictionary object is pickled and stored in this file.
The dictionary has 2 keys; "training" and "validation". Both values are list of which length is 5, and each value of that list is list of `object_id`s.
- Output from  
Save fold.ipynb.
