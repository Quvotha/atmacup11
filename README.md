# atmacup11
Solution of [atmaCup11](https://www.guruguru.science/competitions/17).

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
