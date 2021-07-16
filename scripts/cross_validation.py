import os.path
import pickle
from typing import Tuple, List

from folder import Folder


def load_cv_object_ids(
    filepath: str = os.path.join(Folder.FOLD, 'train_validation_object_ids.pkl')
) -> Tuple[List[str], List[str]]:
    """[summary]

    Parameters
    ----------
    filepath : str, optional
        [description], by default os.path.join(Folder.FOLD, 'train_validation_object_ids.pickle')

    Returns
    -------
    list of `object_ids`s of training/validation fold : Tuple[List[str], List[str]]
        1st list is list of `object_id` s for training set. 2nd one is that for validation set. 
    """
    with open(filepath, 'rb') as f:
        fold_object_ids = pickle.load(f)
    return (fold_object_ids['training'], fold_object_ids['validation'])
