from functools import lru_cache
import os
from typing import Final, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from folder import Folder


def load_csvfiles(csv_directory: str = Folder.CSV) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                            pd.DataFrame, pd.DataFrame]:
    """Load train.csv, test.csv, materials.csv, techniques.csv, and atmaCup#11_sample_submission.csv.

    Parameters
    ----------
    csv_directory : str
        The directory where csv files are stored.

    Returns
    -------
    Tuple[pd.DataFrame]
        train.csv, test.csv, materials.csv, techniques.csv, and atmaCup#11_sample_submission.csv loaded by `pandas.read_csv()`.
    """
    return (
        pd.read_csv(os.path.join(csv_directory, 'train.csv')),
        pd.read_csv(os.path.join(csv_directory, 'test.csv')),
        pd.read_csv(os.path.join(csv_directory, 'materials.csv')),
        pd.read_csv(os.path.join(csv_directory, 'techniques.csv')),
        pd.read_csv(os.path.join(csv_directory, 'atmaCup#11_sample_submission.csv')),
    )


def _id2filename(object_id: str) -> str:
    return f'{object_id}.jpg'


@lru_cache(maxsize=1024)
def load_photofile(object_id: str, photo_directory: str = Folder.PHOTO,
                   resize_to: Optional[Tuple[int, int]] = (224, 224)) -> np.ndarray:
    """Load 1 jpg image of given object_id.

    Parameters
    ----------
    object_id : strpy
        Photo's object id.
    photo_directory : str, optional
        The directory where photo files are stored.
    resize_to : Optional[Tuple[int, int]], optional
        Image data is to be resized after loaded if `resize_to` is given.
        Should be given in (width, height) style. If None is given, resize is not to be applied.

    Returns
    -------
    image: np.ndarray
        Image data represented as numpy's array.
    """
    filepath = os.path.join(photo_directory, _id2filename(object_id))
    image = Image.open(filepath)
    if resize_to is not None:
        image = image.resize(resize_to)
    return np.array(image)


def load_photofiles(object_ids, photo_directory: str = Folder.PHOTO,
                    resize_to: Optional[Tuple[int, int]] = (224, 224)) -> np.ndarray:
    """Iteratively load jpg images of given object ids.

    Parameters
    ----------
    object_ids : Sequence of str
        Photo's object ids.
    photo_directory : str, optional
        The directory where photo files are stored.
    resize_to : Optional[Tuple[int, int]], optional
        Image data is to be resized after loaded if `resize_to` is given.
        Should be given in (width, height) style. If None is given, resize is not to be applied.

    Returns
    -------
    images: np.adarray
        Image data represented as numpy's array. Length is equal to that of object_ids.
    """
    image_arrays = [load_photofile(object_id, photo_directory, resize_to)
                    for object_id in object_ids]
    return np.array(image_arrays)
