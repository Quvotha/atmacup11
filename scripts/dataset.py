from functools import lru_cache
import os
from typing import Optional, Tuple, Iterable, Union

import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from folder import Folder

IMAGE_WIDTH: int = 224
IMAGE_HEIGHT: int = 224
IMAGE_NUM_CHANNELS: int = 3


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
                   resize_to: Optional[Tuple[int, int]] = (IMAGE_WIDTH, IMAGE_HEIGHT),
                   return_ndarray: bool = True) -> Union[np.ndarray, JpegImagePlugin.JpegImageFile]:
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
    return_ndarray : bool
        Return np.ndarry if True. Otherwise return JpegImagePlugin.JpegImageFile.

    Returns
    -------
    image: np.ndarray
        Image data represented as numpy's array.
    """
    filepath = os.path.join(photo_directory, _id2filename(object_id))
    image = Image.open(filepath)
    if resize_to is not None:
        image = image.resize(resize_to)
    return np.array(image) if return_ndarray else image


def load_photofiles(object_ids, photo_directory: str = Folder.PHOTO,
                    resize_to: Optional[Tuple[int, int]] = (IMAGE_WIDTH, IMAGE_HEIGHT)) -> np.ndarray:
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


# Note: AutoEncoder に食わせる画像をポコポコ生み出す
class AtmaImageDatasetV01(Dataset):

    def __init__(self, object_ids: Iterable[str],
                 width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT,
                 num_channels: int = IMAGE_NUM_CHANNELS):
        """Supply competition's image data.

        Parameters
        ----------
        object_ids : Iterable[str]
            `object_id`
        width : int, optional
            Image width. By default IMAGE_WIDTH
        height : int, optional
            Image height. By default IMAGE_HEIGHT
        num_channels : int, optional
            Number of channels. By default IMAGE_NUM_CHANNELS
        """
        self.object_ids = np.array(object_ids)
        self.width = width
        self.height = height
        self.num_channels = num_channels
        self.transformers = T.Compose([T.ToTensor()])

    def __getitem__(self, index):
        object_ids = self.object_ids[index]
        if not isinstance(object_ids, np.ndarray):
            object_ids = np.array([object_ids, ])
        images = load_photofiles(object_ids)
        return images

    def __len__(self):
        return len(self.object_ids)


class AtmaImageDatasetV02(Dataset):

    def __init__(self, object_ids: Iterable[str], transformers,
                 labels: Optional[Iterable[int]] = None):
        """Competition's image data and label.

        Parameters
        ----------
        object_ids : Iterable[str]
            `object_id`
        transformers : [type]
            Transformer object which will be applied to image data.
        labels : Optional[Iterable[int]]
            `target` (training and validation set) or `None` (test set).
        """
        self.object_ids = np.array(object_ids)
        self.labels = None if labels is None else np.array(labels)
        if self.labels is not None and len(self.object_ids) != len(self.labels):
            raise ValueError('`object_ids` and `images` should be same length ({}, {})'
                             .format(len(self.object_ids), len(self.labels)))
        self.transformers = transformers

    def __getitem__(self, index) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        object_id = self.object_ids[index]
        # if not isinstance(object_ids, np.ndarray):
        #     object_ids = np.array([object_ids, ])
        image = load_photofile(object_id, return_ndarray=False, resize_to=None)
        image = self.transform_image(image)
        if self.labels is None:
            return image
        else:
            label = self.labels[index]
            label = torch.tensor(label).float()
            return image, label

    def __len__(self):
        return len(self.object_ids)

    def transform_image(self, image: JpegImagePlugin) -> torch.Tensor:
        return image if self.transformers is None else self.transformers(image)
