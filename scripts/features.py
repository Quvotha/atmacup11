from typing import Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from dataset import load_photofile


def extract_representative_colors(
        image: np.ndarray, *, num_colors: int, n_jobs: Union[None, int],
        seed: int) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError('`image` should be np.ndarray but {} given'.format(type(image)))
    if not isinstance(num_colors, int) or num_colors < 1:
        raise ValueError('`num_colors` should be integer >= 1 but {} given'.format(num_colors))
    if image.ndim != 3:
        raise ValueError('`image` should be RGB image')
    # Calculate R, G, B point
    clusterer = KMeans(
        n_clusters=num_colors, random_state=seed, n_jobs=n_jobs).fit(
        image.reshape(-1, 3))
    # Consistent order (R > G > B order)
    center_df = pd.DataFrame(data=clusterer.cluster_centers_)
    representative_colors = center_df \
        .sort_values(center_df.columns.tolist()) \
        .to_numpy() \
        .flatten()
    return representative_colors


def extract_representative_color_features(
        object_id: str, offset: int = 56, image_size: int = 224, num_colors: int = 5,
        n_jobs: Union[None, int] = -1, seed: int = 1) -> np.ndarray:
    assert(image_size % offset == 0)
    image = load_photofile(object_id, resize_to=(image_size, image_size))
    # representative colors of image
    representative_colors = extract_representative_colors(
        image, num_colors=num_colors, n_jobs=n_jobs, seed=seed)
    # representative colors of image segment
    segments = []
    x_from, x_to = 0, offset
    while x_to <= image_size:
        y_from, y_to = 0, offset
        while y_to <= image_size:
            segments.append([x_from, x_to, y_from, y_to])
            y_from += offset
            y_to = y_from + offset
        x_from += offset
        x_to = x_from + offset
    for (x_from, x_to, y_from, y_to) in segments:
        image_segment = image[y_from:y_to, x_from:x_to, :]
        representative_colors_ = extract_representative_colors(
            image_segment, num_colors=num_colors, n_jobs=n_jobs, seed=seed)
        representative_colors = np.hstack([representative_colors, representative_colors_])
    return representative_colors
