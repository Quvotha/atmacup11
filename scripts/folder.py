import os


class Folder:
    # Where csv files are stored
    CSV: str = os.path.join(os.path.dirname(__file__), '..', 'dataset_atmaCup11')
    # Where image files are stored
    PHOTO: str = os.path.join(CSV, 'images')
    # Where experiments result are stored
    EXPERIMENTS: str = os.path.join(os.path.dirname(__file__), '..', 'experiments')
    # Where fold indice are stored
    FOLD: str = os.path.join(os.path.dirname(__file__), '..', 'fold')


def experiment_dir_of(exp_no: int):
    if not isinstance(exp_no, int) or exp_no < 0:
        raise ValueError(f'`exp` should be positive integer but {exp_no} given')
    directory = os.path.join(Folder.EXPERIMENTS, f'exp{str(exp_no).zfill(3)}')
    if os.path.isdir(directory):
        raise ValueError(f'{directory} already exists')
    os.makedirs(directory)
    return directory
