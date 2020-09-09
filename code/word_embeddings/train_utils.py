import os
import re
from typing import List

import numpy as np


def create_model_checkpoint_filepath(
    checkpoints_dir: str,
    model_name: str,
    dataset_name: str,
    epoch_nr: int,
    train_loss: float,
) -> str:
    """
    Gets the filepath of a model checkpoint.

    Parameters
    ----------
    checkpoints_dir : str
        Checkpoints directory.
    model_name : str
        Name of the model.
    dataset_name : str
        Name of the dataset.
    epoch_nr : int
        Epoch number.
    train_loss : float
        Train loss.

    Returns
    -------
    filepath : str
        Filepath of a model checkpoint.
    """
    filename = f"{model_name}_{dataset_name}_{epoch_nr:02d}_{train_loss}.model"
    filepath = os.path.join(checkpoints_dir, filename)
    return filepath


def get_model_checkpoint_filepaths(
    checkpoints_dir: str, file_ext: str = ".model"
) -> List[str]:
    """
    Gets model checkpoint filepaths from a checkpoints directory sorted by epochs
    (from first to last epoch).

    Parameters
    ----------
    checkpoints_dir : str
        Model checkpoints directory containing the trained models.
    file_ext : str, optional
        File extension of the models.

    Returns
    -------
    filepaths : list of str
        List of filepaths (sorted by first to last epoch) to model checkpoints.
    """
    # Get filenames
    filenames = np.array(
        [fn for fn in os.listdir(checkpoints_dir) if fn.endswith(file_ext)]
    )

    # Extract epoch numbers from filenames
    epoch_nrs = np.array([int(re.findall(r"_(\d{2})_", fn)[0]) for fn in filenames])

    # Sort filenames by epoch numbers
    epoch_nrs_sorted_indices = np.argsort(epoch_nrs)
    filenames = filenames[epoch_nrs_sorted_indices]

    # Append checkpoint directory to filenames
    filepaths = [os.path.join(checkpoints_dir, fn) for fn in filenames]

    return filepaths
