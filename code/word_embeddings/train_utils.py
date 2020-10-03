import re
from os import listdir
from os.path import join
from typing import Dict, List, Union

import numpy as np


def create_model_checkpoint_filepath(
    checkpoints_dir: str,
    model_name: str,
    dataset_name: str,
    epoch_nr: int,
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

    Returns
    -------
    filepath : str
        Filepath of a model checkpoint.
    """
    filename = f"{model_name}_{dataset_name}_{epoch_nr:02d}.model"
    filepath = join(checkpoints_dir, filename)
    return filepath


def get_model_checkpoint_filepaths(
    checkpoints_dir: str, model_name: str, dataset_name: str
) -> Dict[str, Union[str, List[str]]]:
    """
    Gets model checkpoint filepaths of a specific model (trained on a specific dataset)
    from a checkpoints directory.

    Parameters
    ----------
    checkpoints_dir : str
        Model checkpoints directory containing the trained models.
    model_name : str
        Name of the model.
    dataset_name : str
        Name of the dataset which the model has been trained on.

    Returns
    -------
    filepaths_dict : dict
        Dictionary containing filepaths to trained models, intermediate weight embeddings,
        words used during training and training log.
    """
    # List files in checkpoints directory
    checkpoints_filenames = listdir(checkpoints_dir)

    # Filter by model_name and dataset_name entries only
    model_id = f"{model_name}_{dataset_name}"
    checkpoints_filenames = [
        fn for fn in checkpoints_filenames if fn.startswith(model_id)
    ]

    # Get model filenames and sort them by epoch numbers (from first to last).
    model_filenames = np.array(
        [fn for fn in checkpoints_filenames if fn.endswith(".model")]
    )
    model_epoch_nrs = np.array(
        [int(re.findall(r"_(\d{2}).model", fn)[0]) for fn in model_filenames]
    )
    model_filenames = model_filenames[np.argsort(model_epoch_nrs)]

    # Append checkpoint directory to filenames
    model_filepaths = [join(checkpoints_dir, fn) for fn in model_filenames]

    # Get intermediate embedding weights sorted by first to last
    intermediate_embedding_weight_filenames = np.array(
        [fn for fn in checkpoints_filenames if fn.endswith("weights.npy")]
    )
    intermediate_embedding_weight_filepaths = None
    train_words_filepath = None
    if len(intermediate_embedding_weight_filenames) > 0:

        # Extract combined epoch/embedding nrs and sort by them.
        epoch_embedding_nrs = []
        for fn in intermediate_embedding_weight_filenames:
            epoch_nr, embedding_nr = re.findall(r"_(\d{2})_(\d{2})_weights.npy", fn)[0]
            epoch_embedding_nr = int(f"{epoch_nr}{embedding_nr}")
            epoch_embedding_nrs.append(epoch_embedding_nr)
        epoch_embedding_nrs = np.array(epoch_embedding_nrs)
        intermediate_embedding_weight_filenames = intermediate_embedding_weight_filenames[
            np.argsort(epoch_embedding_nrs)
        ]

        # Append checkpoint directory to filenames
        intermediate_embedding_weight_filepaths = [
            join(checkpoints_dir, fn) for fn in intermediate_embedding_weight_filenames
        ]

        train_words_filename = f"{model_id}_words.txt"
        if train_words_filename in checkpoints_filenames:
            train_words_filepath = join(checkpoints_dir, train_words_filename)

    # Add path to train logs
    train_logs_filename = f"{model_id}_logs.csv"
    train_logs_filepath = None
    if train_logs_filename in checkpoints_filenames:
        train_logs_filepath = join(checkpoints_dir, train_logs_filename)

    return {
        "model_filepaths": model_filepaths,
        "intermediate_embedding_weight_filepaths": intermediate_embedding_weight_filepaths,
        "train_words_filepath": train_words_filepath,
        "train_logs_filepath": train_logs_filepath,
    }


def create_model_intermediate_embedding_weights_filepath(
    checkpoints_dir: str,
    model_name: str,
    dataset_name: str,
    epoch_nr: int,
    intermediate_embedding_weight_nr: int,
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
    intermediate_embedding_weight_nr : int
        Intermediate embedding weight number

    Returns
    -------
    filepath : str
        Filepath of a model checkpoint.
    """
    filename = f"{model_name}_{dataset_name}_{epoch_nr:02d}_{intermediate_embedding_weight_nr:02d}_weights.npy"
    filepath = join(checkpoints_dir, filename)
    return filepath


def create_model_train_logs_filepath(
    checkpoints_dir: str,
    model_name: str,
    dataset_name: str,
) -> str:
    """
    Gets the filepath for the train logs of a model.

    Parameters
    ----------
    checkpoints_dir : str
        Checkpoints directory.
    model_name : str
        Name of the model.
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    filepath : str
        Filepath for the train logs of a model.
    """
    filename = f"{model_name}_{dataset_name}_logs.csv"
    filepath = join(checkpoints_dir, filename)
    return filepath
