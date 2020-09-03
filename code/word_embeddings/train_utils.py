import os


def get_model_checkpoint_filepath(
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
