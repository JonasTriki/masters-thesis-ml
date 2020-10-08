from os.path import join


def create_model_checkpoint_filepath(
    output_dir: str,
    model_name: str,
    dataset_name: str,
    epoch_nr: int,
) -> str:
    """
    Gets the filepath of a model checkpoint.

    Parameters
    ----------
    output_dir : str
        Output directory.
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
    filepath = join(output_dir, filename)
    return filepath


def create_model_intermediate_embedding_weights_filepath(
    output_dir: str,
    model_name: str,
    dataset_name: str,
    epoch_nr: int,
    intermediate_embedding_weight_nr: int,
) -> str:
    """
    Gets the filepath of a model checkpoint.

    Parameters
    ----------
    output_dir : str
        Output directory.
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
    filepath = join(output_dir, filename)
    return filepath


def create_model_train_logs_filepath(
    output_dir: str,
    model_name: str,
    dataset_name: str,
) -> str:
    """
    Gets the filepath for the train logs of a model.

    Parameters
    ----------
    output_dir : str
        Output directory.
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
    filepath = join(output_dir, filename)
    return filepath
