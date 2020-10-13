from os.path import join

import tensorflow as tf


def enable_dynamic_gpu_memory() -> bool:
    """
    Enables dynamic GPU memory growth on all GPUs.
    https://www.tensorflow.org/guide/gpu

    Returns
    -------
    dynamic_memory_enabled : bool
        Whether or not we successfully enabled dynamic memory.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            return True
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    return False


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
