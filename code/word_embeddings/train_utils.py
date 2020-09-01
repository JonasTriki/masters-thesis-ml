from os.path import join as join_path


def get_model_checkpoint_filepath(
    checkpoints_dir: str, model_name: str, dataset_name: str, epoch_nr: int, loss: float,
) -> str:
    """
    TODO: Docs
    """
    filename = f"{model_name}_{dataset_name}_{epoch_nr:02d}_{loss}.h5"
    filepath = join_path(checkpoints_dir, filename)
    return filepath
