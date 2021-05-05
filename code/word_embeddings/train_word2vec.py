import argparse
import os
import sys
from datetime import datetime
from os.path import join

from tensorflow.keras.mixed_precision import experimental as tf_mixed_precision

sys.path.append("..")

from utils import get_all_filepaths, text_files_total_line_count  # noqa: E402
from word_embeddings.tokenizer import Tokenizer, load_tokenizer  # noqa: E402
from word_embeddings.train_utils import enable_dynamic_gpu_memory  # noqa: E402
from word_embeddings.word2vec import Word2vec, load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parses arguments sent to the python script.

    Returns
    -------
    parsed_args : argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_data_filepath",
        type=str,
        default="",
        help="Text filepath containing the text we wish to train on",
    )
    parser.add_argument(
        "--text_data_dir",
        type=str,
        default="",
        help="Directory containing text files we wish to train on",
    )
    parser.add_argument(
        "--tokenizer_filepath",
        type=str,
        default="",
        help="Filepath of a built tokenizer",
    )
    parser.add_argument(
        "--save_to_tokenizer_filepath",
        type=str,
        default="",
        help="Filepath to use for saving the tokenizer",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help="Name of the dataset we are training on. "
        "Used to denote saved checkpoints during training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size used for training",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=5,
        help="Number of epochs to train our model on",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.025,
        help="Learning rate to use when training",
    )
    parser.add_argument(
        "--min_learning_rate",
        type=float,
        default=0.0000025,
        help="Minimum learning rate to use when training",
    )
    parser.add_argument(
        "--max_vocab_size",
        type=int,
        default=-1,
        help="Maximum vocabulary size to use when training. Defaults to use all words",
    )
    parser.add_argument(
        "--min_word_count",
        type=int,
        default=5,
        help="Minimum number of times a word might occur for it to be in the vocabulary",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=300,
        help="Number of latent dimensions to use in the embedding layers",
    )
    parser.add_argument(
        "--max_window_size",
        type=int,
        default=5,
        help="Maximum window size to use when generating skip-gram couples",
    )
    parser.add_argument(
        "--num_negative_samples",
        type=int,
        default=10,
        help="Number of negative samples to use when generating skip-gram couples",
    )
    parser.add_argument(
        "--sampling_factor",
        type=float,
        default=1e-5,
        help="Sampling factor to use when computing the probability of "
        "keeping a word during random subsampling of words",
    )
    parser.add_argument(
        "--unigram_exponent_negative_sampling",
        type=float,
        default=3 / 4,
        help="Which exponent to raise the unigram distribution to when performing negative sampling",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory to save metadata files, checkpoints and intermediate model weights",
    )
    parser.add_argument(
        "--pretrained_model_filepath",
        type=str,
        default="",
        help="Load an already trained word2vec model from file",
    )
    parser.add_argument(
        "--starting_epoch_nr",
        type=int,
        default=1,
        help="Epoch number to start the training from",
    )
    parser.add_argument(
        "--train_logs_to_file",
        default=False,
        action="store_true",
        help="Whether or not to save logs from training to file",
    )
    parser.add_argument(
        "--intermediate_embedding_weights_saves",
        type=int,
        default=0,
        help="Number of intermediate saves of embedding weights per epoch during training",
    )
    parser.add_argument(
        "--dynamic_gpu_memory",
        default=False,
        action="store_true",
        help="Whether or not to enable dynamic GPU memory",
    )
    parser.add_argument(
        "--mixed_precision",
        default=False,
        action="store_true",
        help="Whether or not to use mixed float16 precision while training (requires NVIDIA GPU, e.g., RTX, Titan V, V100)",
    )
    parser.add_argument(
        "--tensorboard_logs_dir",
        type=str,
        default="tensorboard_logs",
        help="TensorBoard logs directory",
    )
    parser.add_argument(
        "--cpu_only",
        default=False,
        action="store_true",
        help="Whether or not to train on the CPU only",
    )
    return parser.parse_args()


def train_word2vec(
    text_data_filepath: str,
    text_data_dir: str,
    tokenizer_filepath: str,
    save_to_tokenizer_filepath: str,
    dataset_name: str,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    min_learning_rate: float,
    max_vocab_size: int,
    min_word_count: int,
    embedding_dim: int,
    max_window_size: int,
    num_negative_samples: int,
    sampling_factor: float,
    unigram_exponent_negative_sampling: float,
    output_dir: str,
    pretrained_model_filepath: str,
    starting_epoch_nr: int,
    train_logs_to_file: bool,
    intermediate_embedding_weights_saves: int,
    dynamic_gpu_memory: bool,
    mixed_precision: bool,
    tensorboard_logs_dir: str,
    cpu_only: bool,
) -> None:
    """
    Trains a word2vec model using skip-gram negative sampling.

    Parameters
    ----------
    text_data_filepath : str
        Text filepath containing the text we wish to train on.
    text_data_dir : str
        Directory containing text files we wish to train on.
    tokenizer_filepath : str
        Filepath of the built Tokenizer.
    save_to_tokenizer_filepath : str
        Filepath to use for saving the tokenizer
    dataset_name : str
        Name of the dataset we are training on. Used to denote saved checkpoints
        during training.
    batch_size : int
        Batch size used for training.
    n_epochs : int
        Number of epochs to train our model on.
    learning_rate : float
        Learning rate to use when training.
    min_learning_rate : float
        Minimum learning rate to use when training.
    max_vocab_size : int
        Maximum vocabulary size to use when training.
    min_word_count : int
        Minimum number of times a word might occur for it to be in the vocabulary.
    embedding_dim : int
        Number of latent dimensions to use in the embedding layers.
    max_window_size : int
        Maximum window size to use when generating skip-gram couples.
    num_negative_samples : int
        Number of negative samples to use when generating skip-gram couples.
    sampling_factor : float
        Sampling factor to use when computing the probability of keeping a word during random subsampling of words.
    unigram_exponent_negative_sampling : float
        Which exponent to raise the unigram distribution to when performing negative sampling.
    output_dir : str
        Output directory to save metadata files, checkpoints and intermediate model weights.
    pretrained_model_filepath : str
        Load an already trained word2vec model from file.
    starting_epoch_nr : int
        Epoch number to start the training from.
    train_logs_to_file : bool
        Whether or not to save logs from training to file.
    intermediate_embedding_weights_saves : int
        Number of intermediate saves of embedding weights per epoch during training.
    dynamic_gpu_memory : bool
        Whether or not to enable dynamic GPU memory
    mixed_precision : bool
        Whether or not to use mixed float16 precision while training
        (requires NVIDIA GPU, e.g., RTX, Titan V, V100).
    tensorboard_logs_dir : str
        TensorBoard logs directory
    cpu_only : bool
        Whether or not to train on the CPU only
    """
    if (
        text_data_filepath == ""
        and text_data_dir == ""
        or (text_data_filepath != "" and text_data_dir != "")
    ):
        raise ValueError(
            "Either text_data_filepath or text_data_dir has to be specified."
        )

    if text_data_filepath != "":
        text_data_filepaths = [text_data_filepath]
    else:
        text_data_filepaths = get_all_filepaths(text_data_dir, ".txt")

    if cpu_only:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Only using CPU!")

    # Count number of lines in text data file.
    print("Counting lines in text data files...")
    num_texts = text_files_total_line_count(text_data_filepaths)
    print("Done!")

    if dynamic_gpu_memory and not cpu_only:
        if enable_dynamic_gpu_memory():
            print("Enabled dynamic GPU memory!")

    if mixed_precision:
        tf_mixed_precision.set_policy("mixed_float16")
        print("Enabled mixed precision!")

    # Initialize word2vec instance
    print("Initializing word2vec model...")
    if pretrained_model_filepath != "":
        word2vec = load_model(pretrained_model_filepath)
    else:
        # Initialize tokenizer (and build its vocabulary if necessary)
        if tokenizer_filepath == "":
            tokenizer = Tokenizer()
            tokenizer.build_word_occurrences(
                filepaths=text_data_filepaths,
                num_texts=num_texts,
            )
        else:
            print("Loading tokenizer...")
            tokenizer = load_tokenizer(tokenizer_filepath)
        print("Building vocabulary...")
        tokenizer.build_vocab(
            max_vocab_size=max_vocab_size,
            min_word_count=min_word_count,
            sampling_factor=sampling_factor,
        )
        if save_to_tokenizer_filepath != "":
            print("Done!\nSaving vocabulary...")
            tokenizer.save(save_to_tokenizer_filepath)
        print("Done!")

        word2vec = Word2vec(
            tokenizer=tokenizer,
            embedding_dim=embedding_dim,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            batch_size=batch_size,
            max_window_size=max_window_size,
            num_negative_samples=num_negative_samples,
            unigram_exponent_negative_sampling=unigram_exponent_negative_sampling,
            mixed_precision=mixed_precision,
        )
    print("Done!")

    # Append date/time to output directories.
    output_dir = join(output_dir, datetime.now().strftime("%d-%b-%Y_%H-%M-%S"))
    tensorboard_logs_dir = join(output_dir, tensorboard_logs_dir)

    # Train model
    word2vec.fit(
        text_data_filepaths=text_data_filepaths,
        num_texts=num_texts,
        dataset_name=dataset_name,
        n_epochs=n_epochs,
        output_dir=output_dir,
        tensorboard_logs_dir=tensorboard_logs_dir,
        starting_epoch_nr=starting_epoch_nr,
        train_logs_to_file=train_logs_to_file,
        intermediate_embedding_weights_saves=intermediate_embedding_weights_saves,
    )


if __name__ == "__main__":
    args = parse_args()

    # Perform training
    train_word2vec(
        text_data_filepath=args.text_data_filepath,
        text_data_dir=args.text_data_dir,
        tokenizer_filepath=args.tokenizer_filepath,
        save_to_tokenizer_filepath=args.save_to_tokenizer_filepath,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        max_vocab_size=args.max_vocab_size,
        min_word_count=args.min_word_count,
        embedding_dim=args.embedding_dim,
        max_window_size=args.max_window_size,
        num_negative_samples=args.num_negative_samples,
        sampling_factor=args.sampling_factor,
        unigram_exponent_negative_sampling=args.unigram_exponent_negative_sampling,
        output_dir=args.output_dir,
        pretrained_model_filepath=args.pretrained_model_filepath,
        starting_epoch_nr=args.starting_epoch_nr,
        train_logs_to_file=args.train_logs_to_file,
        intermediate_embedding_weights_saves=args.intermediate_embedding_weights_saves,
        dynamic_gpu_memory=args.dynamic_gpu_memory,
        mixed_precision=args.mixed_precision,
        tensorboard_logs_dir=args.tensorboard_logs_dir,
        cpu_only=args.cpu_only,
    )
