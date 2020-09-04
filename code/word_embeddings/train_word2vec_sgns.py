import argparse

from data_utils import Tokenizer
from utils import text_file_into_texts
from word2vec import Word2vec


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
        "--dataset_name",
        type=str,
        default="",
        help="Name of the dataset we are training on. "
        "Used to denote saved checkpoints during training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size used for training",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs to train our model on",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate to use when training",
    )
    parser.add_argument(
        "--max_vocab_size",
        type=int,
        default=None,
        help="Maximum vocabulary size to use when training. Defaults to use all words",
    )
    parser.add_argument(
        "--min_word_count",
        type=int,
        default=10,
        help="Minimum number of times a word might occur for it to be in the vocabulary",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=300,
        help="Number of latent dimensions to use in the embedding layers",
    )
    parser.add_argument(
        "--sampling_window_size",
        type=int,
        default=4,
        help="Window size to use when generating skip-gram couples",
    )
    parser.add_argument(
        "--num_negative_samples",
        type=int,
        default=15,
        help="Number of negative samples to use when generating skip-gram couples",
    )
    parser.add_argument(
        "--sampling_factor",
        type=float,
        default=1e-5,
        help="Sampling factor to use when generating skip-gram couples",
    )
    parser.add_argument(
        "--model_checkpoints_dir",
        type=str,
        default="checkpoints",
        help="Where to save checkpoints of the model after each epoch",
    )
    parser.add_argument(
        "--pretrained_model_filepath",
        type=str,
        default="",
        help="Load an already trained Word2vec model from file",
    )
    parser.add_argument(
        "--starting_epoch_nr",
        type=int,
        default=1,
        help="Epoch number to start the training from",
    )
    return parser.parse_args()


def train_word2vec_sgns(
    text_data_filepath: str,
    dataset_name: str,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    max_vocab_size: int,
    min_word_count: int,
    embedding_dim: int,
    sampling_window_size: int,
    num_negative_samples: int,
    sampling_factor: float,
    model_checkpoints_dir: str,
    pretrained_model_filepath: str,
    starting_epoch_nr: int,
) -> None:
    """
    Trains a Word2vec model using skip-gram negative sampling.

    Parameters
    ----------
    text_data_filepath : str
        Text filepath containing the text we wish to train on.
    dataset_name : str
        Name of the dataset we are training on. Used to denote saved checkpoints
        during training.
    batch_size : int
        Batch size used for training.
    n_epochs : int
        Number of epochs to train our model on.
    learning_rate : float
        Learning rate to use when training.
    max_vocab_size : int
        Maximum vocabulary size to use when training.
    min_word_count : int
        Minimum number of times a word might occur for it to be in the vocabulary.
    embedding_dim : int
        Number of latent dimensions to use in the embedding layers.
    sampling_window_size : int
        Window size to use when generating skip-gram couples.
    num_negative_samples : int
        Number of negative samples to use when generating skip-gram couples.
    sampling_factor : float
        Sampling factor to use when generating skip-gram couples.
    model_checkpoints_dir : str
        Where to save checkpoints of the model after each epoch.
    pretrained_model_filepath : str
        Load an already trained Word2vec model from file.
    starting_epoch_nr : int
        Epoch number to start the training from.
    """
    # Read text from file
    data_texts = text_file_into_texts(text_data_filepath)

    # Initialize tokenizer and build its vocabulary
    tokenizer = Tokenizer(
        max_vocab_size=max_vocab_size,
        min_word_count=min_word_count,
        sampling_factor=sampling_factor,
    )
    print("Building vocabulary...")
    tokenizer.build_vocab(text_data_filepath)
    print("Done!")

    # Initialize Word2vec instance
    print("Initializing Word2vec model...")
    word2vec = Word2vec(
        tokenizer=tokenizer,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        sampling_window_size=sampling_window_size,
        num_negative_samples=num_negative_samples,
        model_checkpoints_dir=model_checkpoints_dir,
    )
    if pretrained_model_filepath != "":
        word2vec.load_model(pretrained_model_filepath)
    print("Done!")

    # Train model
    word2vec.fit(
        texts=data_texts,
        dataset_name=dataset_name,
        n_epochs=n_epochs,
        batch_size=batch_size,
        starting_epoch_nr=starting_epoch_nr,
    )


if __name__ == "__main__":
    args = parse_args()

    # Perform training
    train_word2vec_sgns(
        args.text_data_filepath,
        args.dataset_name,
        args.batch_size,
        args.n_epochs,
        args.learning_rate,
        args.max_vocab_size,
        args.min_word_count,
        args.embedding_dim,
        args.sampling_window_size,
        args.num_negative_samples,
        args.sampling_factor,
        args.model_checkpoints_dir,
        args.pretrained_model_filepath,
        args.starting_epoch_nr,
    )
