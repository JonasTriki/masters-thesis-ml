import argparse

from data_utils import Tokenizer
from utils import get_all_filepaths, load_model, load_tokenizer, text_file_line_count
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
        default=0.0025,
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
        default=5,
        help="Window size to use when generating skip-gram couples",
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
        "keeping a word during random subsampling of words.",
    )
    parser.add_argument(
        "--unigram_exponent_negative_sampling",
        type=float,
        default=3 / 4,
        help="Which exponent to raise the unigram distribution to when performing negative sampling.",
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
    return parser.parse_args()


def train_word2vec_sgns(
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
    sampling_window_size: int,
    num_negative_samples: int,
    sampling_factor: float,
    unigram_exponent_negative_sampling: float,
    model_checkpoints_dir: str,
    pretrained_model_filepath: str,
    starting_epoch_nr: int,
    train_logs_to_file: bool,
    intermediate_embedding_weights_saves: int,
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
    sampling_window_size : int
        Window size to use when generating skip-gram couples.
    num_negative_samples : int
        Number of negative samples to use when generating skip-gram couples.
    sampling_factor : float
        Sampling factor to use when computing the probability of keeping a word during random subsampling of words.
    unigram_exponent_negative_sampling : float
        Which exponent to raise the unigram distribution to when performing negative sampling.
    model_checkpoints_dir : str
        Where to save checkpoints of the model after each epoch.
    pretrained_model_filepath : str
        Load an already trained word2vec model from file.
    starting_epoch_nr : int
        Epoch number to start the training from.
    train_logs_to_file : bool
        Whether or not to save logs from training to file.
    intermediate_embedding_weights_saves : int
        Number of intermediate saves of embedding weights per epoch during training.
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

    # Count number of lines in text data file.
    print("Counting lines in text data files...")
    num_texts = text_file_line_count(text_data_filepaths)
    print("Done!")

    # Initialize tokenizer (and build its vocabulary if necessary)
    if tokenizer_filepath == "":
        tokenizer = Tokenizer(
            max_vocab_size=max_vocab_size,
            min_word_count=min_word_count,
            sampling_factor=sampling_factor,
        )
        print("Building vocabulary...")
        tokenizer.build_vocab(text_data_filepaths, num_texts)
        if save_to_tokenizer_filepath != "":
            print("Done!\nSaving vocabulary...")
            tokenizer.save(save_to_tokenizer_filepath)
    else:
        print("Loading tokenizer...")
        tokenizer = load_tokenizer(tokenizer_filepath)
    print("Done!")

    # Initialize word2vec instance
    print("Initializing word2vec model...")
    if pretrained_model_filepath != "":
        word2vec = load_model(pretrained_model_filepath)
    else:
        word2vec = Word2vec(
            tokenizer=tokenizer,
            embedding_dim=embedding_dim,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            batch_size=batch_size,
            sampling_window_size=sampling_window_size,
            num_negative_samples=num_negative_samples,
            unigram_exponent_negative_sampling=unigram_exponent_negative_sampling,
            model_checkpoints_dir=model_checkpoints_dir,
        )
    print("Done!")

    # Train model
    word2vec.fit(
        text_data_filepaths=text_data_filepaths,
        num_texts=num_texts,
        dataset_name=dataset_name,
        n_epochs=n_epochs,
        starting_epoch_nr=starting_epoch_nr,
        train_logs_to_file=train_logs_to_file,
        intermediate_embedding_weights_saves=intermediate_embedding_weights_saves,
    )


if __name__ == "__main__":
    args = parse_args()

    # Perform training
    train_word2vec_sgns(
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
        sampling_window_size=args.sampling_window_size,
        num_negative_samples=args.num_negative_samples,
        sampling_factor=args.sampling_factor,
        unigram_exponent_negative_sampling=args.unigram_exponent_negative_sampling,
        model_checkpoints_dir=args.model_checkpoints_dir,
        pretrained_model_filepath=args.pretrained_model_filepath,
        starting_epoch_nr=args.starting_epoch_nr,
        train_logs_to_file=args.train_logs_to_file,
        intermediate_embedding_weights_saves=args.intermediate_embedding_weights_saves,
    )
