import argparse
import sys

from word2phrase import Word2phrase

sys.path.append("..")

from utils import get_all_filepaths, text_files_total_line_count  # noqa: E402


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
        "--dataset_name",
        type=str,
        default="",
        help="Name of the dataset we are training on",
    )
    parser.add_argument(
        "--starting_epoch_nr",
        type=int,
        default=1,
        help="Epoch number to start the training from",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=4,
        help="Number of epochs to train our model on (Mikolov et al. recommends 2-4). Defaults to 4.",
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
        help="Minimum number of times a word might occur for it to be in the vocabulary. Defaults to 5.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=200.0,
        help="Threshold for determining whether a given phrase should be included. Defaults to 200.",
    )
    parser.add_argument(
        "--threshold_decay",
        type=float,
        default=0.75,
        help="Value to use for decaying the threshold over time. Defaults to 0.75.",
    )
    parser.add_argument(
        "--phrase_sep",
        type=str,
        default="_",
        help="Separator to use when combining phrases. Defaults to underscore, i.e. _.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory to save the new text data files",
    )
    return parser.parse_args()


def train_word2phrase(
    text_data_filepath: str,
    text_data_dir: str,
    dataset_name: str,
    starting_epoch_nr: int,
    n_epochs: int,
    max_vocab_size: int,
    min_word_count: int,
    threshold: float,
    threshold_decay: float,
    phrase_sep: str,
    output_dir: str,
) -> None:
    """
    Trains word2phrase on a given set of text data files. Word2phrase converts
    Word2phrase converts words that appear frequently together (i.e. phrases)
    into single words. Saves output to the output directory.

    Parameters
    ----------
    text_data_filepath : str
        Text filepath containing the text we wish to train on.
    text_data_dir : str
        Directory containing text files we wish to train on.
    dataset_name : str
        Name of the dataset we are training on.
    starting_epoch_nr : int
        Epoch number to start the training from.
    n_epochs : int
        Number of epochs to train our model on (Mikolov et al. recommends 2-4).
    max_vocab_size : int
        Maximum vocabulary size to use when training. Defaults to use all words
    min_word_count : int
        Minimum number of times a word might occur for it to be in the vocabulary.
    threshold : float
        Threshold for determining whether a given phrase should be included.
    threshold_decay : float
        Value to use for decaying the threshold over time.
    phrase_sep : str
        Separator to use when combining phrases.
    output_dir : str
        Output directory to save the new text data files.
    """
    # Initialize Word2phrase instance
    word2phrase = Word2phrase(
        min_word_count=min_word_count,
        threshold=threshold,
        threshold_decay=threshold_decay,
        phrase_sep=phrase_sep,
    )

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
    num_texts = text_files_total_line_count(text_data_filepaths)
    print(f"Done, {num_texts} lines!")

    # Start training Word2phrase
    word2phrase.fit(
        text_data_filepaths=text_data_filepaths,
        dataset_name=dataset_name,
        starting_epoch_nr=starting_epoch_nr,
        n_epochs=n_epochs,
        num_texts=num_texts,
        max_vocab_size=max_vocab_size,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    args = parse_args()
    train_word2phrase(
        text_data_filepath=args.text_data_filepath,
        text_data_dir=args.text_data_dir,
        dataset_name=args.dataset_name,
        starting_epoch_nr=args.starting_epoch_nr,
        n_epochs=args.n_epochs,
        max_vocab_size=args.max_vocab_size,
        min_word_count=args.min_word_count,
        threshold=args.threshold,
        threshold_decay=args.threshold_decay,
        phrase_sep=args.phrase_sep,
        output_dir=args.output_dir,
    )
