import argparse
import subprocess
import zipfile
from os import makedirs, rename
from os.path import isfile
from os.path import join as join_path
from typing import List

from nltk.tokenize import sent_tokenize, word_tokenize
from text_preprocessing_utils import (preprocess_text, replace_all_numbers,
                                      replace_contractions)
from tqdm import tqdm
from utils import download_from_url


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
        "--dataset_name",
        type=str,
        default="enwik8",
        help="Name of the wikipedia dataset to download from (either enwik8 or enwik9)",
    )
    parser.add_argument(
        "--compressed_dataset_name",
        type=str,
        default="text8",
        help="Name of the compressed wikipedia dataset (either text8 or fil9)",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="raw_data",
        help="Path to the raw data directory (where files will be downloaded to and extracted from)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path of the processed data directory",
    )
    return parser.parse_args()


def load_and_preprocess_data(
    dataset_name: str, compressed_dataset_name: str, raw_data_dir: str, data_dir: str
) -> None:
    """
    Loads and preprocess text8 data for training a Word2vec model.

    Parameters
    ----------
    dataset_name : str
        Name of the wikipedia dataset to use (either enwik8 or enwik9)
    compressed_dataset_name : str
        Name of the compressed wikipedia dataset to use (either text8 or fil9)
    raw_data_dir : str
        Path to the raw data directory (where files will be downloaded to and extracted from)
    data_dir : str
        Path of the processed data directory
    """
    # Ensure data directories exist
    makedirs(raw_data_dir, exist_ok=True)
    makedirs(data_dir, exist_ok=True)

    # Initialize paths
    raw_data_url = f"http://mattmahoney.net/dc/{dataset_name}.zip"
    raw_data_zip_filepath = join_path(raw_data_dir, f"{dataset_name}.zip")
    raw_data_filepath_no_ext = join_path(raw_data_dir, dataset_name)
    raw_data_filepath = f"{raw_data_filepath_no_ext}.txt"
    raw_data_processed_wikifil_filepath = join_path(
        raw_data_dir, f"{compressed_dataset_name}.txt"
    )
    data_filepath = join_path(data_dir, f"{compressed_dataset_name}.txt")

    # Download raw data if not present
    if not isfile(raw_data_zip_filepath):
        print(f"Downloading raw {dataset_name} data...")
        download_from_url(raw_data_url, raw_data_zip_filepath)
        print("Done!")

    # Extract raw data if not present
    if not isfile(raw_data_filepath):
        print("Extracting raw data...")
        with zipfile.ZipFile(raw_data_zip_filepath) as zip_file:
            zip_file.extractall(raw_data_dir)
        rename(raw_data_filepath_no_ext, raw_data_filepath)
        print("Done!")

    # Preprocess output from `wikifil.pl` script
    if not isfile(raw_data_processed_wikifil_filepath):
        print("Running `wikifil.pl` to process raw Wikipedia data...")
        result = subprocess.run(
            ["perl", "scripts/wikifil.pl", raw_data_filepath],
            capture_output=True,
            text=True,
        )
        print("Done!")
        wikifil_output = result.stdout
        with open(raw_data_processed_wikifil_filepath, "w") as file:
            file.write(wikifil_output)
    else:
        with open(raw_data_processed_wikifil_filepath, "r") as file:
            wikifil_output = file.read()

    # To sentences
    print("Sentence tokenizing...")
    text8_sentences = sent_tokenize(wikifil_output)
    print("Done!")

    # Preprocesses sentences into lists of words and filters out sentences with less than
    # `min_sent_word_count` words in them and convert numbers to their textual
    # representations.
    min_sent_word_count = 5
    new_text8_sentences: List[str] = []
    print("Processing sentences...")
    for sent in tqdm(text8_sentences):

        # Preprocess sentence into a list of words
        words = preprocess_text(sent)

        # Filter out sentences that have less than `min_sent_word_count` words in them
        if len(words) < min_sent_word_count:
            continue

        new_text8_sentences.append(" ".join(words))
    print("Done!")

    # Save to file
    print("Saving to file...")
    with open(data_filepath, "w") as file:
        for sent in new_text8_sentences:
            file.write(f"{sent}\n")

    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    load_and_preprocess_data(
        dataset_name=args.dataset_name,
        compressed_dataset_name=args.compressed_dataset_name,
        raw_data_dir=args.raw_data_dir,
        data_dir=args.data_dir,
    )
