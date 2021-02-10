import argparse
import gzip
import sys
from os import makedirs
from os.path import isfile, join

import numpy as np
from word2vec_utils import load_word2vec_format

sys.path.append("..")

from utils import download_from_url  # noqa: E402


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
        "--raw_data_dir",
        type=str,
        default="raw_data",
        help="Path to the raw data directory (where files will be downloaded to and extracted from)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory to save processed data",
    )
    return parser.parse_args()


def preprocess_external_word_embeddings(raw_data_dir: str, output_dir: str) -> None:
    """
    Downloads and preprocesses external word embeddings:
    - GoogleNews-vectors-negative300.bin.gz [1]

    Parameters
    ----------
    raw_data_dir : str
        Path to the raw data directory (where files will be downloaded to).
    output_dir : str
        Output directory to save processed data.

    References
    ----------
    .. [1] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean.
       Distributed Representations of Words and Phrases and their Compositionality
       (https://arxiv.org/pdf/1310.4546.pdf). In Proceedings of NIPS, 2013.
    """
    # Ensure output directory exists
    makedirs(output_dir, exist_ok=True)

    # Define filepaths
    google_news_vectors_zip_raw_download_url = "https://filesender.uninett.no/download.php?token=b0aea55e-72a7-4ac0-9409-8d5dbb322505&files_ids=645861"
    google_news_vectors_zip_raw_filename = "GoogleNews-vectors-negative300.bin.gz"
    google_news_vectors_zip_raw_filepath = join(
        raw_data_dir, google_news_vectors_zip_raw_filename
    )
    google_news_vectors_bin_raw_filename = "GoogleNews-vectors-negative300.bin"
    google_news_vectors_bin_raw_filepath = join(
        raw_data_dir, google_news_vectors_bin_raw_filename
    )
    google_news_words_filepath = join(
        output_dir, "GoogleNews-vectors-negative300_words.txt"
    )
    google_news_vectors_filepath = join(
        output_dir, "GoogleNews-vectors-negative300.npy"
    )

    # -- GoogleNews-vectors-negative300.bin.gz --
    if not isfile(google_news_vectors_zip_raw_filepath):
        print(f"Downloading {google_news_vectors_zip_raw_filename}...")
        download_from_url(
            url=google_news_vectors_zip_raw_download_url,
            destination_filepath=google_news_vectors_zip_raw_filepath,
        )
        print("Done!")

    if not isfile(google_news_vectors_bin_raw_filepath):
        print(f"Extracting {google_news_vectors_zip_raw_filename}...")
        with gzip.GzipFile(google_news_vectors_zip_raw_filepath, "rb") as gzip_file_raw:
            with open(google_news_vectors_bin_raw_filepath, "wb") as gzip_file_output:
                gzip_file_output.write(gzip_file_raw.read())
        print("Done!")

    # Parse vectors from binary file and save result
    google_news_vectors = load_word2vec_format(
        word2vec_filepath=google_news_vectors_bin_raw_filepath,
        binary=True,
        tqdm_enabled=True,
    )

    # Save words
    with open(google_news_words_filepath, "w") as file:
        for i, word in enumerate(google_news_vectors["words"]):
            if i > 0:
                file.write("\n")
            file.write(word)

    # Save word embeddings
    np.save(google_news_vectors_filepath, google_news_vectors["word_embeddings"])


if __name__ == "__main__":
    args = parse_args()
    preprocess_external_word_embeddings(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
    )
