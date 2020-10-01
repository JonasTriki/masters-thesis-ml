import argparse
import subprocess
import zipfile
from os import makedirs, rename
from os.path import isdir, isfile
from os.path import join as join_path
from typing import List, Optional

from tqdm import tqdm
from utils import download_from_url
from wikiextractor_utils import wikiextractor_outputs_to_file


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
        "--language",
        type=str,
        default="english",
        help="Language of the wikipedia dump",
    )
    parser.add_argument(
        "--wiki_dump_time",
        type=int,
        default="latest",
        help="Time of the wikipedia dump",
    )
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
    parser.add_argument(
        "--num_output_files",
        type=int,
        default=1,
        help="Number of files to split the output into",
    )
    parser.add_argument(
        "--min_sent_word_count",
        type=int,
        default=5,
        help="Minimum sentence word count",
    )
    parser.add_argument(
        "--max_wikipedia_files",
        type=int,
        default=-1,
        help="Maximum number of wikipedia files to process (-1 denotes all files)",
    )
    return parser.parse_args()


def load_and_preprocess_data(
    language: str,
    wiki_dump_time: str,
    raw_data_dir: str,
    output_dir: str,
    num_output_files: int,
    min_sent_word_count: int,
    max_wikipedia_files: int,
) -> None:
    """
    Loads and preprocess text8 data for training a word2vec model.

    Parameters
    ----------
    language : str
        Language of the wikipedia dump.
    wiki_dump_time : str
        Time of the wikipedia dump.
    raw_data_dir : str
        Path to the raw data directory (where files will be downloaded to and extracted from).
    output_dir : str
        Output directory to save processed data.
    num_output_files : int
        Number of files to split the output into.
    min_sent_word_count : int
        Minimum sentence word count.
    max_wikipedia_files : int
        Maximum number of wikipedia files to process (-1 denotes all files).
    """
    # Ensure data directories exist
    makedirs(raw_data_dir, exist_ok=True)
    makedirs(output_dir, exist_ok=True)

    # Initialize paths
    wiki_name = f"{language[:2]}wiki"
    dataset_name = f"{wiki_name}-{wiki_dump_time}"
    raw_data_url = (
        f"https://dumps.wikimedia.org/{wiki_name}/{wiki_dump_time}/"
        f"{dataset_name}-pages-articles-multistream.xml.bz2"
    )
    raw_data_bz2_filepath = join_path(raw_data_dir, f"{dataset_name}.xml.bz2")
    raw_data_bz2_extracted_dir = join_path(raw_data_dir, f"{dataset_name}_extracted")

    # Download raw data if not present
    if not isfile(raw_data_bz2_filepath):
        print(f"Downloading {wiki_name}-{wiki_dump_time} dump...")
        download_from_url(url=raw_data_url, destination_filepath=raw_data_bz2_filepath)
        print("Done!")

    # Extract raw data if not present
    if not isdir(raw_data_bz2_extracted_dir):
        print(f"Extracting articles from {wiki_name}-{wiki_dump_time} dump...")
        subprocess.run(
            [
                "python",
                "-m",
                "wikiextractor.WikiExtractor",
                "-cb",
                "250K",
                "-o",
                raw_data_bz2_extracted_dir,
                raw_data_bz2_filepath,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        print("Done!")

    print("Combining and processing extracted files into single text file...")
    wikiextractor_outputs_to_file(
        extracted_dir=raw_data_bz2_extracted_dir,
        language=language,
        dataset_name=dataset_name,
        output_dir=output_dir,
        num_output_files=num_output_files,
        max_num_files=max_wikipedia_files,
        min_sent_word_count=min_sent_word_count,
    )
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    load_and_preprocess_data(
        language=args.language,
        wiki_dump_time=args.wiki_dump_time,
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        num_output_files=args.num_output_files,
        min_sent_word_count=args.min_sent_word_count,
        max_wikipedia_files=args.max_wikipedia_files,
    )
