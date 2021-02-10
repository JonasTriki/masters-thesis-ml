import argparse
import sys
import tarfile
import xml.etree.ElementTree as ET
from html import unescape
from multiprocessing import Pool, cpu_count
from os import listdir, makedirs
from os.path import isdir, isfile, join

import joblib
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.append("..")

from text_preprocessing_utils import preprocess_text  # noqa: E402
from utils import (  # noqa: E402
    batch_list_gen,
    download_from_url,
    get_cached_download_text_file,
)


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


def preprocess_topological_polysemy_data(raw_data_dir: str, output_dir: str) -> None:
    """
    Preprocesses data for the topological polysemy paper [1].

    Parameters
    ----------
    raw_data_dir : str
        Path to the raw data directory (where files will be downloaded to and extracted from).
    output_dir : str
        Output directory to save processed data.

    References
    ----------
    .. [1] Alexander Jakubowski, Milica Gašić, & Marcus Zibrowius. (2020).
       Topology of Word Embeddings: Singularities Reflect Polysemy.
    """
    print("Processing TPS paper...")

    # Download data from SemEval-2010 task 14
    semeval_2010_14_data_url = (
        "https://www.cs.york.ac.uk/semeval2010_WSI/files/training_data.tar.gz"
    )
    semeval_2010_14_raw_data_filepath = join(
        raw_data_dir, "semeval_training_data.tar.gz"
    )
    semeval_2010_14_raw_data_dir = join(raw_data_dir, "semeval_training_data")
    semeval_2010_14_nouns_dir = join(
        semeval_2010_14_raw_data_dir, "training_data", "nouns"
    )
    semeval_2010_14_verbs_dir = join(
        semeval_2010_14_raw_data_dir, "training_data", "verbs"
    )
    semeval_2010_14_york_datasets_url = (
        "https://www.cs.york.ac.uk/semeval2010_WSI/datasets.html"
    )
    semeval_2010_14_word_senses_filepath = join(
        output_dir, "semeval_2010_14_word_senses.joblib"
    )
    semeval_2010_14_training_data_sentences_dir = join(
        output_dir, "semeval_2010_14_training_data"
    )

    if not isfile(semeval_2010_14_word_senses_filepath):

        # Scrape website for SemEval gold standard senses
        print("Downloading SemEval 2010 task 14 website...")
        semeval_2010_14_york_datasets_source = get_cached_download_text_file(
            semeval_2010_14_york_datasets_url,
            target_dir=raw_data_dir,
            filename="semeval_2010_14_york_datasets.html",
        )
        semeval_2010_14_york_datasets_soup = BeautifulSoup(
            semeval_2010_14_york_datasets_source, features="lxml"
        )
        semeval_2010_14_york_datasets_tables_soup = (
            semeval_2010_14_york_datasets_soup.find_all("tbody")
        )

        # Scrape tables for word/sense pairs
        semeval_2010_14_word_senses: dict = {"verbs": {}, "nouns": {}, "all": {}}
        for table in semeval_2010_14_york_datasets_tables_soup:
            table_rows = table.find_all("tr")[1:]
            for table_row in table_rows:
                table_cols = table_row.find_all("td")

                # Get word and its GS senses
                target_word = table_cols[0].get_text().strip()
                target_word_is_verb = target_word.endswith(".v")
                target_word = target_word.split(".")[0]
                target_word_senses = int(table_cols[3].get_text().strip())

                if target_word_is_verb:
                    semeval_2010_14_word_senses["verbs"][
                        target_word
                    ] = target_word_senses
                else:
                    semeval_2010_14_word_senses["nouns"][
                        target_word
                    ] = target_word_senses
        semeval_2010_14_word_senses["all"] = {
            **semeval_2010_14_word_senses["verbs"],
            **semeval_2010_14_word_senses["nouns"],
        }

        # Save result
        joblib.dump(semeval_2010_14_word_senses, semeval_2010_14_word_senses_filepath)

    if not isfile(semeval_2010_14_raw_data_filepath):
        print("Downloading training data from SemEval-2010 task 14...")
        download_from_url(semeval_2010_14_data_url, semeval_2010_14_raw_data_filepath)
        print("Done!")

    if not isdir(semeval_2010_14_raw_data_dir):
        print("Extracting raw training data from SemEval-2010 task 14...")
        with tarfile.open(semeval_2010_14_raw_data_filepath) as tar_file:
            tar_file.extractall(semeval_2010_14_raw_data_dir)
        print("Done!")

    if not isdir(semeval_2010_14_training_data_sentences_dir):
        makedirs(semeval_2010_14_training_data_sentences_dir)

        # Default to all CPUs
        num_output_files = cpu_count()

        # Prepare arguments for multiprocessing
        num_output_files_str_len = len(str(num_output_files))
        semeval_2010_14_dirs = [semeval_2010_14_nouns_dir, semeval_2010_14_verbs_dir]
        semeval_2010_14_dir_filepaths = [
            join(semeval_dir, fn)
            for semeval_dir in semeval_2010_14_dirs
            for fn in listdir(semeval_dir)
        ]
        num_xml_files_per_output_file = int(
            len(semeval_2010_14_dir_filepaths) // num_output_files
        )

        print("Processing SemEval-2010 task 14 training data for word2vec...")
        with Pool() as pool:
            for i, mp_args in zip(
                range(num_output_files),
                batch_list_gen(
                    semeval_2010_14_dir_filepaths, num_xml_files_per_output_file
                ),
            ):
                output_filename = f"semeval_2010_task_14-{str(i + 1).zfill(num_output_files_str_len)}.txt"
                output_filepath = join(
                    semeval_2010_14_training_data_sentences_dir, output_filename
                )
                print(f"Writing to {output_filename}...")
                with open(output_filepath, "w", encoding="utf8") as output_semeval_file:
                    for j, result in enumerate(
                        tqdm(
                            pool.imap_unordered(
                                preprocess_semeval_2010_task_14_training_xml_file,
                                mp_args,
                            ),
                            total=num_xml_files_per_output_file,
                        )
                    ):
                        if j > 0:
                            output_semeval_file.write("\n")
                        output_semeval_file.write(result)
        print("Done!")


def preprocess_semeval_2010_task_14_training_xml_file(semeval_filepath: str) -> str:
    """
    Preprocesses a single XML training data file from
    SemEval-2010 task 14.

    Parameters
    ----------
    semeval_filepath : str
        Filepath to .xml training data

    Returns
    -------
    output_sentences : str
        Processed SemEval-2010 task 14 training data sentences.
    """
    xml_tree = ET.parse(semeval_filepath)
    xml_root = xml_tree.getroot()
    j = 0
    output_sentences = ""
    for child in xml_root:
        if child.text is None:
            continue
        clear_text = unescape(child.text)

        # Only replace punctuation and cast sentence to lowercase.
        clean_text_words = preprocess_text(
            text=clear_text,
            should_replace_contractions=False,
            should_remove_digits=False,
            should_replace_numbers=False,
            should_remove_stopwords=False,
        )
        clean_text = " ".join(clean_text_words)

        if j > 0:
            output_sentences += "\n"
        output_sentences += clean_text
        j += 1

    return output_sentences


def preprocess_tda_data(raw_data_dir: str, output_dir: str) -> None:
    """
    Preprocesses data for topological data analysis.

    Parameters
    ----------
    raw_data_dir : str
        Path to the raw data directory (where files will be downloaded to and extracted from).
    output_dir : str
        Output directory to save processed data.
    """
    # Ensure data directories exist
    makedirs(raw_data_dir, exist_ok=True)
    makedirs(output_dir, exist_ok=True)

    # Data from TPS paper
    preprocess_topological_polysemy_data(
        raw_data_dir=raw_data_dir, output_dir=output_dir
    )


if __name__ == "__main__":
    args = parse_args()
    preprocess_tda_data(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
    )
