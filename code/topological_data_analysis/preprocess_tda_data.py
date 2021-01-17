import argparse
import sys
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter
from os import listdir, makedirs
from os.path import isdir, isfile, join

import joblib
from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import wordnet as wn
from tqdm import tqdm

sys.path.append("..")

from utils import download_from_url, get_cached_download_text_file


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
    semeval_2010_14_raw_data_filepath = join(raw_data_dir, "semeval_training_data.tar.gz")
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
    semeval_2010_14_vocabulary_filepath = join(
        output_dir, "semeval_2010_14_vocabulary.joblib"
    )
    semeval_2010_14_wordnet_senses_filepath = join(
        output_dir, "semeval_2010_14_wordnet_senses.joblib"
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
        semeval_2010_14_word_senses = {"verbs": {}, "nouns": {}, "all": {}}
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
                    semeval_2010_14_word_senses["verbs"][target_word] = target_word_senses
                else:
                    semeval_2010_14_word_senses["nouns"][target_word] = target_word_senses
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

    if not isfile(semeval_2010_14_vocabulary_filepath):

        # Build vocabulary of SemEval 2010 task 14 train data
        semeval_2010_14_counter = Counter()
        for semeval_2010_14_dir in [semeval_2010_14_nouns_dir, semeval_2010_14_verbs_dir]:
            print(f"Iterating over {semeval_2010_14_dir}...")
            semeval_2010_14_dir_filepaths = [
                join(semeval_2010_14_dir, fn) for fn in listdir(semeval_2010_14_dir)
            ]
            for filepath in tqdm(semeval_2010_14_dir_filepaths):
                xml_tree = ET.parse(filepath)
                xml_root = xml_tree.getroot()
                for child in xml_root:
                    for sentence in sent_tokenize(child.text):
                        if sentence.endswith("."):
                            sentence = sentence[: len(sentence) - 1]
                        words_in_sentence = [
                            word.lower() for word in word_tokenize(sentence)
                        ]
                        semeval_2010_14_counter.update(words_in_sentence)

        # Save vocabulary
        joblib.dump(semeval_2010_14_counter, semeval_2010_14_vocabulary_filepath)
    else:
        semeval_2010_14_counter = joblib.load(semeval_2010_14_vocabulary_filepath)

    if not isfile(semeval_2010_14_wordnet_senses_filepath):

        # Save WordNet synsets that are both in the SemEval 2010 task 14
        # vocabulary as well as WordNet.
        semeval_2010_14_words_in_wordnet = {}
        print("Iterating over SemEval 2010 task 14 vocabulary finding WordNet synsets...")
        for word in tqdm(semeval_2010_14_counter):
            num_synsets_word = len(wn.synsets(word))
            if num_synsets_word > 0:
                semeval_2010_14_words_in_wordnet[word] = num_synsets_word

        # Save senses
        joblib.dump(
            semeval_2010_14_words_in_wordnet, semeval_2010_14_wordnet_senses_filepath
        )


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
    preprocess_topological_polysemy_data(raw_data_dir=raw_data_dir, output_dir=output_dir)


if __name__ == "__main__":
    args = parse_args()
    preprocess_tda_data(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
    )
