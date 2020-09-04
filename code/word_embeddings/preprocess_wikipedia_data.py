import subprocess
import zipfile
from os import makedirs, rename
from os.path import isfile
from os.path import join as join_path
from typing import List

from nltk.tokenize import sent_tokenize, word_tokenize
from text_preprocessing_utils import replace_all_numbers, replace_contractions
from tqdm import tqdm
from utils import download_from_url

# Constants
dataset_name = "enwik8"
compressed_dataset_name = "text8"

raw_data_dir = "raw_data"
makedirs(raw_data_dir, exist_ok=True)
raw_data_url = f"http://mattmahoney.net/dc/{dataset_name}.zip"
raw_data_zip_filepath = join_path(raw_data_dir, f"{dataset_name}.zip")
raw_data_filepath_no_ext = join_path(raw_data_dir, dataset_name)
raw_data_filepath = f"{raw_data_filepath_no_ext}.txt"
raw_data_processed_wikifil_filepath = join_path(
    raw_data_dir, f"{compressed_dataset_name}.txt"
)

data_dir = "data"
makedirs(data_dir, exist_ok=True)
data_filepath = join_path(data_dir, f"{compressed_dataset_name}.txt")


def load_and_preprocess_data() -> None:
    """
    Loads and preprocess text8 data for training a Word2vec model.
    """
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

    # TODO: Do this?
    # Replace contractions in sentences, filter out sentences with less than
    # `min_sent_word_count` words in them and convert numbers to their textual
    # representations.
    min_sent_word_count = 5
    new_text8_sentences: List[str] = []
    print("Processing sentences...")
    for sent in tqdm(text8_sentences):

        # Fix contractions
        sent_clean = replace_contractions(sent)

        # Convert sentence into list of words
        words = word_tokenize(sent_clean)

        # Remove "." if it is in the list of words
        if len(words) > 0 and words[-1] == ".":
            words = words[:-1]

        # Filter out sentences that have less than `min_sent_word_count` words in them
        if len(words) < min_sent_word_count:
            continue

        # Convert numbers to its textual representation.
        words = replace_all_numbers(words)

        new_text8_sentences.append(" ".join(words))
    print("Done!")

    # Save to file
    print("Saving to file...")
    with open(data_filepath, "w") as file:
        for sent in new_text8_sentences:
            file.write(f"{sent}\n")

    print("Done!")


if __name__ == "__main__":
    load_and_preprocess_data()
