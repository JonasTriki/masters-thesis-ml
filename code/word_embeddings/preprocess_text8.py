import zipfile
from os import makedirs, rename
from os.path import isfile
from os.path import join as join_path

from .text_preprocessing_utils import preprocess_text8
from .utils import build_vocabulary, download_from_url, save_vocabulary_to_file

# Constants
data_dir = "data"
makedirs(data_dir, exist_ok=True)
data_zip_filepath = join_path(data_dir, "text8.zip")
data_filepath_no_ext = join_path(data_dir, "text8")
data_filepath = f"{data_filepath_no_ext}.txt"
data_vocab_filepath = join_path(data_dir, "text8_vocab.pickle")
data_url = "http://mattmahoney.net/dc/text8.zip"
vocab_size = None


def load_and_preprocess_data() -> None:
    """
    Loads and preprocess text8 data for training a Word2vec model.
    """
    # Download raw data if not present
    if not isfile(data_zip_filepath):
        print("Downloading text8 data...")
        download_from_url(data_url, data_zip_filepath)
        print("Done!")

    # Extract raw data if not present
    if not isfile(data_filepath):
        print("Extracting raw data...")
        with zipfile.ZipFile(data_zip_filepath) as zip_file:
            zip_file.extractall(data_dir)
        rename(data_filepath_no_ext, data_filepath)
        print("Done!")

    # Preprocess raw data
    with open(data_filepath, "r") as file:
        raw_content = file.read()

    # Build vocabulary
    print("Building vocabulary...")
    (
        data_word_dict,
        data_rev_word_dict,
        data_word_counts_dict,
        data_word_noise_dict,
    ) = build_vocabulary(raw_content, preprocess_text8, vocab_size)

    # Save vocab to file
    print("Saving vocabulary to file...")
    save_vocabulary_to_file(
        data_vocab_filepath,
        data_word_dict,
        data_rev_word_dict,
        data_word_counts_dict,
        data_word_noise_dict,
    )

    print("Done!")


if __name__ == "__main__":
    load_and_preprocess_data()
