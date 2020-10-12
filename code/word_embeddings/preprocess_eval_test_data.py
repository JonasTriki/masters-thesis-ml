import argparse
import pickle
import re
import tarfile
from os import makedirs
from os.path import isdir, isfile, join

from tqdm import tqdm

from ..utils import download_from_url, get_cached_download_text_file


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
        help="Path to the raw data directory (where files will be downloaded to)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory to save processed data",
    )
    return parser.parse_args()


def parse_question_words(questions_words_content: str) -> dict:
    """
    Parses a "questions-words.txt" file into a section-separated dictionary
    for looking up word pairs from each section.

    Parameters
    ----------
    questions_words_content: str
        Raw content of the "questions-words.txt" file

    Returns
    -------
    questions_words: dict
        Dictionary mapping from section to a list of word pairs
    """
    # Parse question words pairs for each section
    questions_words_sections = re.findall(r"(: .+)", questions_words_content)
    questions_words_delimiters = "|".join(questions_words_sections)

    # Split question words content into list
    questions_words_content_splits = []
    for content_split in re.split(questions_words_delimiters, questions_words_content):
        if len(content_split) == 0:
            continue

        content_split_lines = content_split[1 : len(content_split) - 1].split("\n")

        questions_words_split_content: list = []
        for word_line in content_split_lines:

            # Split string of words into tuple of lower-case words and append to list
            words = word_line.split()
            words = tuple([word.lower() for word in words])
            questions_words_split_content.append(words)
        questions_words_content_splits.append(questions_words_split_content)

    # Construct dictionary with question-word entries
    questions_words = {
        questions_words_sections[i][2:]: questions_words_content_splits[i]
        for i in range(len(questions_words_sections))
    }

    return questions_words


def preprocess_questions_words(raw_data_dir: str, output_dir: str) -> None:
    """
    Downloads and preprocess test data for evaluating a word2vec model
    on the Semantic-Syntactic Word Relationship test set (SSWR) from
    Mikolov et al. (https://arxiv.org/pdf/1301.3781.pdf)

    Parameters
    ----------
    raw_data_dir : str
        Path to the raw data directory (where files will be downloaded to).
    output_dir : str
        Output directory to save processed data.
    """
    print("Processing questions-words...")

    # Fetch questions-words.txt from Tensorflow
    filename = "questions-words.txt"
    txt_url = f"http://download.tensorflow.org/data/{filename}"
    questions_words_txt = get_cached_download_text_file(txt_url, raw_data_dir, filename)

    # Parse the raw content
    questions_words_dict = parse_question_words(questions_words_txt)
    print("Done!")

    # Save questions-words dict to file
    dest_filename = "questions-words.pkl"
    questions_words_filepath = join(output_dir, dest_filename)
    print("Saving to file...")
    with open(questions_words_filepath, "wb") as file:
        pickle.dump(questions_words_dict, file)
    print("Done!")


def preprocess_msr(raw_data_dir: str, output_dir: str) -> None:
    """
    Downloads and preprocess test data for evaluating a word2vec model
    on the Microsoft Research Syntactic Analogies Dataset (MSR) from
    Mikolov et al. (https://www.aclweb.org/anthology/N13-1090.pdf)

    Parameters
    ----------
    raw_data_dir : str
        Path to the raw data directory (where files will be downloaded to).
    output_dir : str
        Output directory to save processed data.
    """
    print("Processing MSR...")

    # Initialize paths
    dataset_name = "msr"
    raw_data_url = "https://download.microsoft.com/download/A/B/4/AB4F476B-48A6-47CF-9716-5FF9D0D1F7EA/FeatureAugmentedRNNToolkit-v1.1.tgz"
    raw_data_zip_filepath = join(raw_data_dir, f"{dataset_name}.tgz")
    raw_data_extracted_zip_filepath = join(raw_data_dir, dataset_name)
    output_filepath = join(output_dir, f"{dataset_name}.pkl")

    # Download raw data if not present
    if not isfile(raw_data_zip_filepath):
        print(f"Downloading raw {dataset_name} data...")
        download_from_url(raw_data_url, raw_data_zip_filepath)
        print("Done!")

    # Extract raw data if not present
    if not isdir(raw_data_extracted_zip_filepath):
        print("Extracting raw data...")
        with tarfile.open(raw_data_zip_filepath) as tar_file:
            tar_file.extractall(raw_data_extracted_zip_filepath)
        print("Done!")

    # Read content from extracted zip, process them and combine into one test dataset.
    with open(
        join(raw_data_extracted_zip_filepath, "test_set", "word_relationship.questions"),
        "r",
    ) as file:
        word_relationship_questions = [
            line.split(" ") for line in file.read().split("\n") if len(line) > 0
        ]
    with open(
        join(raw_data_extracted_zip_filepath, "test_set", "word_relationship.answers"),
        "r",
    ) as file:
        word_relationship_answers = [
            line.split(" ") for line in file.read().split("\n") if len(line) > 0
        ]

    # Combine lists
    print("Combining files...")
    word_relationship_questions_answers = {"adjectives": [], "nouns": [], "verbs": []}
    for i in tqdm(range(len(word_relationship_questions))):
        questions = word_relationship_questions[i]
        qa_label, answer = word_relationship_answers[i]

        # Convert from label to category
        qa_category = None
        if qa_label.startswith("J"):
            qa_category = "adjectives"
        elif qa_label.startswith("N"):
            qa_category = "nouns"
        elif qa_label.startswith("V"):
            qa_category = "verbs"

        # Append pair to category
        word_relationship_questions_answers[qa_category].append(questions + [answer])
    print("Done!")

    # Save list of analogies from MSR to file
    print("Saving to file...")
    with open(output_filepath, "wb") as file:
        pickle.dump(word_relationship_questions_answers, file)
    print("Done!")


def preprocess_eval_test_data(raw_data_dir: str, output_dir: str) -> None:
    """
    Downloads and preprocess test data for evaluating a word2vec model.

    Parameters
    ----------
    raw_data_dir : str
        Path to the raw data directory (where files will be downloaded to).
    output_dir : str
        Output directory to save processed data.
    """
    # Ensure data directories exist
    makedirs(raw_data_dir, exist_ok=True)
    makedirs(output_dir, exist_ok=True)

    # Prepare test data sets
    preprocess_questions_words(raw_data_dir, output_dir)
    preprocess_msr(raw_data_dir, output_dir)


if __name__ == "__main__":
    args = parse_args()
    preprocess_eval_test_data(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
    )
