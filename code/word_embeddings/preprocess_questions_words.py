import argparse
import os
import pickle
import re

from utils import get_cached_download_text_file


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
    parser.add_argument(
        "--git_commit_sha",
        type=str,
        default="45da685079a9a9a29f976d595e4987bd104eb2ae",
        help="Git commit sha to use when downloading test data",
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

            # Split string of words into list of words and append to list
            words = word_line.split()
            words = [word.lower() for word in words]
            questions_words_split_content.append(words)
        questions_words_content_splits.append(questions_words_split_content)

    # Construct dictionary with question-word entries
    questions_words = {
        questions_words_sections[i][2:]: questions_words_content_splits[i]
        for i in range(len(questions_words_sections))
    }

    return questions_words


def preprocess_questions_words(
    raw_data_dir: str, output_dir: str, git_commit_sha: str
) -> None:
    """
    Downloads and preprocess the questions-words test set from the paper
    "Efficient Estimation of Word Representations in Vector Space" by Mikolov et al.

    Parameters
    ----------
    raw_data_dir : str
        Path to the raw data directory (where files will be downloaded to).
    output_dir : str
        Output directory to save processed data.
    git_commit_sha : str
        Git commit sha to use when downloading test data
    """
    print("Processing questions-words...")

    # Fetch questions-words.txt from Mikolov's word2vec Github repository.
    filename = "questions-words.txt"
    txt_url = (
        f"https://raw.githubusercontent.com/tmikolov/word2vec/{git_commit_sha}/{filename}"
    )
    questions_words_txt = get_cached_download_text_file(txt_url, raw_data_dir, filename)

    # Parse the raw content
    questions_words_dict = parse_question_words(questions_words_txt)
    print("Done!")

    # Save questions-words dict to file
    dest_filename = "questions-words.pkl"
    questions_words_filepath = os.path.join(output_dir, dest_filename)
    print("Saving to file...")
    with open(questions_words_filepath, "wb") as file:
        pickle.dump(questions_words_dict, file)
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    preprocess_questions_words(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        git_commit_sha=args.git_commit_sha,
    )
