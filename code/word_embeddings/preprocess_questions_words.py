import os
import pickle
import re

from utils import get_cached_download_text_file


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
            questions_words_split_content.append(word_line.split())
        questions_words_content_splits.append(questions_words_split_content)

    # Construct dictionary with question-word entries
    questions_words = {
        questions_words_sections[i][2:]: questions_words_content_splits[i]
        for i in range(len(questions_words_sections))
    }

    return questions_words


def preprocess_questions_words() -> None:
    """
    Preprocesses the "questions-words.txt" file used in the skip-grams paper by
    Mikolov et al.: https://arxiv.org/pdf/1310.4546.pdf.
    """
    print("Processing questions-words...")

    # Fetch questions-words.txt from Mikolov's word2vec Github repository.
    raw_data_dir = "raw_data"
    filename = "questions-words.txt"
    commit_sha = "45da685079a9a9a29f976d595e4987bd104eb2ae"
    txt_url = f"https://raw.githubusercontent.com/tmikolov/word2vec/{commit_sha}/questions-words.txt"
    questions_words_txt = get_cached_download_text_file(txt_url, raw_data_dir, filename)

    # Parse the raw content
    questions_words_dict = parse_question_words(questions_words_txt)
    print("Done!")

    # Save questions-words dict to file
    data_dir = "data"
    dest_filename = "questions-words.pickle"
    questions_words_filepath = os.path.join(data_dir, dest_filename)
    print("Saving to file...")
    with open(questions_words_filepath, "wb") as file:
        pickle.dump(questions_words_dict, file)
    print("Done!")


if __name__ == "__main__":
    preprocess_questions_words()
