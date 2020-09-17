import bz2
from multiprocessing import Pool
from typing import List, Tuple

import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from text_preprocessing_utils import preprocess_text
from tqdm import tqdm
from utils import get_all_filepaths_recursively

nltk.download("punkt")


def process_wiki_doc_text(doc_text: str, language: str, min_sent_word_count: int) -> str:
    """
    Processes text of a single Wikipedia article.

    Parameters
    ----------
    doc_text : str
        Text of Wikipedia article
    language : str
        Language of Wikipedia article
    min_sent_word_count : int
        Minimum sentence word count. Skips any sentences with less than
        `min_sent_word_count` words in them.

    Returns
    -------
    processed_text : str
        Processed Wikipedia article.
    """
    # Tokenize into sentences
    doc_sentences = sent_tokenize(doc_text)

    # Preprocess each sentence individually and add to text
    processed_text = ""
    for i, sent in enumerate(doc_sentences):

        # Preprocess sentence and convert into list of words
        processed_sent_words = preprocess_text(sent, language)

        # Filter out sentences that have less than `min_sent_word_count` words in them
        if len(processed_sent_words) < min_sent_word_count:
            continue

        if i > 0:
            processed_text += "\n"
        processed_text += " ".join(processed_sent_words)

    return processed_text


def process_wiki_file(args: Tuple[str, str, int]):
    """
    Processes an extracted Wikipedia dump file.

    Parameters
    ----------
    args : tuple of str, str and int
        Tuple consisting of filepath to extracted Wikipedia dump file,
        language of Wikipedia article and minimum number of words to have in a sentence.

    Returns
    -------
    wiki_dump_content : str
        Processed Wikipedia articles.
    """
    filepath, language, min_sent_word_count = args
    with bz2.open(filepath, "rt", encoding="utf8") as bz2_file:

        # Extract text between <doc> xml tags
        soup = BeautifulSoup(bz2_file.read(), "lxml")
        docs = soup.find_all("doc")
        wiki_dump_content = ""
        for i, doc in enumerate(docs):
            processed_text = process_wiki_doc_text(
                doc.text, language, min_sent_word_count
            )

            # Append to result
            if i > 0:
                wiki_dump_content += "\n"
            wiki_dump_content += processed_text

        return wiki_dump_content


def wikiextractor_outputs_to_file(
    extracted_dir: str,
    language: str,
    output_filepath: str,
    max_num_files: int,
    min_sent_word_count: int,
) -> None:
    """
    Combines WikiExtractor outputs into a single text file.

    Parameters
    ----------
    extracted_dir:
        Location of WikiExtractor outputs.
    language:
        Language of Wikipedia dump.
    output_filepath:
        Output filepath.
    max_num_files:
        Maximum number of wikipedia files to process (-1 denotes all files).
    min_sent_word_count : int
        Minimum sentence word count.
    """
    # Get list of files in extracted directory
    list_of_files = get_all_filepaths_recursively(extracted_dir, ".bz2")
    if max_num_files > -1:
        list_of_files = list_of_files[:max_num_files]

    # Prepare arguments for multiprocessing
    process_wiki_files_args = [
        (file, language, min_sent_word_count) for file in list_of_files
    ]

    # Process files using multiprocessing
    with Pool() as pool:
        with open(output_filepath, "w", encoding="utf8") as file:
            for i, result in enumerate(
                tqdm(
                    pool.imap_unordered(
                        process_wiki_file,
                        process_wiki_files_args,
                    ),
                    total=len(list_of_files),
                )
            ):
                if i > 0:
                    file.write("\n")
                file.writelines(result)
