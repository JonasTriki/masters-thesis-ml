import bz2
from multiprocessing import Pool, cpu_count
from os.path import join
from typing import Generator, List, Tuple

import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from text_preprocessing_utils import preprocess_text
from tqdm import tqdm

from ..utils import get_all_filepaths_recursively

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


def batch_list_gen(lst: List, batch_size: int) -> Generator[List, None, None]:
    """
    Creates a generator for batching list into chunks of `batch_size`.

    Parameters
    ----------
    lst : List
        List of elements.
    batch_size : int
        Size of batches.

    Yields
    ------
    sub_lst : List
        Batches sublist of `lst`.
    """
    lst_len = len(lst)
    for i in range(0, lst_len, batch_size):
        yield lst[i : min(i + batch_size, lst_len)]


def wikiextractor_outputs_to_file(
    extracted_dir: str,
    language: str,
    dataset_name: str,
    output_dir: str,
    num_output_files: int,
    max_num_files: int,
    min_sent_word_count: int,
) -> None:
    """
    Combines WikiExtractor outputs into text files.

    Parameters
    ----------
    extracted_dir : str
        Location of WikiExtractor outputs.
    language : str
        Language of Wikipedia dump.
    dataset_name : str
        Name of the Wikipedia dataset.
    output_dir : str
        Output directory.
    num_output_files : int
        Number of files to split the output into (-1 denotes maximum number of cores).
    max_num_files : int
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

    # Check if we should default the amount of files the the number of CPUs.
    if num_output_files == -1:
        num_output_files = cpu_count()
    num_output_files_str_len = len(str(num_output_files))

    # Compute how many extracted files to have for each output file
    num_extracted_files_per_output_file = int(len(list_of_files) // num_output_files)

    # Process files using multiprocessing
    with Pool() as pool:
        for i, mp_args in zip(
            range(num_output_files),
            batch_list_gen(process_wiki_files_args, num_extracted_files_per_output_file),
        ):
            output_filepath = join(
                output_dir,
                f"{dataset_name}-{str(i + 1).zfill(num_output_files_str_len)}.txt",
            )
            with open(output_filepath, "w", encoding="utf8") as file:
                for j, result in enumerate(
                    tqdm(
                        pool.imap_unordered(
                            process_wiki_file,
                            mp_args,
                        ),
                        total=num_extracted_files_per_output_file,
                    )
                ):
                    if j > 0:
                        file.write("\n")
                    file.writelines(result)
