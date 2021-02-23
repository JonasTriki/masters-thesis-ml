import argparse
import gzip
import sys
import zipfile
from os import makedirs
from os.path import isfile, join

import annoy
import numpy as np
from tqdm import tqdm
from word_embeddings_utils import (
    load_word2vec_binary_format,
    load_word_embeddings_text_format,
)

sys.path.append("..")

from utils import download_from_url  # noqa: E402


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
    parser.add_argument(
        "--annoy_index_n_trees",
        type=int,
        default="",
        help="Number of trees to pass to Annoy's build method. More trees => higher precision",
    )
    return parser.parse_args()


def preprocess_google_news(
    raw_data_dir: str, output_dir: str, annoy_index_n_trees: int
) -> None:
    """
    Downloads and preprocessed external word embeddings from [1].

    Parameters
    ----------
    raw_data_dir : str
        Path to the raw data directory (where files will be downloaded to).
    output_dir : str
        Output directory to save processed data.
    annoy_index_n_trees : int
        Number of trees to pass to Annoys build method. More trees => higher precision.

    References
    ----------
    .. [1] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean.
       Distributed Representations of Words and Phrases and their Compositionality
       (https://arxiv.org/pdf/1310.4546.pdf). In Proceedings of NIPS, 2013.
    """
    # Ensure output directory exists
    output_dir = join(output_dir, "GoogleNews")
    makedirs(output_dir, exist_ok=True)

    # Define filepaths
    google_news_vectors_zip_raw_download_url = "https://filesender.uninett.no/download.php?token=b0aea55e-72a7-4ac0-9409-8d5dbb322505&files_ids=645861"
    google_news_vectors_zip_raw_filename = "GoogleNews-vectors-negative300.bin.gz"
    google_news_vectors_zip_raw_filepath = join(
        raw_data_dir, google_news_vectors_zip_raw_filename
    )
    google_news_vectors_bin_raw_filepath = join(
        raw_data_dir, "GoogleNews-vectors-negative300.bin"
    )
    google_news_words_filepath = join(
        output_dir, "GoogleNews-vectors-negative300_words.txt"
    )
    google_news_vectors_filepath = join(
        output_dir, "GoogleNews-vectors-negative300.npy"
    )
    google_news_normalized_vectors_filepath = join(
        output_dir, "GoogleNews-vectors-negative300_normalized.npy"
    )
    google_news_vectors_annoy_index_filepath = join(
        output_dir, "GoogleNews-vectors-negative300_annoy_index.ann"
    )

    # -- GoogleNews-vectors-negative300.bin.gz --
    if not isfile(google_news_vectors_zip_raw_filepath):
        print(f"Downloading {google_news_vectors_zip_raw_filename}...")
        download_from_url(
            url=google_news_vectors_zip_raw_download_url,
            destination_filepath=google_news_vectors_zip_raw_filepath,
        )
        print("Done!")

    if not isfile(google_news_vectors_bin_raw_filepath):
        print(f"Extracting {google_news_vectors_zip_raw_filename}...")
        with gzip.GzipFile(google_news_vectors_zip_raw_filepath, "rb") as gzip_file_raw:
            with open(google_news_vectors_bin_raw_filepath, "wb") as gzip_file_output:
                gzip_file_output.write(gzip_file_raw.read())
        print("Done!")

    # Parse vectors from binary file and save result
    should_load_vectors = (
        not isfile(google_news_words_filepath)
        or not isfile(google_news_vectors_filepath)
        or not isfile(google_news_normalized_vectors_filepath)
    )
    if should_load_vectors:
        google_news_word_embeddings, google_news_words = load_word2vec_binary_format(
            word2vec_filepath=google_news_vectors_bin_raw_filepath,
            tqdm_enabled=True,
        )

    # Save words
    if not isfile(google_news_words_filepath):
        with open(google_news_words_filepath, "w") as file:
            for i, word in enumerate(google_news_words):
                if i > 0:
                    file.write("\n")
                file.write(word)

    # Save word embeddings
    if not isfile(google_news_vectors_filepath):
        np.save(google_news_vectors_filepath, google_news_word_embeddings)

    # Save normalized word embeddings
    google_news_word_embeddings_normalized = None
    if not isfile(google_news_normalized_vectors_filepath):
        google_news_word_embeddings_normalized = (
            google_news_word_embeddings
            / np.linalg.norm(google_news_word_embeddings, axis=1).reshape(-1, 1)
        )
        np.save(
            google_news_normalized_vectors_filepath,
            google_news_word_embeddings_normalized,
        )

    if not isfile(google_news_vectors_annoy_index_filepath):
        if google_news_word_embeddings_normalized is None:
            google_news_word_embeddings_normalized = np.load(
                google_news_normalized_vectors_filepath
            )
        vocab_size, embedding_dim = google_news_word_embeddings_normalized.shape

        # Add word embeddings to index and build it
        ann_index = annoy.AnnoyIndex(f=embedding_dim, metric="euclidean")
        print("Adding word embeddings to index...")
        for i in tqdm(range(vocab_size)):
            ann_index.add_item(i, google_news_word_embeddings_normalized[i])
        print("Done!")

        print("Building index...")
        ann_index.build(n_trees=annoy_index_n_trees, n_jobs=-1)
        print("Done!")

        print("Saving to file...")
        ann_index.save(google_news_vectors_annoy_index_filepath)
        print("Done!")


def preprocess_glove(
    raw_data_dir: str, output_dir: str, annoy_index_n_trees: int
) -> None:
    """
    Downloads and preprocessed external word embeddings from [1].

    Parameters
    ----------
    raw_data_dir : str
        Path to the raw data directory (where files will be downloaded to).
    output_dir : str
        Output directory to save processed data.
    annoy_index_n_trees : int
        Number of trees to pass to Annoys build method. More trees => higher precision.

    References
    ----------
    .. [1] Jeffrey Pennington, Richard Socher, & Christopher D. Manning (2014).
       GloVe: Global Vectors for Word Representation. In Empirical Methods in Natural
       Language Processing (EMNLP) (pp. 1532–1543).
    """
    # Ensure output directory exists
    output_dir = join(output_dir, "GloVe")
    makedirs(output_dir, exist_ok=True)

    # Define constants
    glove_data_filename = "glove.840B.300d"
    glove_word_vectors_url = f"http://nlp.stanford.edu/data/{glove_data_filename}.zip"
    glove_word_vectors_raw_zip_filepath = join(
        raw_data_dir, f"{glove_data_filename}.zip"
    )
    glove_word_vectors_raw_txt_filename = f"{glove_data_filename}.txt"
    glove_word_vectors_raw_txt_filepath = join(
        raw_data_dir, glove_word_vectors_raw_txt_filename
    )
    glove_word_vectors_words_filepath = join(
        output_dir, f"{glove_data_filename}_words.txt"
    )
    glove_word_vectors_filepath = join(output_dir, f"{glove_data_filename}.npy")
    glove_word_vectors_normalized_filepath = join(
        output_dir, f"{glove_data_filename}_normalized.npy"
    )
    glove_word_vectors_annoy_index_filepath = join(
        output_dir, f"{glove_data_filename}_annoy_index.ann"
    )

    if not isfile(glove_word_vectors_raw_zip_filepath):
        print(f"Downloading {glove_data_filename}...")
        download_from_url(
            url=glove_word_vectors_url,
            destination_filepath=glove_word_vectors_raw_zip_filepath,
        )
        print("Done!")

    if not isfile(glove_word_vectors_raw_txt_filepath):
        print(f"Extracting {glove_data_filename}...")
        with zipfile.ZipFile(glove_word_vectors_raw_zip_filepath, "r") as zip_ref:
            zip_ref.extractall(raw_data_dir)
        print("Done!")

    # Parse vectors from text file and save result
    should_load_vectors = (
        not isfile(glove_word_vectors_words_filepath)
        or not isfile(glove_word_vectors_filepath)
        or not isfile(glove_word_vectors_normalized_filepath)
    )
    if should_load_vectors:
        glove_word_embeddings, glove_words = load_word_embeddings_text_format(
            word_embeddings_text_filepath=glove_word_vectors_raw_txt_filepath,
            first_line_header=False,
            tqdm_enabled=True,
        )

    # Save words
    if not isfile(glove_word_vectors_words_filepath):
        with open(glove_word_vectors_words_filepath, "w") as file:
            for i, word in enumerate(glove_words):
                if i > 0:
                    file.write("\n")
                file.write(word)

    # Save word embeddings
    if not isfile(glove_word_vectors_filepath):
        np.save(glove_word_vectors_filepath, glove_word_embeddings)

    # Save normalized word embeddings
    glove_word_embeddings_normalized = None
    if not isfile(glove_word_vectors_normalized_filepath):
        glove_word_embeddings_normalized = glove_word_embeddings / np.linalg.norm(
            glove_word_embeddings, axis=1
        ).reshape(-1, 1)
        np.save(
            glove_word_vectors_normalized_filepath,
            glove_word_embeddings_normalized,
        )

    if not isfile(glove_word_vectors_annoy_index_filepath):
        if glove_word_embeddings_normalized is None:
            glove_word_embeddings_normalized = np.load(
                glove_word_vectors_normalized_filepath
            )
        vocab_size, embedding_dim = glove_word_embeddings_normalized.shape

        # Add word embeddings to index and build it
        ann_index = annoy.AnnoyIndex(f=embedding_dim, metric="euclidean")
        print("Adding word embeddings to index...")
        for i in tqdm(range(vocab_size)):
            ann_index.add_item(i, glove_word_embeddings_normalized[i])
        print("Done!")

        print("Building index...")
        ann_index.build(n_trees=annoy_index_n_trees, n_jobs=-1)
        print("Done!")

        print("Saving to file...")
        ann_index.save(glove_word_vectors_annoy_index_filepath)
        print("Done!")


def preprocess_fasttext(
    raw_data_dir: str, output_dir: str, annoy_index_n_trees: int
) -> None:
    """
    Downloads and preprocessed external word embeddings from [1].

    Parameters
    ----------
    raw_data_dir : str
        Path to the raw data directory (where files will be downloaded to).
    output_dir : str
        Output directory to save processed data.
    annoy_index_n_trees : int
        Number of trees to pass to Annoys build method. More trees => higher precision.

    References
    ----------
    .. [1] Grave, E., Bojanowski, P., Gupta, P., Joulin, A., & Mikolov, T. (2018).
       Learning Word Vectors for 157 Languages. In Proceedings of the International
       Conference on Language Resources and Evaluation (LREC 2018).
    """
    # Ensure output directory exists
    output_dir = join(output_dir, "fastText")
    makedirs(output_dir, exist_ok=True)

    # Define constants
    fasttext_data_filename = "cc.en.300.vec"
    fasttext_vectors_url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{fasttext_data_filename}.gz"
    fasttext_word_vectors_raw_gzip_filepath = join(
        raw_data_dir, f"{fasttext_data_filename}.gz"
    )
    fasttext_word_vectors_raw_txt_filepath = join(raw_data_dir, fasttext_data_filename)
    fasttext_word_vectors_words_filepath = join(
        output_dir, f"{fasttext_data_filename}_words.txt"
    )
    fasttext_word_vectors_filepath = join(output_dir, f"{fasttext_data_filename}.npy")
    fasttext_word_vectors_normalized_filepath = join(
        output_dir, f"{fasttext_data_filename}_normalized.npy"
    )
    fasttext_word_vectors_annoy_index_filepath = join(
        output_dir, f"{fasttext_data_filename}_annoy_index.ann"
    )

    if not isfile(fasttext_word_vectors_raw_gzip_filepath):
        print(f"Downloading {fasttext_data_filename}...")
        download_from_url(
            url=fasttext_vectors_url,
            destination_filepath=fasttext_word_vectors_raw_gzip_filepath,
        )
        print("Done!")

    if not isfile(fasttext_word_vectors_raw_txt_filepath):
        print(f"Extracting {fasttext_data_filename}...")
        with gzip.GzipFile(
            fasttext_word_vectors_raw_gzip_filepath, "rb"
        ) as gzip_file_raw:
            with open(fasttext_word_vectors_raw_txt_filepath, "wb") as gzip_file_output:
                gzip_file_output.write(gzip_file_raw.read())
        print("Done!")

    # Parse vectors from text file and save result
    should_load_vectors = (
        not isfile(fasttext_word_vectors_words_filepath)
        or not isfile(fasttext_word_vectors_filepath)
        or not isfile(fasttext_word_vectors_normalized_filepath)
    )
    if should_load_vectors:
        fasttext_word_embeddings, fasttext_words = load_word_embeddings_text_format(
            word_embeddings_text_filepath=fasttext_word_vectors_raw_txt_filepath,
            first_line_header=True,
            tqdm_enabled=True,
        )

    # Save words
    if not isfile(fasttext_word_vectors_words_filepath):
        with open(fasttext_word_vectors_words_filepath, "w") as file:
            for i, word in enumerate(fasttext_words):
                if i > 0:
                    file.write("\n")
                file.write(word)

    # Save word embeddings
    if not isfile(fasttext_word_vectors_filepath):
        np.save(fasttext_word_vectors_filepath, fasttext_word_embeddings)

    # Save normalized word embeddings
    fasttext_word_embeddings_normalized = None
    if not isfile(fasttext_word_vectors_normalized_filepath):
        fasttext_word_embeddings_normalized = fasttext_word_embeddings / np.linalg.norm(
            fasttext_word_embeddings, axis=1
        ).reshape(-1, 1)
        np.save(
            fasttext_word_vectors_normalized_filepath,
            fasttext_word_embeddings_normalized,
        )

    if not isfile(fasttext_word_vectors_annoy_index_filepath):
        if fasttext_word_embeddings_normalized is None:
            fasttext_word_embeddings_normalized = np.load(
                fasttext_word_vectors_normalized_filepath
            )
        vocab_size, embedding_dim = fasttext_word_embeddings_normalized.shape

        # Add word embeddings to index and build it
        ann_index = annoy.AnnoyIndex(f=embedding_dim, metric="euclidean")
        print("Adding word embeddings to index...")
        for i in tqdm(range(vocab_size)):
            ann_index.add_item(i, fasttext_word_embeddings_normalized[i])
        print("Done!")

        print("Building index...")
        ann_index.build(n_trees=annoy_index_n_trees, n_jobs=-1)
        print("Done!")

        print("Saving to file...")
        ann_index.save(fasttext_word_vectors_annoy_index_filepath)
        print("Done!")


def preprocess_external_word_embeddings(
    raw_data_dir: str, output_dir: str, annoy_index_n_trees: int
) -> None:
    """
    Downloads and preprocesses external word embeddings:
    - GoogleNews-vectors-negative300.bin.gz [1]
    - GloVe Common Crawl, 840B tokens 300d (glove.840B.300d.zip) [2]
    - fastText pre-trained English 300d vectors (cc.en.300.bin.gz) [3]

    Parameters
    ----------
    raw_data_dir : str
        Path to the raw data directory (where files will be downloaded to).
    output_dir : str
        Output directory to save processed data.
    annoy_index_n_trees : int
        Number of trees to pass to Annoys build method. More trees => higher precision.

    References
    ----------
    .. [1] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean.
       Distributed Representations of Words and Phrases and their Compositionality
       (https://arxiv.org/pdf/1310.4546.pdf). In Proceedings of NIPS, 2013.
    .. [2] Jeffrey Pennington, Richard Socher, & Christopher D. Manning (2014).
       GloVe: Global Vectors for Word Representation. In Empirical Methods in Natural
       Language Processing (EMNLP) (pp. 1532–1543).
    .. [3] Grave, E., Bojanowski, P., Gupta, P., Joulin, A., & Mikolov, T. (2018).
       Learning Word Vectors for 157 Languages. In Proceedings of the International
       Conference on Language Resources and Evaluation (LREC 2018).
    """
    print("-- Preprocessing Google News... --")
    preprocess_google_news(
        raw_data_dir=raw_data_dir,
        output_dir=output_dir,
        annoy_index_n_trees=annoy_index_n_trees,
    )
    print("-- Preprocessing GloVe... --")
    preprocess_glove(
        raw_data_dir=raw_data_dir,
        output_dir=output_dir,
        annoy_index_n_trees=annoy_index_n_trees,
    )
    print("-- Preprocessing fastText... --")
    preprocess_fasttext(
        raw_data_dir=raw_data_dir,
        output_dir=output_dir,
        annoy_index_n_trees=annoy_index_n_trees,
    )


if __name__ == "__main__":
    args = parse_args()
    preprocess_external_word_embeddings(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        annoy_index_n_trees=args.annoy_index_n_trees,
    )
