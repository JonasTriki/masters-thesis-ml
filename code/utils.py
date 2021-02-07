import os
import re
from multiprocessing import Pool
from os import listdir
from os.path import isdir, isfile, join
from typing import AnyStr, Callable, Dict, Generator, List, Union

import numpy as np
import requests
from fastdist import fastdist
from tqdm import tqdm


def download_from_url(
    url: str, destination_filepath: str, chunk_size: int = 1024
) -> None:
    """
    Downloads a file from url to a specific destination filepath

    Parameters
    ----------
    url : str
        URL to download file from.
    destination_filepath : str
        Where to save the file after downloading it.
    chunk_size : int, optional
        Chunk size for downloading (default 1024).
    """
    file_size = int(
        requests.head(url, headers={"Accept-Encoding": "identity"}).headers[
            "Content-Length"
        ]
    )
    with tqdm(total=file_size, initial=0, unit="B", unit_scale=True) as progressbar:
        req = requests.get(url, stream=True)
        req.encoding = "utf-8"
        with (open(destination_filepath, "ab")) as f:
            for chunk in req.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progressbar.update(chunk_size)


def get_cached_download_text_file(
    url: str,
    target_dir: str,
    filename: str,
) -> str:
    """
    Downloads and caches text file from url.

    Parameters
    ----------
    url : str
        URL to download file from.
    target_dir : str
        Which directory to save the file
    filename : str
        Name of file to save (including extension).

    Returns
    -------
    file_content : str
        Raw content of file as a string.
    """
    # Create target directory if it does not exist.
    os.makedirs(target_dir, exist_ok=True)
    destination_filepath = join(target_dir, filename)

    if not os.path.exists(destination_filepath):

        # Download file from url to destination filepath
        download_from_url(url, destination_filepath)

    # Read cached content from file
    with open(destination_filepath, "r") as file:
        file_content = file.read()

    return file_content


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


def text_file_into_texts(filepath: str) -> List[str]:
    """
    Reads a text file from disk and splits it into texts delimited by new line.

    Parameters
    ----------
    filepath : str
        Path of file to read

    Returns
    -------
    texts : list of str
        Text content of file split into a list of texts delimited by a new line.
    """
    # Read file
    with open(filepath, "r") as file:
        text_content = file.read()

    # Split into texts
    texts = text_content.split("\n")

    return texts


def _make_file_gen(
    reader: Callable[[int], AnyStr], buffer_size: int = 1024 * 1024
) -> Generator[AnyStr, None, None]:
    """
    Helper function for reading file in batches (used in `text_files_total_line_count`).

    Parameters
    ----------
    reader : Callable[[int], AnyStr]
        file.read function.
    buffer_size : int
        Buffer size.

    Returns
    -------
    b : AnyStr
        File buffer.
    """
    b = reader(buffer_size)
    while b:
        yield b
        b = reader(buffer_size)


def text_file_total_line_count(filepath: str) -> int:
    """
    Counts number of lines in text file.

    Parameters
    ----------
    filepath : str
        Filepath of text file to count.

    Returns
    -------
    line_count : int
        Number of lines in text file
    """
    f = open(filepath, "rb")
    f_gen = _make_file_gen(f.read)
    line_count = sum(buf.count(b"\n") for buf in f_gen)
    f.close()
    return line_count


def text_files_total_line_count(filepaths: List[str]) -> int:
    """
    Counts number of lines in text files.

    Parameters
    ----------
    filepaths : str
        Filepaths of text files to count

    Returns
    -------
    line_count : int
        Number of lines in text files
    """
    total = 0
    with Pool() as pool:
        results = pool.imap_unordered(text_file_total_line_count, filepaths)
        for result in results:
            total += result
    return total


def get_all_filepaths(file_dir: str, file_ext: str) -> List[str]:
    """
    Gets all paths of files of a specific file extension in a directory.

    Parameters
    ----------
    file_dir : str
        Directory containing files.
    file_ext : str
        File extension (including dot).

    Returns
    -------
    filepaths : list of str
        List of filepaths in file directory with given file extension.
    """
    filepaths = [
        join(file_dir, f)
        for f in listdir(file_dir)
        if isfile(join(file_dir, f)) and f.endswith(file_ext)
    ]
    return filepaths


def get_all_filepaths_recursively(root_dir: str, file_ext: str) -> List[str]:
    """
    Gets all paths of files of a specific file extension recursively in a directory.

    Parameters
    ----------
    root_dir : str
        Root directory to start the search from.
    file_ext : str
        File extension (including dot).

    Returns
    -------
    filepaths : list of str
        List of filepaths in root directory with given file extension.
    """
    filepaths = get_all_filepaths(root_dir, file_ext)
    dirs = [
        dir_in_root
        for dir_in_root in listdir(root_dir)
        if isdir(join(root_dir, dir_in_root))
    ]
    for dir in dirs:
        files_in_dir = get_all_filepaths_recursively(join(root_dir, dir), file_ext)
        if files_in_dir:
            for filepath in files_in_dir:

                # Ensure the file size is > 0
                if os.stat(filepath).st_size == 0:
                    continue

                filepaths.append(join(filepath))
    return filepaths


def get_model_checkpoint_filepaths(
    output_dir: str, model_name: str, dataset_name: str
) -> Dict[str, Union[str, List[str]]]:
    """
    Gets model checkpoint filepaths of a specific model (trained on a specific dataset)
    in an output directory.

    Parameters
    ----------
    output_dir : str
        Output directory.
    model_name : str
        Name of the model.
    dataset_name : str
        Name of the dataset which the model has been trained on.

    Returns
    -------
    filepaths_dict : dict
        Dictionary containing filepaths to trained models, intermediate weight embeddings,
        words used during training and training log.
    """
    # List files in output directory
    output_filenames = listdir(output_dir)

    # Filter by model_name and dataset_name entries only
    model_id = f"{model_name}_{dataset_name}"
    output_filenames = [fn for fn in output_filenames if fn.startswith(model_id)]

    # Get model training configuration filepath
    model_training_conf_filepath = join(output_dir, f"{model_id}.conf")

    # Get model filenames and sort them by epoch numbers (from first to last).
    model_filenames = np.array([fn for fn in output_filenames if fn.endswith(".model")])
    model_epoch_nrs = np.array(
        [int(re.findall(r"_(\d{2}).model", fn)[0]) for fn in model_filenames]
    )
    model_filenames = model_filenames[np.argsort(model_epoch_nrs)]

    # Append output directory to filenames
    model_filepaths = [join(output_dir, fn) for fn in model_filenames]

    # Get intermediate embedding weights sorted by first to last
    intermediate_embedding_weight_filenames = np.array(
        [fn for fn in output_filenames if fn.endswith("weights.npy")]
    )
    intermediate_embedding_weight_filepaths = None
    intermediate_embedding_weight_normalized_filepaths = None
    intermediate_embedding_weight_annoy_index_filepaths = None
    if len(intermediate_embedding_weight_filenames) > 0:

        # Extract combined epoch/embedding nrs and sort by them.
        epoch_embedding_nrs = []
        for fn in intermediate_embedding_weight_filenames:
            epoch_nr, embedding_nr = re.findall(r"_(\d{2})_(\d{2})_weights.npy", fn)[0]
            epoch_embedding_nr = int(f"{epoch_nr}{embedding_nr}")
            epoch_embedding_nrs.append(epoch_embedding_nr)
        epoch_embedding_nrs = np.array(epoch_embedding_nrs)
        intermediate_embedding_weight_filenames = intermediate_embedding_weight_filenames[
            np.argsort(epoch_embedding_nrs)
        ]

        # Append output directory to filenames
        intermediate_embedding_weight_filepaths = [
            join(output_dir, fn) for fn in intermediate_embedding_weight_filenames
        ]

        # Check for normalized/Annoy index filepaths
        for fn in intermediate_embedding_weight_filenames:
            fn_no_ext = fn.rsplit(".", 1)[0]
            for output_fn in output_filenames:
                output_filepath = join(output_dir, output_fn)
                if output_fn.startswith(fn_no_ext):
                    if output_fn.endswith("_normalized.npy"):
                        if intermediate_embedding_weight_normalized_filepaths is None:
                            intermediate_embedding_weight_normalized_filepaths = []
                        intermediate_embedding_weight_normalized_filepaths.append(
                            output_filepath
                        )
                    elif output_fn.endswith("_annoy_index.ann"):
                        if intermediate_embedding_weight_annoy_index_filepaths is None:
                            intermediate_embedding_weight_annoy_index_filepaths = []
                        intermediate_embedding_weight_annoy_index_filepaths.append(
                            output_filepath
                        )

    train_words_filename = f"{model_id}_words.txt"
    train_words_filepath = None
    if train_words_filename in output_filenames:
        train_words_filepath = join(output_dir, train_words_filename)

    train_word_counts_filename = f"{model_id}_word_counts.txt"
    train_word_counts_filepath = None
    if train_word_counts_filename in output_filenames:
        train_word_counts_filepath = join(output_dir, train_word_counts_filename)

    # Add path to train logs
    train_logs_filename = f"{model_id}_logs.csv"
    train_logs_filepath = None
    if train_logs_filename in output_filenames:
        train_logs_filepath = join(output_dir, train_logs_filename)

    return {
        "model_training_conf_filepath": model_training_conf_filepath,
        "model_filepaths": model_filepaths,
        "intermediate_embedding_weight_filepaths": intermediate_embedding_weight_filepaths,
        "intermediate_embedding_weight_normalized_filepaths": intermediate_embedding_weight_normalized_filepaths,
        "intermediate_embedding_weight_annoy_index_filepaths": intermediate_embedding_weight_annoy_index_filepaths,
        "train_words_filepath": train_words_filepath,
        "train_word_counts_filepath": train_word_counts_filepath,
        "train_logs_filepath": train_logs_filepath,
    }


def pairwise_cosine_distances(X: np.ndarray) -> np.ndarray:
    """
    Computes pairwise cosine distances.

    Parameters
    ----------
    X : np.ndarray
        Numpy (n-m)-matrix to compute pairwise cosine distances of.

    Returns
    -------
    X_cosine_dists : np.ndarray
        Square (n-n) Numpy matrix containing pairwise cosine distance.
    """
    # Compute pairwise cosine distances (1 - similarity) in X
    X_cosine_dists = 1 - fastdist.cosine_matrix_to_matrix(X, X)

    # Ensure diagonal is filled with zeros
    np.fill_diagonal(X_cosine_dists, 0)

    # Clip values between 0 and 1 to ensure validity
    X_cosine_dists = np.clip(X_cosine_dists, 0, 1)

    return X_cosine_dists


def create_word_embeddings_distances_matrix(
    word_embeddings: np.ndarray,
    vocabulary: np.ndarray,
) -> np.ndarray:
    """
    Creates distance matrix for word embeddings

    Parameters
    ----------
    word_embeddings : np.ndarray
        Word embeddings
    vocabulary : np.ndarray
        Array consisting of word integers to use as vocabulary

    Returns
    -------
    word_embeddings_distances : np.ndarray
        Pairwise cosine distances between word embeddings from vocabulary
    """
    # Compute cosine distance matrix
    word_embeddings_to_precompute = word_embeddings[vocabulary]
    word_embeddings_distances = pairwise_cosine_distances(word_embeddings_to_precompute)
    return word_embeddings_distances


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the cosine distance between two points x and y.

    Parameters
    ----------
    x : np.ndarray
        First point
    y : np.ndarray
        Second point

    Returns
    -------
    cosine_dist : float
        Cosine distance between x and y
    """
    return 1 - fastdist.cosine(x, y)


def cosine_vector_to_matrix_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the cosine distance between a 1D vector x and a
    matrix of vectors y.

    Parameters
    ----------
    x : np.ndarray
        1D vector
    y : np.ndarray
        Matrix

    Returns
    -------
    cosine_dist : np.ndarray
        Cosine distance between x and y as a vector
    """
    return 1 - fastdist.cosine_vector_to_matrix(x, y)


def words_to_vectors(
    words_vocabulary: list,
    word_to_int: dict,
    word_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Gets word embeddings for a list of words.

    Parameters
    ----------
    words_vocabulary : list
        List of words/integers from vocabulary to find word embeddings from
    word_to_int : dict
        Dictionary mapping from word to integer
    word_embeddings : np.ndarray
        Word embeddings

    Returns
    -------
    word_vectors : np.ndarray
        Word embeddings of input words
    """
    # Create word vectors from given words/vocabulary
    if all(isinstance(elem, str) for elem in words_vocabulary):
        word_vectors = np.zeros((len(words_vocabulary), word_embeddings.shape[1]))
        for i, word in enumerate(words_vocabulary):
            word_vectors[i] = word_embeddings[word_to_int[word]]
    elif all(isinstance(elem, int) for elem in words_vocabulary):
        word_vectors = word_embeddings[words_vocabulary]
    else:
        raise TypeError(
            "words_vocabulary argument must contain list of words or integers representing the vocabulary."
        )
    return word_vectors
