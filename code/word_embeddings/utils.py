import os
from os import listdir
from os.path import isdir, isfile, join
from typing import AnyStr, Callable, Generator, List

import requests
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
    Helper function for reading file in batches (used in `text_file_line_count`).

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


def text_file_line_count(filepath: str) -> int:
    """
    Counts number of lines in a given text file.

    Parameters
    ----------
    filepath : str
        Filepath of the text file to count

    Returns
    -------
    line_count : int
        Number of lines in text file
    """
    f = open(filepath, "rb")
    f_gen = _make_file_gen(f.read)  # f.raw.read
    return sum(buf.count(b"\n") for buf in f_gen)


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
    filepaths = [
        join(root_dir, f)
        for f in listdir(root_dir)
        if isfile(join(root_dir, f)) and f.endswith(file_ext)
    ]
    dirs = [d for d in listdir(root_dir) if isdir(join(root_dir, d))]
    for d in dirs:
        files_in_d = get_all_filepaths_recursively(join(root_dir, d), file_ext)
        if files_in_d:
            for f in files_in_d:
                filepaths.append(join(f))
    return filepaths
