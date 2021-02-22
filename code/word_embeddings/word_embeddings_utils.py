import numpy as np
from tqdm import tqdm


def load_word2vec_binary_format(
    word2vec_filepath: str, tqdm_enabled: bool = False
) -> tuple:
    """
    Loads a word2vec model from the original C word2vec format
    (https://code.google.com/archive/p/word2vec/).

    Parameters
    ----------
    word2vec_filepath : str
        Filepath of word2vec model.
    tqdm_enabled : bool, optional
        Whether or not tqdm progressbar is enabled (defaults to False).

    Returns
    -------
    result : tuple
        Tuple of word embeddings and words in vocabulary.
    """
    with open(word2vec_filepath, "rb") as file:

        # Parse head
        header = file.readline().decode("utf-8")
        vocab_size, embedding_dim = (int(x) for x in header.split())
        word_vector_embedding_len = np.dtype(np.float32).itemsize * embedding_dim

        # Parse words and word embeddings
        word_embeddings = np.zeros((vocab_size, embedding_dim))
        words = []
        for i in tqdm(range(vocab_size), disable=not tqdm_enabled):

            # Parse word
            word = []
            while True:
                ch = file.read(1)
                if ch == b" ":
                    break
                if ch != b"\n":
                    word.append(ch)
            word = b"".join(word).decode("utf-8")
            words.append(word)

            # Parse word vector
            word_embeddings[i] = np.frombuffer(
                buffer=file.read(word_vector_embedding_len), dtype=np.float32
            )

    return word_embeddings, words


def load_word_embeddings_text_format(
    word_embeddings_text_filepath: str,
    first_line_header: bool,
    tqdm_enabled: bool = True,
) -> tuple:
    """
    Loads word embeddings from text format (e.g. GloVe or fastText embeddings).

    Parameters
    ----------
    word_embeddings_text_filepath : str
        Filepath of word embeddings text file.
    first_line_header : bool
        Whether or not the first line in the text file contains number of words and
        word vector dimensionality.
    tqdm_enabled : bool, optional
        Whether or not tqdm progressbar is enabled (defaults to False).

    Returns
    -------
    result : tuple
        Tuple of word embeddings and words in vocabulary.
    """
    words = []
    word_embeddings = []
    with open(
        word_embeddings_text_filepath, "r", encoding="utf-8"
    ) as word_embeddings_file:

        # Skip header on first line
        # Parse header on first line
        if first_line_header:
            _ = word_embeddings_file.readline()

        for line in tqdm(word_embeddings_file, disable=not tqdm_enabled):

            # Parse tokens
            tokens = line.rstrip().split(" ")
            word = tokens[0]
            word_vector = list(map(float, tokens[1:]))

            # Append result
            words.append(word)
            word_embeddings.append(word_vector)

        # Convert to Numpy
        words = np.asarray(words)
        word_embeddings = np.asarray(word_embeddings)

    return word_embeddings, words
