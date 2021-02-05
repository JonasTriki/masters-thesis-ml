import numpy as np
from tqdm import tqdm


def load_word2vec_format(
    word2vec_filepath: str, binary: bool, tqdm_enabled: bool = False
) -> dict:
    """
    Loads a word2vec model from the original C word2vec format
    (https://code.google.com/archive/p/word2vec/).

    Parameters
    ----------
    word2vec_filepath : str
        Filepath of word2vec model.
    binary : bool
        Whether or not the word2vec model is in binary format.
    tqdm_enabled : bool, optional
        Whether or not tqdm progressbar is enabled (defaults to False).

    Returns
    -------
    result : dict
        Result as a dictionary, containing words (vocabulary) and word embeddings.
    """
    result = {}
    if binary:
        with open(word2vec_filepath, "rb") as file:

            # Parse head
            header = file.readline().decode("utf-8")
            vocab_size, embedding_dim = (int(x) for x in header.split())
            word_vector_embedding_len = np.dtype(np.float32).itemsize * embedding_dim

            # Parse words and word embeddings
            word_embeddings = np.zeros((vocab_size, embedding_dim))
            word_vocabulary = []
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
                word_vocabulary.append(word)

                # Parse word vector
                word_embeddings[i] = np.frombuffer(
                    buffer=file.read(word_vector_embedding_len), dtype=np.float32
                )

        result["words"] = word_vocabulary
        result["word_embeddings"] = word_embeddings

    return result
