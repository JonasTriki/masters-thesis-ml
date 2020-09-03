from typing import Dict

import numpy as np


def similar_words_vec(
    target_word_vec: np.ndarray,
    weights: np.ndarray,
    words: np.ndarray,
    top_n: int = 10,
    skip_first: int = 0,
) -> list:
    """
    Finds the `top_n` words closest to a word vector.

    Parameters
    ----------
    target_word_vec : np.ndarray
        Target word vector to find closest words to.
    weights : np.ndarray
        Numpy matrix (vocabulary size, embedding dim) containing word vectors.
    words : np.ndarray
        Numpy array containing words from the vocabulary.
    top_n : int, optional
        Number of similar words (defaults to 10).
    skip_first : int, optional
        Number of similar words to skip (defaults to 0).

    Returns
    -------
    pairs : list of tuples of str and int
        List of `top_n` similar words and their cosine similarities.
    """
    word_vec_weights_dotted = target_word_vec @ weights.T
    word_vec_weights_norm = np.linalg.norm(target_word_vec) * np.linalg.norm(weights, axis=1)
    cos_sims = word_vec_weights_dotted / word_vec_weights_norm
    cos_sims = np.clip(cos_sims, 0, 1)
    sorted_indices = cos_sims.argsort()[::-1]
    # print(words.shape, weights.shape)
    top_words = words[sorted_indices][skip_first : skip_first + top_n]
    top_sims = cos_sims[sorted_indices][skip_first : skip_first + top_n]

    # Create word similarity pairs
    pairs = list(zip(top_words, top_sims))

    return pairs


def get_word_vec(
    target_word: str, word_to_int: Dict[str, int], weights: np.ndarray
) -> np.ndarray:
    """
    Gets the word vector of a word.

    Parameters
    ----------
    target_word : str
        Target word to find word vector of.
    word_to_int : dict of str and int
        Dictionary mapping from word to its integer representation.
    weights : np.ndarray
        Numpy matrix (vocabulary size, embedding dim) containing word vectors.

    Returns
    -------
    word_vec : np.ndarray
        Word vector of a word
    """
    return weights[word_to_int[target_word]]


def similar_words(
    target_word: str,
    weights: np.ndarray,
    word_to_int: Dict[str, int],
    words: np.ndarray,
    top_n: int = 10,
) -> list:
    """
    Finds the most similar words of a given word

    Parameters
    ----------
    target_word : str
        Target word to find word vector of.
    weights : np.ndarray
        Numpy matrix (vocabulary size, embedding dim) containing word vectors.
    word_to_int : dict of str and int
        Dictionary mapping from word to its integer representation.
    words : np.ndarray
        Numpy array containing words from the vocabulary.
    top_n : int, optional
        Number of similar words (defaults to 10).

    Returns
    -------
    pairs : list of tuples of str and int
        List of `top_n` similar words and their cosine similarities.
    """
    # Get word vector of word
    word_vec = get_word_vec(target_word, word_to_int, weights)

    return similar_words_vec(word_vec, weights, words, top_n, skip_first=1)
