from typing import Dict

import numpy as np


def similar_words_vec(
    word_vec: np.ndarray,
    weights: np.ndarray,
    words: np.ndarray,
    top_n: int = 10,
    skip_first: int = 0,
) -> list:
    """
    Finds the most similar words given a word vector
    """
    word_vec_weights_dotted = word_vec @ weights.T
    word_vec_weights_norm = np.linalg.norm(word_vec) * np.linalg.norm(weights, axis=1)
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
    word: str, word_to_int: Dict[str, int], weights: np.ndarray
) -> np.ndarray:
    """
    Gets the word vector of a word given words and weights
    """
    return weights[word_to_int[word]]


def similar_words(
    word: str,
    weights: np.ndarray,
    word_to_int: Dict[str, int],
    words: np.ndarray,
    top_n: int = 10,
) -> list:
    """
    Finds the most similar words of a given word
    """
    # Get word vector of word
    word_vec = get_word_vec(word, word_to_int, weights)

    return similar_words_vec(word_vec, weights, words, top_n, skip_first=1)
