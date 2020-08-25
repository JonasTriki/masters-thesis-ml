import os
import pickle
from collections import Counter
from os.path import join as join_path
from typing import Optional, Tuple

import numpy as np
import requests
from tensorflow.keras.models import Model

from text_preprocessing_utils import preprocess_text


def get_cached_download(
    filename: str, data_dir: str, url: str, file_ext: str = ".txt"
) -> str:
    """
    Downloads and caches file from url. Returns content
    of cached file.
    """
    # Create data directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)
    filepath = join_path(data_dir, f"{filename}{file_ext}")

    if not os.path.exists(filepath):
        r = requests.get(url)
        r.encoding = "utf-8"
        file_content = r.text

        # Cache content
        with open(filepath, "w") as file:
            file.write(file_content)
    else:
        with open(filepath, "r") as file:
            file_content = file.read()
    return file_content


def tokenize_words(words: list, word_dict: dict) -> list:
    """
    Tokenizes a list of words into a sequence of
    integers using a word dictionary
    """
    # Convert words into a sequence of integers
    # using the word dictionary table
    tokenized_text = []
    for word in words:
        if word in word_dict:
            tokenized_text.append(word_dict[word])
        else:
            tokenized_text.append(0)  # 0 = unk index

    return tokenized_text


def tokenize_text(text: str, word_dict: dict) -> list:
    """
    Tokenizes text into a sequence of integers using
    a word dictionary
    """
    # Preprocess text and convert into list of words
    words = preprocess_text(text)

    # Tokenize words into sequence of integers
    tokenized_text = tokenize_words(words, word_dict)

    return tokenized_text


def build_vocabulary(
    text: str, max_vocab_size: Optional[int] = None, verbose: bool = False
) -> Tuple[list, dict, dict]:
    """
    Builds a vocabulary for training a Word2vec model
    """
    # Preprocess text and convert into list of words
    words = preprocess_text(text)

    # Count word occurences
    # "unk" refers to a word out of dictionary (unknown word)
    # Counter.most_common returns a list which is sorted by
    # word occurences; this is very convenient for creating a
    # dictionary, which we need.
    word_occurences = [["unk", -1]]
    most_common_words: list = Counter(words).most_common(max_vocab_size)
    word_occurences.extend(most_common_words)
    if verbose:
        print(f"# unique words: {len(word_occurences)}")

    # Create dictionary of all words in vocabulary
    # (excluding the unknown word)
    word_dict = {}
    for i, (word, _) in enumerate(word_occurences[1:]):

        # Add 1 to index to pad for the unknown word
        word_dict[word] = i + 1

    # Reversed dictionary for looking up word from number
    rev_word_dict = {i: word for word in enumerate(words)}

    # Count number of unknown words in text
    # (outside our vocabulary)
    unk_count = 0
    for word in words:
        if word not in word_dict:
            unk_count += 1

    # Set number of unknown words
    word_occurences[0][1] = unk_count

    return word_occurences, word_dict, rev_word_dict


def save_vocabulary_to_file(
    filepath: str, word_occurences: list, word_dict: dict, rev_word_dict: dict
) -> None:
    """
    Saves a vocabulary to file
    """
    vocab_obj = {
        "word_occurences": word_occurences,
        "word_dict": word_dict,
        "rev_word_dict": rev_word_dict,
    }
    with open(filepath, "wb") as file:
        pickle.dump(vocab_obj, file)


def read_vocabulary_from_file(filepath: str) -> Tuple[list, dict, dict]:
    """
    Reads a vocabulary from file
    """
    with open(filepath, "rb") as file:
        vocab_obj = pickle.load(file)
        return (
            vocab_obj["word_occurences"],
            vocab_obj["word_dict"],
            vocab_obj["rev_word_dict"],
        )


def get_words_from_word_dict(word_dict: dict, exclude_unk: bool = True) -> np.ndarray:
    """
    Gets the words of a word vocabulary dict
    """
    # Extract words
    words = list(word_dict.keys())
    words = np.asarray(words)

    return words


def load_text_data_tokenized(
    text_filepath: str, vocab_filepath: str
) -> Tuple[list, dict]:
    """
    Loads text data from file and tokenizes it into
    a sequence of integers using word vocabulary dictionary
    """
    # Read text file
    with open(text_filepath, "r") as file:
        text_file_content = file.read()

    # Load vocabulary
    _, word_dict, _ = read_vocabulary_from_file(vocab_filepath)

    # Tokenize text into sequence of integers
    tokenized_text = tokenize_text(text_file_content, word_dict)

    return tokenized_text, word_dict


def get_target_embedding_weights(
    model: Model, target_embedding_layer_name: str = "target_embedding"
) -> np.ndarray:
    """
    Gets the target embedding weights from a Word2vec model
    """
    # Get target embedding layer
    target_embedding_layer = model.get_layer(name=target_embedding_layer_name)

    # Get weights
    weights = target_embedding_layer.get_weights()[0]

    # Remove first row of matrix since it represents
    # the unknown word, which acts as noise if we use it further
    weights = weights[1:]

    return weights


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


def get_word_vec(word: str, words: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Gets the word vector of a word given words and weights
    """
    print("get_word_vec_idx", np.where(words == word)[0][0])
    return weights[np.where(words == word)[0][0]]


def similar_words(
    word: str, weights: np.ndarray, words: np.ndarray, top_n: int = 10
) -> list:
    """
    Finds the most similar words of a given word
    """
    return similar_words_vec(
        get_word_vec(word, words, weights), weights, words, top_n, skip_first=1
    )
