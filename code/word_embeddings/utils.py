import os
import pickle
import random
from collections import Counter
from os.path import join as join_path
from typing import Callable, Generator, List, Optional, Tuple

import numpy as np
import requests
from tensorflow.keras.models import Model
from tqdm import tqdm


def download_from_url(
    url: str, destination_filepath: str, chunk_size: int = 1024
) -> None:
    """
    Downloads a file from url to a specific destination filepath
    """
    file_size = int(requests.head(url).headers["Content-Length"])
    pbar = tqdm(total=file_size, initial=0, unit="B", unit_scale=True)
    req = requests.get(url, stream=True)
    with (open(destination_filepath, "ab")) as f:
        for chunk in req.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(chunk_size)
    pbar.close()


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


def tokenize_text(
    text: str, word_dict: dict, preprocess_text: Callable[[str], list]
) -> list:
    """
    Tokenizes text into a sequence of integers using
    a word dictionary
    """
    # Preprocess text and convert into list of words
    words = preprocess_text(text)

    # Tokenize words into sequence of integers
    tokenized_text = tokenize_words(words, word_dict)

    return tokenized_text


def create_noise_distribution(word_counter: dict, alpha: float = 3 / 4) -> dict:
    """
    Creates a noise distribution using the word frequencies
    from a dictionary of word counts
    """
    # Normalize word frequencies
    Z_word_counter = sum(word_counter.values())
    word_freqs_normalized = {
        key: value / Z_word_counter for key, value in word_counter.items()
    }

    # Create noise distribution
    noise_dist = {key: value ** alpha for key, value in word_freqs_normalized.items()}
    Z_noise_dist = sum(noise_dist.values())
    noise_dist_normalized = {
        key: value / Z_noise_dist for key, value in noise_dist.items()
    }

    return noise_dist_normalized


def sample_words_from_noise_distribution(noise_dist: dict, num_words: int) -> np.ndarray:
    """
    Samples words from a noise distribution
    """
    return np.random.choice(
        list(noise_dist.keys()), size=num_words, p=list(noise_dist.values())
    )


def build_vocabulary(
    text: str,
    preprocess_text: Callable[[str], list],
    max_vocab_size: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[dict, dict, dict, dict]:
    """
    Builds a vocabulary for training a Word2vec model
    """
    # Preprocess text and convert into list of words
    words = preprocess_text(text)

    # Count word occurences
    # "UNK" refers to a word out of dictionary (unknown word)
    # It is capitalized to ensure that we have no words
    # that match it in the input text (since we expect
    # the preprocessing to apply lowercase to all words)
    # Counter.most_common returns a list which is sorted by
    # word occurences; this is very convenient for creating a
    # dictionary, which we need.
    word_counter = Counter(words)
    word_occurences = [["UNK", -1]]
    most_common_words: list = word_counter.most_common(max_vocab_size)
    corpus_size = sum([word_count for _, word_count in most_common_words])
    word_occurences.extend(most_common_words)
    if verbose:
        print(f"# unique words: {len(word_occurences)}")

    # Create (reversed) dictionary of all words in vocabulary
    word_dict = {}
    rev_word_dict = {}
    for i, (word, word_count) in enumerate(word_occurences):
        word_dict[word] = i
        rev_word_dict[i] = word

    # Count number of unknown words in text
    # (outside our vocabulary)
    unk_count = 0
    for word in words:
        if word not in word_dict:
            unk_count += 1

    # Set number of unknown words
    word_occurences[0][1] = unk_count

    word_keep_probs = []
    for i, (_, word_count) in enumerate(word_occurences):

        # Compute probability of keeping a word
        sampling_factor = 0.00001
        if word_count > 0:
            frac = word_count / float(corpus_size)

            # As specified by word2vec's source code:
            # - https://github.com/tmikolov/word2vec/blob/20c129af10659f7c50e86e3be406df663beff438/word2vec.c#L407
            # - https://www.quora.com/How-does-sub-sampling-of-frequent-words-work-in-the-context-of-Word2Vec
            keep_prob = np.sqrt(sampling_factor / frac) + sampling_factor / frac
            keep_prob = np.minimum(keep_prob, 1.0)
        else:
            keep_prob = 0
        word_keep_probs.append(keep_prob)

    # Convert word occurences into dictionary
    word_counts_dict = {word_dict[word]: count for word, count in most_common_words}

    # Create noise distribution
    word_noise_dict = create_noise_distribution(word_counts_dict)
    if verbose:
        print(
            "Sample word integers from noise dist",
            sample_words_from_noise_distribution(word_noise_dict, 10),
        )

    return word_dict, rev_word_dict, word_counts_dict, word_noise_dict


def save_vocabulary_to_file(
    filepath: str,
    word_dict: dict,
    rev_word_dict: dict,
    word_counts_dict: dict,
    word_noise_dict: dict,
) -> None:
    """
    Saves a vocabulary to file
    """
    vocab_obj = {
        "word_dict": word_dict,
        "rev_word_dict": rev_word_dict,
        "word_counts_dict": word_counts_dict,
        "word_noise_dict": word_noise_dict,
    }
    with open(filepath, "wb") as file:
        pickle.dump(vocab_obj, file)


def read_vocabulary_from_file(filepath: str) -> Tuple[dict, dict, dict, dict]:
    """
    Reads a vocabulary from file
    """
    with open(filepath, "rb") as file:
        vocab_obj = pickle.load(file)
        return (
            vocab_obj["word_dict"],
            vocab_obj["rev_word_dict"],
            vocab_obj["word_counts_dict"],
            vocab_obj["word_noise_dict"],
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
    text_filepath: str, vocab_filepath: str, preprocess_text: Callable[[str], list]
) -> Tuple[list, dict, dict, dict]:
    """
    Loads text data from file and tokenizes it into
    a sequence of integers using word vocabulary dictionary
    """
    # Read text file
    with open(text_filepath, "r") as file:
        text_file_content = file.read()

    # Load vocabulary
    word_dict, _, word_counts_dict, word_noise_dict = read_vocabulary_from_file(
        vocab_filepath
    )

    # Tokenize text into sequence of integers
    tokenized_text = tokenize_text(text_file_content, word_dict, preprocess_text)

    return tokenized_text, word_dict, word_counts_dict, word_noise_dict


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

    # TODO: Remove this
    # Remove first row of matrix since it represents
    # the unknown word, which acts as noise if we use it further
    # weights = weights[1:]

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


def get_word_vec(word: str, word_to_int: dict, weights: np.ndarray) -> np.ndarray:
    """
    Gets the word vector of a word given words and weights
    """
    return weights[word_to_int[word]]


def similar_words(
    word: str, weights: np.ndarray, word_to_int: dict, words: np.ndarray, top_n: int = 10
) -> list:
    """
    Finds the most similar words of a given word
    """
    # Get word vector of word
    word_vec = get_word_vec(word, word_to_int, weights)

    return similar_words_vec(word_vec, weights, words, top_n, skip_first=1)


def subsample_words_by_freq(words: list, sampling_factor: float = 1e-5) -> list:
    """
    Subsamples words by using formula 5 from Mikolov et al.:
    https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
    """
    # Create dictionary of probabilities of keeping a word
    # print(f"Old length: {len(words)}")
    word_counts = Counter(words)
    total_num_words = sum(word_counts.values())
    word_freqs = {
        word: word_count / total_num_words for word, word_count in word_counts.items()
    }
    word_keep_prob = {
        word: np.sqrt(sampling_factor / word_freqs[word]) for word in word_counts
    }

    # Subsample words
    words_sub = [word for word in words if np.random.uniform() < word_keep_prob[word]]
    # print(f"New length: {len(words_sub)}")

    return words_sub


def calc_skipgram_pairs_number(
    sequence: list, window_size: int, negative_samples: int,
) -> int:
    """
    Calculates the number of possible generated
    skipgram pairs in a sequence of word integers.

    Uses the general formula:
        window_size * 2 * negative_samples * len(sequence)

    With the exception of corner-cases such as the
    start or the end of a sequence
    """
    num_positive_samples = 0
    for i, wi in enumerate(sequence):

        # Count positive samples
        window_start = max(0, i - window_size)
        window_end = min(len(sequence), i + window_size + 1)
        for j in range(window_start, window_end):
            if j != i:
                num_positive_samples += 1

    # For each positive sample, we generate 1 positive
    # sample plus `negative_samples`
    return (negative_samples + 1) * num_positive_samples


def skipgram_pairs_generator(
    sequence: list,
    word_noise_dict: dict,
    window_size: int = 4,
    negative_samples: int = 1,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Generator[list, None, None]:
    """Generates skipgram word pairs.

    Code is from Tensorflow's `skipgram` method:
    https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/skipgrams
    
    It is modified/optimized to fit our needs.
    
    This function transforms a sequence of word indexes (list of integers)
    into tuples of words of the form:

    - (word, word in the same window), with label 1 (positive samples).
    - (word, random word from the vocabulary), with label 0 (negative samples).

    Read more about Skipgram in this gnomic paper by Mikolov et al.:
    [Efficient Estimation of Word Representations in
    Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)
    """
    for i, wi in enumerate(sequence):
        pairs = []

        # Add positive samples
        window_start = max(0, i - window_size)
        window_end = min(len(sequence), i + window_size + 1)
        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                pairs.append([wi, wj, 1])

        # Add negative samples
        if negative_samples > 0:
            num_negative_samples = len(pairs) * negative_samples
            words = [p[0] for p in pairs]
            # random.shuffle(words)

            negative_words = sample_words_from_noise_distribution(
                word_noise_dict, num_negative_samples
            )
            pairs += [
                [words[i % len(words)], negative_words[i], 0]
                for i in range(num_negative_samples)
            ]

        # if shuffle:
        #    if seed is None:
        #        seed = random.randint(0, int(10e6))
        #    random.seed(seed)
        #    random.shuffle(pairs)

        for pair in pairs:
            yield pair
