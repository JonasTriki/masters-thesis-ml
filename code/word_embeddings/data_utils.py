import pickle
from collections import Counter
from typing import Generator, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE


class Tokenizer:
    """
    TODO: Docs
    """

    def __init__(
        self,
        max_vocab_size: Optional[int] = None,
        min_word_count: int = 10,
        sampling_factor: float = 1e-5,
        noise_dist_alpha: float = 3 / 4,
        unknown_word_int: int = -1,
    ) -> None:
        """
        TODO: Docs
        """
        self._max_vocab_size = max_vocab_size
        self._min_word_count = min_word_count
        self._sampling_factor = sampling_factor
        self._noise_dist_alpha = noise_dist_alpha
        self._unknown_word_int = unknown_word_int

        self._vocab_size: Optional[int] = None
        self._word_to_int: Optional[dict] = None
        self._int_to_word: Optional[dict] = None
        self._words: Optional[list] = None
        self._word_counts: Optional[dict] = None
        self._word_keep_probs: Optional[list] = None
        self._word_noise_dist: Optional[dict] = None

    @property
    def vocab_size(self) -> int:
        """
        TODO: Docs
        """
        if self._vocab_size is None:
            raise TypeError(
                "Vocabulary size is not determined yet."
                "Did you forget to build the vocabulary?"
            )
        return self._vocab_size

    @property
    def word_negative_sampling_probs(self) -> List[float]:
        """
        TODO: Docs
        """
        if self._word_noise_dist is None:
            raise TypeError(
                "Noise distribution for negative sampling is None."
                "Did you forget to build the vocabulary?"
            )
        return list(self._word_noise_dist.values())

    @property
    def word_keep_probs(self) -> List[float]:
        """
        TODO: Docs
        """
        if self._word_keep_probs is None:
            raise TypeError(
                "Word keep probabilities list is empty."
                "Did you forget to build the vocabulary?"
            )
        return self._word_keep_probs

    @property
    def unknown_word_int(self) -> int:
        """
        TODO: Docs
        """
        return self._unknown_word_int

    @property
    def words(self) -> np.ndarray:
        """
        TODO: Docs
        """
        if self._words is None:
            raise TypeError(
                "List of words is empty." "Did you forget to build the vocabulary?"
            )
        return self._words

    @property
    def word_to_int(self) -> dict:
        """
        TODO: Docs
        """
        if self._word_to_int is None:
            raise TypeError(
                "Word to integer dictionary is None."
                "Did you forget to build the vocabulary?"
            )
        return self._word_to_int

    @property
    def int_to_word(self) -> dict:
        """
        TODO: Docs
        """
        if self._int_to_word is None:
            raise TypeError(
                "Integer to word dictionary is None."
                "Did you forget to build the vocabulary?"
            )
        return self._int_to_word

    def _build_word_occurences(self, filepath: str) -> List[Tuple[str, int]]:
        """
        Builds a list containing word and its word count

        Returns:
            word_occurences: a list containing a tuple with word and
                its word count in the text corpus
        """
        # Read file content and split into words
        with open(filepath, "r") as file:
            file_text_content = file.read()
            file_words = file_text_content.split()

        # Count word occurences
        # word_occurences: List[Tuple[str, int]] = [(self._unknown_word_key, -1)]
        word_occurences = Counter(file_words).most_common(self._max_vocab_size)

        # Exclude words with less than `self._min_word_count` occurences
        word_occurences = [
            (word, word_count)
            for word, word_count in word_occurences
            if word_count >= self._min_word_count
        ]

        return word_occurences

    def _create_noise_distribution(
        self, word_counts: dict, word_to_int: dict, corpus_size: int, alpha: float
    ) -> dict:
        """
        Creates a noise distribution using the word frequencies
        from a dictionary of word counts
        """
        # Normalize word frequencies
        word_freqs_normalized = {
            word_to_int[word]: word_count / corpus_size
            for word, word_count in word_counts.items()
        }

        # Create noise distribution
        noise_dist = {
            word_int: word_freq_frac ** alpha
            for word_int, word_freq_frac in word_freqs_normalized.items()
        }
        Z_noise_dist = sum(noise_dist.values())
        noise_dist_normalized = {
            word_int: word_freq_frac / Z_noise_dist
            for word_int, word_freq_frac in noise_dist.items()
        }

        return noise_dist_normalized

    def build_vocab(self, filepath: str) -> None:
        """
        Builds the vocabulary for the tokenizer class.

        Sets the following class variables:
        - word_to_int = Dictionary to lookup a word and get its integer representation
        - int_to_word = Dictionary to lookup a word integer and get the respective word
        - words = List of words sorted by word counts (descendingly)
        - word_counts = Dictionary to lookup the word count of a word
        - word_keep_probs = List of probabilities of keeping a word during subsampling
        - word_noise_dist = Noise distribution used for negative sampling
        """
        # Get word occurences from text file
        word_occurences = self._build_word_occurences(filepath)

        # Set vocabulary size
        self._vocab_size = len(word_occurences)

        # Calculate how many words we have in the text corpus
        self._corpus_size = sum([word_count for _, word_count in word_occurences])

        # Set word_to_int, int_to_word, word_counts
        # and word_keep_probs
        self._word_to_int = {}
        self._int_to_word = {}
        self._words = []
        self._word_counts = {}
        self._word_keep_probs = []
        for word_idx, (word, word_count) in enumerate(word_occurences):

            # Lookup tables
            self._word_to_int[word] = word_idx
            self._int_to_word[word_idx] = word

            # Add word and its count to the lists
            self._words.append(word)
            self._word_counts[word] = word_count

            # Add probability of keeping a word during subsampling
            if word_count > 0:
                word_frequency_frac = word_count / float(self._corpus_size)

                # As specified by word2vec's source code:
                # - https://github.com/tmikolov/word2vec/blob/e092540633572b883e25b367938b0cca2cf3c0e7/word2vec.c#L407
                # - https://www.quora.com/How-does-sub-sampling-of-frequent-words-work-in-the-context-of-Word2Vec
                keep_prob = (
                    np.sqrt(self._sampling_factor / word_frequency_frac)
                    + self._sampling_factor / word_frequency_frac
                )
                keep_prob = np.minimum(keep_prob, 1.0)
            else:
                keep_prob = 0
            self._word_keep_probs.append(keep_prob)

        # Convert words to numpy array
        self._words = np.asarray(self._words)

        # Create noise distribution for negative sampling
        self._word_noise_dist = self._create_noise_distribution(
            self._word_counts,
            self._word_to_int,
            self._corpus_size,
            self._noise_dist_alpha,
        )

    def tokenize_text(self, text: str) -> list:
        """
        Tokenizes a text where each word is separated by a space.
        """
        if self._word_to_int is None:
            raise Exception(
                "Word to word integer lookup table is None."
                "Did you forget to build the vocabulary?"
            )
        # Split text by space to get words
        words = text.split()

        # Tokenizes the words
        tokenized_words = [
            self._word_to_int[word]
            if word in self._word_to_int
            else self._unknown_word_int
            for word in words
        ]

        return tokenized_words

    def save_vocab(self, vocab_filepath: str) -> None:
        """
        Saves the tokenizer's vocabulary to file
        """
        with open(vocab_filepath, "wb") as file:
            pickle.dump(self.__dict__, file)

    def load_vocab(self, vocab_filepath: str) -> None:
        """
        Loads the tokenizer's vocabulary from file
        """
        with open(vocab_filepath, "rb") as file:
            self.__dict__ = pickle.loads(file.read())


def tokenized_text_generator(
    texts: list, tokenizer: Tokenizer
) -> Generator[list, None, None]:
    """
    TODO: Docs
    """
    for text in texts:
        yield tokenizer.tokenize_text(text)


def sample_words_from_noise_distribution(
    word_indices_sampling_prob: tf.Tensor, num_words: int
) -> tf.Tensor:
    """
    TODO: Docs
    """
    sample_indices = tf.random.categorical(
        tf.math.log([word_indices_sampling_prob]), num_words
    )[0]
    return sample_indices


def generate_skipgram_pairs(
    word_indices: tf.Tensor,
    window_size: int,
    num_negative_samples: int,
    word_indices_sampling_prob: tf.Tensor,
) -> tf.Tensor:
    """
    TODO: Docs
    """

    def skipgram_pairs_from_word(
        word_index: int, init_array: tf.TensorArray
    ) -> Tuple[int, tf.TensorArray]:
        """
        TODO: Docs
        """

        # Get word integer
        word_int = word_indices[word_index]

        # Generate positive samples
        reduced_size = tf.random.uniform([], maxval=window_size, dtype=tf.int32)
        left = tf.range(
            tf.maximum(word_index - window_size + reduced_size, 0), word_index
        )
        right = tf.range(
            word_index + 1,
            tf.minimum(
                word_index + 1 + window_size - reduced_size, tf.size(word_indices)
            ),
        )
        context_indices = tf.concat([left, right], axis=0)
        context_word_indices = tf.gather(word_indices, context_indices)
        # num_positive_samples = context_word_indices.shape[0]
        positive_samples = tf.stack(
            [
                tf.fill(tf.shape(context_word_indices), word_int),
                context_word_indices,
                tf.ones(tf.shape(context_word_indices), dtype=tf.int64),
            ],
            axis=1,
        )

        # Generate negative samples
        # Sample negative samples from noise distribution
        negative_words = sample_words_from_noise_distribution(
            word_indices_sampling_prob,
            num_negative_samples,  # positive_samples.shape[0] * num_negative_samples
        )

        # Merge negative words into target/context --> 0 triples
        negative_samples = tf.stack(
            [
                tf.fill(tf.shape(negative_words), word_int),
                negative_words,
                tf.zeros(tf.shape(negative_words), dtype=tf.int64),
            ],
            axis=1,
        )

        # Merge positive and negative samples
        pairs = tf.concat([positive_samples, negative_samples], axis=0)

        return word_index + 1, init_array.write(word_index, pairs)

    size = tf.size(word_indices)
    # initialize a tensor array of length `tf.size(word_indices)`
    init_array = tf.TensorArray(tf.int64, size=size, infer_shape=False)
    _, result_array = tf.while_loop(
        lambda i, ta: i < size, skipgram_pairs_from_word, [0, init_array]
    )
    instances = tf.cast(result_array.concat(), tf.int64)
    instances.set_shape([None, 3])

    return instances


def subsample_words(
    word_indices: tf.Tensor, word_keep_probs: tf.Tensor, unknown_word_int: int
) -> tf.Tensor:
    """
    TODO: Docs
    """

    # Filter out unknown words
    word_indices_filtered = tf.boolean_mask(
        word_indices, tf.not_equal(word_indices, unknown_word_int)
    )
    word_keep_probs_filtered = tf.gather(word_keep_probs, word_indices_filtered)

    # Generate random values from 0 to 1 to determine
    # if we keep or discard a word, with probabilities
    # defined in `word_keep_probs_filtered`
    rng_values = tf.random.uniform(
        tf.shape(word_keep_probs_filtered), 0, 1, dtype=tf.float64
    )

    # Subsample word indices
    word_indices_subsampled = tf.boolean_mask(
        word_indices_filtered, tf.less(rng_values, word_keep_probs_filtered)
    )

    return word_indices_subsampled


# Create dataset
def create_dataset(
    texts: list,
    tokenizer: Tokenizer,
    sampling_window_size: int,
    num_negative_samples: int,
    batch_size: int,
) -> tf.data.Dataset:
    """
    TODO: Docs
    """

    # Convert word noise distribution and word keep probs to tensors
    word_indices_sampling_prob_tf = tf.convert_to_tensor(
        tokenizer.word_negative_sampling_probs
    )
    word_keep_probs_tf = tf.convert_to_tensor(tokenizer.word_keep_probs)

    # Initialize tf.data.Dataset
    dataset = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_generator(
                generator=lambda: tokenized_text_generator(texts, tokenizer),
                output_types=tf.int64,
                output_shapes=[None],
            ),
            tf.data.Dataset.from_tensor_slices(tf.range(len(texts))),
        )
    )

    # Apply subsampling
    dataset = dataset.map(
        lambda word_indices, sent_idx: (
            subsample_words(word_indices, word_keep_probs_tf, tokenizer.unknown_word_int),
            sent_idx,
        ),
    )

    # Filter out texts with less than 2 words in them
    dataset = dataset.filter(
        lambda word_indices, sent_idx: tf.greater(tf.size(word_indices), 1)
    )

    # Generate skipgram target/context pairs
    dataset = dataset.map(
        lambda word_indices, sent_idx: (
            generate_skipgram_pairs(
                word_indices,
                sampling_window_size,
                num_negative_samples,
                word_indices_sampling_prob_tf,
            ),
            sent_idx,
        ),
    )

    # Reshape `sent_idx` to have the same size as `word_indices`
    dataset = dataset.map(
        lambda word_indices, sent_idx: (
            word_indices,
            tf.fill(tf.shape(word_indices)[:1], sent_idx),
        ),
    )

    # Create a dataset by unstacking word_indices
    dataset = dataset.flat_map(
        lambda word_indices, progress: tf.data.Dataset.from_tensor_slices(
            (word_indices, progress)
        )
    )

    # Perform batching
    dataset = dataset.batch(batch_size)  # , drop_remainder=True

    # Enable prefetching to prepare data while training
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset
