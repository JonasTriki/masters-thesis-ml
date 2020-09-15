from collections import Counter
from typing import Generator, List, Optional, Tuple

import numpy as np
import tensorflow as tf


class Tokenizer:
    """
    Text tokenization class.
    """

    def __init__(
        self,
        max_vocab_size: Optional[int] = None,
        min_word_count: int = 10,
        sampling_factor: float = 1e-5,
        unknown_word_int: int = -1,
    ) -> None:
        """
        Initializes the Tokenizer class.

        Parameters
        ------–---
        max_vocab_size : int, optional
            Maximum vocabulary size to use (defaults to None,
            i.e. all words in vocabulary).

            If specified, the top `max_vocab_size` words will be taken into account
            when tokenizing texts.
        min_word_count : int, optional
            Minimum word count (defaults to 10).

            Words that have fewer occurrences than `min_word_count`
            will be ignored during tokenization of texts.
        sampling_factor : float, optional
            Sampling factor to use when computing the probability
            of keeping a word during random subsampling of words (defaults to 1e-5).
        unknown_word_int : int, optional
            Integer value to use for characterizing unknown words, i.e words that are
            out of the vocabulary (defaults to -1).
        """
        self._max_vocab_size = max_vocab_size
        self._min_word_count = min_word_count
        self._sampling_factor = sampling_factor
        self._unknown_word_int = unknown_word_int

        self._corpus_size: Optional[int] = None
        self._vocab_size: Optional[int] = None
        self._word_to_int: Optional[dict] = None
        self._int_to_word: Optional[dict] = None
        self._words: Optional[np.ndarray] = None
        self._word_counts: Optional[list] = None
        self._word_keep_probs: Optional[list] = None

    @property
    def vocab_size(self) -> int:
        """
        Gets the vocabulary size.

        Returns
        -------
        vocab_size : int
            Size of the tokenizers vocabulary.

        Raises
        ------
        TypeError
            If the vocabulary has not been built yet.
        """
        if self._vocab_size is None:
            raise TypeError(
                "Vocabulary size is not determined yet. "
                "Did you forget to build the vocabulary?"
            )
        return self._vocab_size

    @property
    def word_keep_probs(self) -> List[float]:
        """
        Gets a list of probabilities of keeping a word during subsampling of texts.
        The list is sorted by the order of the words in the vocabulary (i.e. most common
        words have indices 0, 1, 2, etc.).

        Returns
        -------
        word_keep_probs : list of float
            List of probabilities of keeping a word during sampling of texts.

        Raises
        ------
        TypeError
            If the vocabulary has not been built yet.
        """
        if self._word_keep_probs is None:
            raise TypeError(
                "Word keep probabilities list is empty. "
                "Did you forget to build the vocabulary?"
            )
        return self._word_keep_probs

    @property
    def unknown_word_int(self) -> int:
        """
        Gets the value for denoting an unknown word during tokenization.

        Returns
        -------
        unknown_word_int : int
            Unknown word integer value.
        """
        return self._unknown_word_int

    @property
    def words(self) -> np.ndarray:
        """
        Gets a numpy array of words containing the words of the vocabulary.

        Returns
        -------
        words : np.ndarray
            Numpy array of words from the vocabulary.

        Raises
        ------
        TypeError
            If the vocabulary has not been built yet.
        """
        if self._words is None:
            raise TypeError(
                "List of words is empty. " "Did you forget to build the vocabulary?"
            )
        return self._words

    @property
    def word_counts(self) -> List[int]:
        """
        Gets a list containing word counts sorted by the most occurring word.

        Returns
        -------
        word_counts : np.ndarray
            List containing word counts

        Raises
        ------
        TypeError
            If the vocabulary has not been built yet.
        """
        if self._word_counts is None:
            raise TypeError(
                "List of words counts is empty. "
                "Did you forget to build the vocabulary?"
            )
        return self._word_counts

    @property
    def word_to_int(self) -> dict:
        """
        Gets a dictionary mapping from a word to its integer representation

        Returns
        -------
        word_to_int : dict
            Dictionary for mapping a word to its integer representation

        Raises
        ------
        TypeError
            If the vocabulary has not been built yet.
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
        Gets a dictionary mapping from an integer representation to its word

        Returns
        -------
        int_to_word : dict
            Dictionary for mapping an integer representation to its word

        Raises
        ------
        TypeError
            If the vocabulary has not been built yet.
        """
        if self._int_to_word is None:
            raise TypeError(
                "Integer to word dictionary is None."
                "Did you forget to build the vocabulary?"
            )
        return self._int_to_word

    def _build_word_occurrences(self, filepath: str) -> List[Tuple[str, int]]:
        """
        Builds a list containing word and its word count

        Parameters
        ----------
        filepath : str
            Filepath of text file to build on.

        Returns
        -------
            word_occurrences : list of tuples of str and int
                A list containing a tuple with word and its word count in the text corpus.
        """
        # Read file content and split into words
        with open(filepath, "r") as file:
            file_text_content = file.read()
            file_words = file_text_content.split()

        # Count word occurrences
        word_occurrences = Counter(file_words).most_common(self._max_vocab_size)

        # Exclude words with less than `self._min_word_count` occurrences
        word_occurrences = [
            (word, word_count)
            for word, word_count in word_occurrences
            if word_count >= self._min_word_count
        ]

        return word_occurrences

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

        Parameters
        ----------
        filepath : str
            Filepath of the text file to build the vocabulary on.
        """
        # Get word occurrences from text file
        word_occurrences = self._build_word_occurrences(filepath)

        # Set vocabulary size
        self._vocab_size = len(word_occurrences)

        # Calculate how many words we have in the text corpus
        self._corpus_size = sum([word_count for _, word_count in word_occurrences])

        # Set word_to_int, int_to_word, word_counts
        # and word_keep_probs
        self._word_to_int = {}
        self._int_to_word = {}
        self._words = []
        self._word_counts = []
        self._word_keep_probs = []
        for word_idx, (word, word_count) in enumerate(word_occurrences):

            # Lookup tables
            self._word_to_int[word] = word_idx
            self._int_to_word[word_idx] = word

            # Add word and its count to the lists
            self._words.append(word)
            self._word_counts.append(word_count)

            # Add probability of keeping a word during subsampling
            if word_count > 0:
                word_frequency_frac = word_count / float(self._corpus_size)

                # As specified by word2vec's source code:
                # - https://github.com/tmikolov/word2vec/blob/e092540633572b883e25b367938b0cca2cf3c0e7/word2vec.c#L407 # noqa: E501
                # - https://www.quora.com/How-does-sub-sampling-of-frequent-words-work-in-the-context-of-Word2Vec # noqa: E501
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

    def tokenize_text(self, text: str) -> list:
        """
        Tokenizes a text where each word is separated by a space.

        Parameters
        ----------
        text : str
            Space-separated text to tokenize.

        Returns
        -------
        tokenized_words : list of str
            List of words tokenized into their integer representations.
        """
        if self._word_to_int is None:
            raise TypeError(
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


def tokenized_text_generator(
    texts: list, tokenizer: Tokenizer
) -> Generator[list, None, None]:
    """
    Generator that yields tokenized texts from a list of texts.

    Parameters
    --–-------
    texts : list
        List of texts to tokenize.
    tokenizer : Tokenizer
        Tokenizer instance to use for tokenizing texts.

    Yields
    ------
    tokenized_text : list
        Tokenized text in a list.
    """
    for text in texts:
        yield tokenizer.tokenize_text(text)


def sample_words_from_noise_distribution(
    word_indices_sampling_prob: tf.Tensor, num_words: int
) -> tf.Tensor:
    """
    Sampling words from a noise distribution.

    Parameters
    ----------
    word_indices_sampling_prob : tf.Tensor
        Tensor containing probabilities of sampling a word from the noise distribution.
    num_words : int
        Number of words to sample from noise distribution.

    Returns
    -------
    sample_indices : tf.Tensor
        Tensor containing sampled word indices.
    """
    sampled_indices: tf.Tensor = tf.random.categorical(
        tf.math.log([word_indices_sampling_prob]), num_words
    )[0]
    return sampled_indices


def generate_skip_gram_pairs(
    word_indices: tf.Tensor,
    window_size: int,
) -> tf.Tensor:
    """
    Generates skip-gram target/context pairs.

    Parameters
    ----------
    word_indices : tf.Tensor
        Tokenized words in a Tensor.
    window_size : int
        Number of words to the left and right of the target word to generate positive samples from.
    """

    def skip_gram_pairs_from_word(
        word_index: int, skip_grams_array: tf.TensorArray
    ) -> Tuple[int, tf.TensorArray]:
        """
        Helper method for generating skip-gram target/context pairs from a single word integer.

        Parameters
        ----------
        word_index : int
            Word integer representation of word.
        skip_grams_array : tf.TensorArray
            TensorArray containing generated skip-gram target/context pairs.

        Returns
        -------
        next_word_index : int
            Next word_index to generate from.
        next_skip_grams_array : tf.TensorArray
            TensorArray containing newly generated skip-gram target/context pairs.
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
        positive_samples = tf.stack(
            [tf.fill(tf.shape(context_word_indices), word_int), context_word_indices],
            axis=1,
        )

        return word_index + 1, skip_grams_array.write(word_index, positive_samples)

    size = tf.size(word_indices)
    # initialize a tensor array of length `tf.size(word_indices)`
    init_array = tf.TensorArray(tf.int64, size=size, infer_shape=False)
    _, result_array = tf.while_loop(
        lambda i, ta: i < size, skip_gram_pairs_from_word, [0, init_array]
    )
    instances = tf.cast(result_array.concat(), tf.int64)
    instances.set_shape([None, 2])

    return instances


def subsample_words(
    word_indices: tf.Tensor, word_keep_probs: tf.Tensor, unknown_word_int: int
) -> tf.Tensor:
    """
    Applies subsampling to a tensor of words with a certain probability for each word.

    Parameters
    ----------
    word_indices : tf.Tensor
        Tokenized words in a Tensor.
    word_keep_probs : tf.Tensor
        Tensor containing probabilities of keeping a word during subsampling.

        It is ordered such that the first probability corresponds to the word with the
        most occurrences, the second probability to the second most occurring word, etc.
    unknown_word_int : int
        Word integer representation of the unknown word, e.g. a word outside
        the vocabulary.

    Returns
    -------
    word_indices_subsampled : tf.Tensor
        Tensor containing subsampled word indices.
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
    batch_size: int,
) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset for training a Word2vec model using skip-grams and negative sampling.

    Parameters
    ----------
    texts : list
        List of texts to generate skip-gram target/context pairs from.
    tokenizer : Tokenizer
        Tokenizer instance for tokenizing individual texts.
    sampling_window_size : int
        Number of words to the left and right of a target word during sampling of positive words.
    batch_size : int
        Number of skip-gram target/context pairs to yield for each batch of data.

    Returns
    -------
    dataset : tf.data.Dataset
        Dataset used for yielding skip-gram target/context pairs.
    """

    # Convert word keep probs to tensors
    word_keep_probs_tf = tf.convert_to_tensor(tokenizer.word_keep_probs)

    # Initialize tf.data.Dataset
    dataset = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_generator(
                generator=lambda: tokenized_text_generator(texts, tokenizer),
                output_types=tf.int64,
                output_shapes=[None],
            ),
            tf.data.Dataset.from_tensor_slices(tf.range(len(texts)) / len(texts)),
        )
    )

    # Apply subsampling
    dataset = dataset.map(
        lambda word_indices, sent_percentage: (
            subsample_words(word_indices, word_keep_probs_tf, tokenizer.unknown_word_int),
            sent_percentage,
        ),
    )

    # Filter out texts with less than 2 words in them
    dataset = dataset.filter(
        lambda word_indices, sent_percentage: tf.greater(tf.size(word_indices), 1)
    )

    # Generate skip-gram target/context pairs
    dataset = dataset.map(
        lambda word_indices, sent_percentage: (
            generate_skip_gram_pairs(
                word_indices,
                sampling_window_size,
            ),
            sent_percentage,
        ),
    )

    # Reshape `sent_percentage` to have the same size as `word_indices`
    dataset = dataset.map(
        lambda word_indices, sent_percentage: (
            word_indices,
            tf.fill(tf.shape(word_indices)[:1], sent_percentage),
        ),
    )

    # Create a dataset by unstacking word_indices
    dataset = dataset.flat_map(
        lambda word_indices, sent_percentages: tf.data.Dataset.from_tensor_slices(
            (word_indices, sent_percentages)
        )
    )

    # Perform batching
    dataset = dataset.batch(batch_size, drop_remainder=True)

    def prepare_input_for_training(
        skip_gram_pairs_batch: tf.Tensor, sent_percentages: tf.Tensor
    ):
        """
        Prepares the target/context skip-gram pairs for training. Also converts
        the list of percentages into a single percentage.

        Parameters
        ----------
        skip_gram_pairs_batch : tf.Tensor
            Tensor containing input target/context pairs
        sent_percentages : tf.Tensor
            Tensor containing percentages of training progress
        Returns
        -------
        training_data : tuple of tf.Tensor
            Tuple consisting of targets, contexts and training progress (percentage).
        """

        # Set shape of tf.Tensor and extract targets/contexts
        skip_gram_pairs_batch.set_shape([batch_size, 2])
        input_targets = skip_gram_pairs_batch[:, :1]
        input_contexts = skip_gram_pairs_batch[:, 1:]

        # Ensure that dimensions are correct
        input_targets = tf.squeeze(input_targets, axis=1)
        input_contexts = tf.squeeze(input_contexts, axis=1)

        # Return percentage as a single number
        sent_percentage = sent_percentages[0]
        sent_percentage = tf.cast(sent_percentage, "float32")

        # Combine into tuple
        training_data = (input_targets, input_contexts, sent_percentage)

        return training_data

    # Prepare input for training
    dataset = dataset.map(
        lambda skip_gram_pairs_batch, sent_percentages: prepare_input_for_training(
            skip_gram_pairs_batch, sent_percentages
        )
    )

    # Enable prefetching to prepare data while training
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
