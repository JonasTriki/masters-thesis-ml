import itertools
import pickle
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm


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
        self._static_vocab_table: Optional[tf.lookup.StaticHashTable] = None

    def __getstate__(self):
        """
        Gets the internal state of the class.
        """
        state = self.__dict__.copy()

        # Remove unpickable static vocabulary table
        del state["_static_vocab_table"]

        return state

    def __setstate__(self, state):
        """
        Sets the internal state of the class.
        """
        self.__dict__.update(state)

        # Initialize static vocabulary table if tokenizer is built
        if self._words is not None:
            self._init_static_vocabulary_table()

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

    def _init_static_vocabulary_table(self):
        """
        Initializes the static vocabulary table for tokenizing tensors of text.
        """
        if self._words is None:
            raise TypeError(
                "List of words is empty. " "Did you forget to build the vocabulary?"
            )

        # Initialize static vocabulary table
        self._static_vocab_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                key_dtype=tf.string,
                keys=self._words,
                value_dtype=tf.int64,
                values=tf.cast(tf.range(len(self._words)), tf.int64),
            ),
            default_value=self._unknown_word_int,
        )

    def _build_word_occurrences(
        self, filepaths: List[str], num_texts: int
    ) -> List[Tuple[str, int]]:
        """
        Builds a list containing word and its word count

        Parameters
        ----------
        filepaths : str
            Filepaths of text files to build on.
        num_texts : int
            Number of texts (or sentences) of the content of `filepath`.

        Returns
        -------
        word_occurrences : list of tuples of str and int
            A list containing a tuple with word and its word count in the text corpus.
        """
        # Read file content and split into words
        lines = []
        for filepath in filepaths:
            with tf.io.gfile.GFile(filepath) as f:
                lines.append(f)
        lines = itertools.chain(*lines)

        word_occurrences = Counter()
        for line in tqdm(
            lines,
            desc="- Building word occurences",
            total=num_texts,
        ):
            word_occurrences.update(line.strip().split())
        print(f"Initial vocabulary size: {len(word_occurrences)}")
        word_occurrences = word_occurrences.most_common(self._max_vocab_size)
        print(f"New vocabulary size after maximization: {len(word_occurrences)}")

        # Exclude words with less than `self._min_word_count` occurrences
        word_occurrences = [
            (word, word_count)
            for word, word_count in tqdm(
                word_occurrences, desc="- Filtering word occurences"
            )
            if word_count >= self._min_word_count
        ]
        print(
            f"Final vocabulary size after filtering on minimum word count: {len(word_occurrences)}"
        )

        return word_occurrences

    def build_vocab(self, filepaths: List[str], num_texts: int) -> None:
        """
        Builds the vocabulary for the tokenizer class.

        Sets the following class variables:
        - word_to_int = Dictionary to lookup a word and get its integer representation
        - int_to_word = Dictionary to lookup a word integer and get the respective word
        - words = List of words sorted by word counts (descending)
        - word_counts = Dictionary to lookup the word count of a word
        - word_keep_probs = List of probabilities of keeping a word during subsampling
        - word_noise_dist = Noise distribution used for negative sampling

        Parameters
        ----------
        filepaths : str
            Filepaths of text files to build the vocabulary on.
        num_texts : int
            Number of texts (or sentences) of the content of `filepath`.
        """
        # Get word occurrences from text file
        word_occurrences = self._build_word_occurrences(filepaths, num_texts)

        # Set vocabulary size
        self._vocab_size = len(word_occurrences)

        # Calculate how many words we have in the text corpus
        self._corpus_size = sum(
            [
                word_count
                for _, word_count in tqdm(
                    word_occurrences, desc="- Computing corpus size"
                )
            ]
        )

        # Set word_to_int, int_to_word, word_counts
        # and word_keep_probs
        self._word_to_int = {}
        self._int_to_word = {}
        self._words = []
        self._word_counts = []
        self._word_keep_probs = []
        for word_idx, (word, word_count) in tqdm(
            enumerate(word_occurrences),
            desc="- Finalizing vocabulary",
            total=len(word_occurrences),
        ):

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

        # Initialize static vocabulary table
        self._init_static_vocabulary_table()

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

    def tokenize_text_tf(self, text: tf.Tensor) -> tf.Tensor:
        """
        Tokenizes a text where each word is separated by a space.

        Parameters
        ----------
        text : tf.Tensor
            Space-separated text tensor to tokenize.

        Returns
        -------
        tokenized_words : tf.Tensor
            Tensor of words tokenized into their integer representations.
        """
        if self._static_vocab_table is None:
            raise TypeError(
                "Static vocabulary lookup table is None."
                "Did you forget to build the vocabulary?"
            )

        # Split text into words
        words = tf.strings.split(text)

        # Tokenize words
        tokenized_words = self._static_vocab_table.lookup(words)

        return tokenized_words

    def save(self, destination_filepath: str) -> None:
        """
        Saves the tokenizer to file.

        Parameters
        ----------
        destination_filepath : str
            Where to save the tokenizer to.
        """
        with open(destination_filepath, "wb") as file:
            pickle.dump(self, file)