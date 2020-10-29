from collections import Counter
from itertools import chain, tee, zip_longest
from os import makedirs
from os.path import basename, join
from typing import Iterable, List, Optional

import tensorflow as tf
from tqdm import tqdm


class Word2phrase:
    """
    Converts words that appear frequently together (i.e. phrases) into
    single words, e.g.  new york times  --> new_york_times
                        south africa    --> south_africa
                        larry page      --> larry_page

    Python port of Mikolov et al.'s original word2phrase.c code:
    https://github.com/tmikolov/word2vec/blob/master/word2phrase.c
    """

    def __init__(
        self,
        min_word_count: int,
        threshold: float,
        threshold_decay: float,
        phrase_sep: str,
    ) -> None:
        """
        Initializes the Word2phrase instance.

        Parameters
        ----------
        min_word_count : int
            Minimum number of times a word might occur for it to be in the vocabulary.
        threshold : float
            Threshold for determining whether a given phrase should be included.
        threshold_decay : float
            Value to use for decaying the threshold over time.
        phrase_sep : str
            Separator to use when combining phrases.
        """
        self._min_word_count = min_word_count
        self._threshold = threshold
        self._threshold_decay = threshold_decay
        self._phrase_sep = phrase_sep

        self._word_occurrences_counter: Optional[Counter] = None
        self._total_unigram_words = 0

    @staticmethod
    def _pairwise_grouping_iter(iterable: Iterable) -> zip_longest:
        """
        Groups elements of an iterable with pairwise tuples.

        Parameters
        ----------
        iterable : Iterable
            Iterable to apply pairwise grouping to.

        Returns
        -------
        pairwise_iterable : zip_longest
            Pairwise iterable as a zip_longest object
        """
        left, right = tee(iterable)
        try:
            next(right)
        except StopIteration:
            pass
        return zip_longest(left, right)

    def _build_word_occurrences(
        self, filepaths: List[str], num_texts: int, max_vocab_size: int
    ) -> None:
        """
        Builds the internal vocabulary using text data files.

        Parameters
        ----------
        filepaths : list of str
            Filepaths of text data files to build the vocabulary on.
        num_texts : int
            Number of texts (or sentences) of the content of `filepath`.
        max_vocab_size : int
            Maximum vocabulary size to use (-1 indicates all words in vocabulary).

            In other words, only the top `max_vocab_size` words will be taken into
            account when counting word occurrences.
        """
        # Read file content and split into words
        lines = []
        for filepath in filepaths:
            with tf.io.gfile.GFile(filepath) as f:
                lines.append(f)
        lines = chain(*lines)

        self._total_unigram_words = 0
        self._word_occurrences_counter = Counter()
        for line in tqdm(
            lines,
            desc="- Building word occurrences",
            total=num_texts,
        ):
            words = line.strip().split()
            pairwise_words = [
                f"{a}{self._phrase_sep}{b}" for a, b in zip(words, words[1:])
            ]

            # Count unigram word occurrences
            self._word_occurrences_counter.update(words)
            self._total_unigram_words += len(words)

            # Count bigram word occurrences
            self._word_occurrences_counter.update(pairwise_words)

        print(f"Initial vocabulary size: {len(self._word_occurrences_counter)}")

        # Only use most common words
        if max_vocab_size == -1:
            max_vocab_size = None
        word_occurrences = self._word_occurrences_counter.most_common(max_vocab_size)
        print(f"New vocabulary size after maximization: {len(word_occurrences)}")

        # Exclude words with less than `self._min_word_count` occurrences
        word_occurrences = [
            (word, word_count)
            for word, word_count in tqdm(
                word_occurrences, desc="- Filtering word occurrences"
            )
            if word_count >= self._min_word_count
        ]
        print(
            f"Final vocabulary size after filtering on minimum word count: {len(word_occurrences)}"
        )

    def fit(
        self,
        text_data_filepaths: List[str],
        dataset_name: str,
        n_epochs: int,
        num_texts: int,
        max_vocab_size: int,
        output_dir: str,
    ):
        """
        Trains/fits the word2phrase instance and saves new text data files
        where phrases have been replaced with single words.

        Parameters
        ----------
        text_data_filepaths : list of str
            Filepaths of text data files to train on.
        dataset_name : str
            Name of the dataset we are fitting/training on.
        n_epochs : int
            Number of passes through the text data files; more runs
            yields longer phrases.
        num_texts : int
            Number of texts (or sentences) of the content of `filepaths`.
        max_vocab_size : int, optional
            Maximum vocabulary size to use (-1 indicates all words in vocabulary).

            In other words, only the top `max_vocab_size` words will be taken into
            account when counting word occurrences.
        output_dir : str
            Output directory to save the new text data files.
        """
        for epoch in range(1, n_epochs + 1):
            print(f"Epoch {epoch}/{n_epochs}")

            # Compute threshold
            threshold = self._threshold * (1 - self._threshold_decay) ** (epoch - 1)

            # Builds vocabulary using text data files
            self._build_word_occurrences(
                filepaths=text_data_filepaths,
                num_texts=num_texts,
                max_vocab_size=max_vocab_size,
            )

            # Create output directory for current epoch
            current_output_dir = join(
                output_dir, f"{dataset_name}_phrases", f"epoch_{epoch}"
            )
            makedirs(current_output_dir, exist_ok=True)

            # Iterate over all texts/sentences for each text data file.
            progressbar = tqdm(
                total=num_texts, desc="- Computing scores for each text data file"
            )
            new_filepaths = []
            for filepath in text_data_filepaths:
                filename = basename(filepath)
                with open(filepath, "r") as input_file:
                    new_filepath = join(current_output_dir, filename)
                    new_filepaths.append(new_filepath)
                    with open(new_filepath, "w") as output_file:
                        for i, line in enumerate(input_file.readlines()):
                            new_line = []
                            words = line.strip().split()
                            pairwise_words = self._pairwise_grouping_iter(words)
                            for pair in pairwise_words:
                                left_word, right_word = pair
                                bigram_word = f"{left_word}{self._phrase_sep}{right_word}"
                                pa = self._word_occurrences_counter.get(left_word)
                                pb = self._word_occurrences_counter.get(right_word)
                                pab = self._word_occurrences_counter.get(bigram_word)
                                all_words_in_vocab = pa and pb and pab

                                # Compute score
                                if all_words_in_vocab:
                                    score = (
                                        (pab - self._min_word_count)
                                        / pa
                                        / pb
                                        * self._total_unigram_words
                                    )
                                else:
                                    score = 0.0

                                if score > threshold:
                                    try:
                                        # Skip next pair of words, since we combined current pair into
                                        # a single word.
                                        next(pairwise_words)
                                    except StopIteration:
                                        pass
                                    new_line.append(bigram_word)
                                else:
                                    new_line.append(left_word)

                            # Write line to output file
                            if i > 0:
                                output_file.write("\n")
                            output_file.write(" ".join(new_line))
                            progressbar.update(1)

            # Change text data filepaths to the newly saved text filepaths
            text_data_filepaths = new_filepaths.copy()
