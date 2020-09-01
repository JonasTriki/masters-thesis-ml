from typing import Callable, Optional

import numpy as np
from tensorflow.keras.preprocessing.sequence import make_sampling_table, skipgrams
from tensorflow.keras.utils import Sequence, to_categorical
from tqdm import tqdm

from utils import (
    calc_skipgram_pairs_number,
    load_text_data_tokenized,
    skipgram_pairs_generator,
    subsample_words_by_freq,
)


class SGNSDataGenerator(Sequence):
    """
    Skipgram negative sampling data generator for training
    a Word2vec model
    """

    def __init__(
        self,
        text_filepath: str,
        vocab_filepath: str,
        preprocess_text: Callable[[str], list],
        batch_size: int,
        sampling_window_size: int,
        num_negative_samples: int,
        vocab_size: Optional[int] = None,
        sampling_factor: float = 1e-5,
        one_hot_skipgram_pairs: bool = False,
        shuffle: bool = True,
        input_target_layer_name: str = "input_target",
        input_context_layer_name: str = "input_context",
    ) -> None:
        """
        Initializes a skipgram negative sampling generator
        """
        print("Loading data...")
        (
            self.tokenized_text,
            self.word_dict,
            self.word_counts_dict,
            self.word_noise_dict,
        ) = load_text_data_tokenized(text_filepath, vocab_filepath, preprocess_text)
        print("Done!")

        # Subsample data using sampling factor
        print("Randomly subsampling data...")
        self.tokenized_text = subsample_words_by_freq(
            self.tokenized_text, sampling_factor=sampling_factor
        )
        print("Done!")

        self.batch_size = batch_size
        self.sampling_window_size = sampling_window_size
        self.num_negative_samples = num_negative_samples

        print("Counting number of skipgram pairs...")
        self.num_skipgram_pairs = calc_skipgram_pairs_number(
            self.tokenized_text, self.sampling_window_size, self.num_negative_samples
        )
        print(f"Done! Will generate {self.num_skipgram_pairs} skipgram pairs per epoch.")

        if vocab_size is None:
            self.vocab_size = len(self.word_dict)
        else:
            # Ensure that the vocabulary size never exceeds
            # the word dictionary
            if vocab_size > len(self.word_dict):
                self.vocab_size = len(self.word_dict)
            else:
                # Add 1 to account for the unknown word
                self.vocab_size = vocab_size + 1

        self.one_hot_skipgram_pairs = one_hot_skipgram_pairs

        self.shuffle = shuffle

        self.input_target_layer_name = input_target_layer_name
        self.input_context_layer_name = input_context_layer_name

        self.on_epoch_end()

    def on_epoch_end(self) -> None:
        """
        Generates new pair of skipgram couples and
        updates indices after every epoch
        """
        # Initialize skipgram pairs generator
        self.skipgram_pairs_gen = skipgram_pairs_generator(
            self.tokenized_text,
            self.word_noise_dict,
            self.sampling_window_size,
            self.num_negative_samples,
        )
        self.add_to_next_batch: list = []

        lala: list = []
        for pair in tqdm(self.skipgram_pairs_gen, total=self.num_skipgram_pairs):
            lala += pair
        print(len(lala))
        # Create skipgram pairs
        # print("Preparing skipgram pairs...")
        # skipgram_pairs, skipgram_labels = skipgrams(
        #     self.tokenized_text,
        #     self.vocab_size,
        #     sampling_table=self.sampling_table,
        #     window_size=self.sampling_window_size,
        #     negative_samples=self.num_negative_samples,
        # )
        # self.skipgram_pairs = np.array(skipgram_pairs)
        # self.skipgram_labels = np.array(skipgram_labels)
        # self.indices = np.arange(len(self.skipgram_pairs))
        # print(f"Generated {len(skipgram_pairs)} skipgram couples => {len(self)} batches")

        # if self.shuffle:
        #     np.random.shuffle(self.indices)

    def __len__(self) -> int:
        """
        Denotes the number of batches per epoch
        """
        return int(np.ceil(self.num_skipgram_pairs / float(self.batch_size)))

    def __getitem__(self, index: int) -> object:
        """
        Generates one batch of skipgram negative sampling data
        """
        # Generate batch of data
        pairs_batch = []
        if len(self.add_to_next_batch) > 0:
            pairs_batch += self.add_to_next_batch
            self.add_to_next_batch = []
        while len(pairs_batch) < self.batch_size:
            pairs_batch += next(self.skipgram_pairs_gen)

        if len(pairs_batch) > self.batch_size:
            self.add_to_next_batch += pairs_batch[self.batch_size :]
            pairs_batch = pairs_batch[: self.batch_size]

        # Convert to numpy matrix to easier access the columns
        pairs_batch_mat = np.array(pairs_batch)

        # Get batched skipgram pairs/labels
        # batch = [target, context, label]
        X_batch_target = pairs_batch_mat[:, 0]
        X_batch_context = pairs_batch_mat[:, 1]
        y_batch = pairs_batch_mat[:, 2]

        # One-hot encode target/context pairs
        if self.one_hot_skipgram_pairs:
            X_batch_target = to_categorical(X_batch_target, self.vocab_size)
            X_batch_context = to_categorical(X_batch_context, self.vocab_size)

        # Create batch using object notation
        batch_obj = (
            {
                self.input_target_layer_name: X_batch_target,
                self.input_context_layer_name: X_batch_context,
            },
            y_batch,
        )

        return batch_obj
