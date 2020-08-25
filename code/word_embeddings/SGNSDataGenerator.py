from typing import Optional

import numpy as np
from tensorflow.keras.preprocessing.sequence import make_sampling_table, skipgrams
from tensorflow.keras.utils import Sequence, to_categorical

from utils import load_text_data_tokenized


class SGNSDataGenerator(Sequence):
    """
    Skipgram negative sampling data generator for training
    a Word2vec model
    """

    def __init__(
        self,
        text_filepath: str,
        vocab_filepath: str,
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
        self.tokenized_text, self.word_dict = load_text_data_tokenized(
            text_filepath, vocab_filepath
        )
        self.batch_size = batch_size

        self.sampling_window_size = sampling_window_size
        self.num_negative_samples = num_negative_samples
        if vocab_size is None:
            self.vocab_size = len(self.word_dict)
        else:
            # Ensure that the vocabulary size never exceeds
            # the word dictionary
            if vocab_size > len(self.word_dict):
                self.vocab_size = len(self.word_dict)
            else:
                self.vocab_size = vocab_size
        self.sampling_table = make_sampling_table(
            self.vocab_size + 1, sampling_factor=sampling_factor
        )
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
        # Create skipgram pairs
        skipgram_pairs, skipgram_labels = skipgrams(
            self.tokenized_text,
            self.vocab_size + 1,
            sampling_table=self.sampling_table,
            window_size=self.sampling_window_size,
            negative_samples=self.num_negative_samples,
        )
        self.skipgram_pairs = np.array(skipgram_pairs)
        self.skipgram_labels = np.array(skipgram_labels)
        self.indices = np.arange(len(self.skipgram_pairs))
        print(f"Generated {len(skipgram_pairs)} skipgram couples => {len(self)} batches")

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        """
        Denotes the number of batches per epoch
        """
        return int(np.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, index: int) -> object:
        """
        Generates one batch of skipgram negative sampling data
        """
        # Get indices of batch
        batch_indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        # Get batched skipgram couples
        # (X = target/context pairs, y = labels)
        X_batch = self.skipgram_pairs[batch_indices]
        X_batch_target = X_batch[:, 0]
        X_batch_context = X_batch[:, 1]
        y_batch = self.skipgram_labels[batch_indices]

        # One-hot encode target/context pairs
        if self.one_hot_skipgram_pairs:
            X_batch_target = to_categorical(X_batch_target, self.vocab_size + 1)
            X_batch_context = to_categorical(X_batch_context, self.vocab_size + 1)

        # Create batch using object notation
        batch_obj = (
            {
                self.input_target_layer_name: X_batch_target,
                self.input_context_layer_name: X_batch_context,
            },
            y_batch,
        )

        return batch_obj
