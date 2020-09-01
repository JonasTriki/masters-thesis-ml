import os

import numpy as np
from tensorflow.keras.utils import Progbar

from data_utils import Tokenizer, create_dataset
from models import build_word2vec_model
from train_utils import get_model_checkpoint_filepath


class Word2vec:
    """
    TODO: Docs
    """

    def __init__(
        self,
        name: str,
        tokenizer: Tokenizer,
        embedding_dim: int = 300,
        learning_rate: float = 0.001,
        sampling_window_size: int = 4,
        num_negative_samples: int = 15,
        target_embedding_layer_name: str = "target_embedding",
        model_checkpoints_dir: str = "checkpoints",
    ) -> None:
        """
        TODO: Docs
        """
        self._name = name
        self._tokenizer = tokenizer
        self._embedding_dim = embedding_dim
        self._learning_rate = learning_rate
        self._sampling_window_size = sampling_window_size
        self._num_negative_samples = num_negative_samples
        self._target_embedding_layer_name = target_embedding_layer_name
        self._model_checkpoints_dir = model_checkpoints_dir

        # Build Word2vec model
        self._model = build_word2vec_model(
            self._tokenizer.vocab_size,
            self._embedding_dim,
            self._learning_rate,
            self._target_embedding_layer_name,
        )

    def fit(
        self,
        texts: list,
        dataset_name: str,
        n_epochs: int,
        batch_size: int,
        starting_epoch: int = 0,
        verbose: int = 1,
    ) -> None:
        """
        TODO: Docs
        """

        # Ensure checkpoints directory exists before training
        os.makedirs(self._model_checkpoints_dir, exist_ok=True)

        # Train model
        if verbose:
            print("---")
            print(
                f"Fitting Word2vec on {dataset_name} with arguments:\n"
                f"- vocab_size={self._tokenizer.vocab_size}\n"
                f"- embedding_dim={self._embedding_dim}\n"
                f"- learning_rate={self._learning_rate}\n"
                f"- window_size={self._sampling_window_size}\n"
                f"- num_negative_samples={self._num_negative_samples}\n"
                f"for {n_epochs} epochs in batches of {batch_size}..."
            )
            print("---")
        num_sents = len(texts)
        for epoch_nr in range(1, n_epochs + 1):
            print(f"Epoch {epoch_nr}/{n_epochs}")

            # Initialize progressbar
            progbar = Progbar(num_sents, verbose=verbose)
            progbar.update(0)

            # Initialize new dataset per epoch
            train_dataset = create_dataset(
                texts,
                self._tokenizer,
                self._sampling_window_size,
                self._num_negative_samples,
                batch_size,
            )

            avg_loss = 0.0
            steps = 0
            for skipgram_pairs_batch, sent_idx_tf in train_dataset:

                # Extract from Tensors
                sent_nr = sent_idx_tf[-1].numpy() + 1

                # Prepare batch for training
                skipgram_pairs_batch_target = skipgram_pairs_batch[:, 0]
                skipgram_pairs_batch_context = skipgram_pairs_batch[:, 1]
                skipgram_pairs_batch_input = [
                    skipgram_pairs_batch_target,
                    skipgram_pairs_batch_context,
                ]
                skipgram_pairs_batch_labels = skipgram_pairs_batch[:, 2]

                # Train on batch
                loss = self._model.train_on_batch(
                    x=skipgram_pairs_batch_input, y=skipgram_pairs_batch_labels
                )
                avg_loss += loss
                steps += 1

                # Update progressbar
                progbar.update(sent_nr, values=[("loss", loss)])
            print()

            # Compute average loss
            avg_loss /= steps

            # Save intermediate model to file
            print("Saving model to file...")
            checkpoint_path = get_model_checkpoint_filepath(
                self._model_checkpoints_dir, self._name, dataset_name, epoch_nr, avg_loss,
            )
            self._model.save(checkpoint_path)
            print("Done!")

            # TODO: Generate UMAP embeddings?

    @property
    def embedding_weights(self) -> np.ndarray:
        """
        TODO: Docs
        """
        # Get target embedding layer
        target_embedding_layer = self._model.get_layer(
            name=self._target_embedding_layer_name
        )

        # Get weights
        weights = target_embedding_layer.get_weights()[0]

        return weights
