import os
import pickle
from typing import Optional

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar

from .data_utils import Tokenizer, create_dataset
from .models import build_word2vec_model
from .train_utils import get_model_checkpoint_filepath


class Word2vec:
    """
    Helper class for training a Word2vec model.
    """

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        embedding_dim: int = 300,
        learning_rate: float = 0.001,
        sampling_window_size: int = 4,
        num_negative_samples: int = 15,
        model_name: str = "word2vec_sgns",
        target_embedding_layer_name: str = "target_embedding",
        model_checkpoints_dir: str = "checkpoints",
    ) -> None:
        """
        Initializes a Word2vec instance.

        Parameters
        ----------
        tokenizer : Tokenizer, optional
            Tokenizer instance (defaults to None).
        embedding_dim : int, optional
            Word2vec embedding dimensions (defaults to 300).
        learning_rate : float, optional
            Training learning rate (defaults to 0.0001).
        sampling_window_size : int
            Window size to use when generating skip-gram couples (defaults to 4).
        num_negative_samples : int
            Number of negative samples to use when generating skip-gram couples
            (defaults to 15).
        model_name : str, optional
            Name of the Word2vec model (defaults to "word2vec_sgns").
        target_embedding_layer_name : str, optional
            Name to use for the target embedding layer (defaults to "target_embedding").
        model_checkpoints_dir : str, optional
            Where to save checkpoints of the model after each epoch
            (defaults to "checkpoints").
        """
        self._tokenizer = tokenizer
        self._embedding_dim = embedding_dim
        self._learning_rate = learning_rate
        self._sampling_window_size = sampling_window_size
        self._num_negative_samples = num_negative_samples
        self._model_name = model_name
        self._target_embedding_layer_name = target_embedding_layer_name
        self._model_checkpoints_dir = model_checkpoints_dir

        # Build Word2vec model
        if self._tokenizer is not None:
            self.build_model()
        else:
            self._model: Optional[Model] = None

    @property
    def tokenizer(self) -> Tokenizer:
        """
        Gets the current tokenizer instance

        Returns
        -------
        tokenizer : Tokenizer
            Current tokenizer.
        """
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Tokenizer) -> None:
        """
        Sets the current tokenizer instance

        Parameters
        ----------
        tokenizer : Tokenizer
            The new tokenizer instance to use.
        """
        self._tokenizer = tokenizer

    def build_model(self) -> None:
        """
        Builds a Word2vec Keras model.
        """
        if self._tokenizer is None:
            raise TypeError("Can not build Word2vec model: tokenizer has not been set.")
        self._model = build_word2vec_model(
            self._tokenizer.vocab_size,
            self._embedding_dim,
            self._learning_rate,
            self._target_embedding_layer_name,
            self._model_name,
        )

    @property
    def embedding_weights(self) -> np.ndarray:
        """
        Gets the embedding weights of the target embedding layer of the internal Keras model.

        Returns
        -------
        embedding_weights : np.ndarray
            Embedding weights of the target embedding layer.
        """
        if self._model is None:
            raise TypeError(
                "Model has not been built yet. Did you forget to call `build_model`?"
            )
        # Get target embedding layer
        target_embedding_layer = self._model.get_layer(
            name=self._target_embedding_layer_name
        )

        # Get weights
        embedding_weights = target_embedding_layer.get_weights()[0]

        return embedding_weights

    def fit(
        self,
        texts: list,
        dataset_name: str,
        n_epochs: int,
        batch_size: int,
        starting_epoch_nr: int = 1,
        verbose: int = 1,
    ) -> None:
        """
        Fits/trains the Word2vec model.

        Parameters
        ----------
        texts : list
            List of texts to fit/train the Word2vec model on.
        dataset_name : str
            Name of the dataset we are fitting/training on.
        n_epochs : int
            Number of epochs to fit/train.
        batch_size : int
            Size of batches during fitting/training.
        starting_epoch_nr : int, optional
            Denotes the starting epoch number (defaults to 1).
        verbose : int, optional
            Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose).
            Defaults to 1 (verbose).
        """
        if self._model is None:
            raise TypeError(
                "Model has not been built yet. Did you forget to call `build_model`?"
            )

        # Ensure checkpoints directory exists before training
        os.makedirs(self._model_checkpoints_dir, exist_ok=True)

        # Train model
        if verbose == 1:
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
        for epoch_nr in range(starting_epoch_nr, n_epochs + starting_epoch_nr):
            if verbose >= 1:
                print(f"Epoch {epoch_nr}/{n_epochs + starting_epoch_nr - 1}")

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
            for skip_gram_pairs_batch, sent_idx_tf in train_dataset:

                # Extract from Tensors
                sent_nr = sent_idx_tf[-1].numpy() + 1

                # Prepare batch for training
                skip_gram_pairs_batch_target = skip_gram_pairs_batch[:, 0]
                skip_gram_pairs_batch_context = skip_gram_pairs_batch[:, 1]
                skip_gram_pairs_batch_input = [
                    skip_gram_pairs_batch_target,
                    skip_gram_pairs_batch_context,
                ]
                skip_gram_pairs_batch_labels = skip_gram_pairs_batch[:, 2]

                # Train on batch
                loss = self._model.train_on_batch(
                    x=skip_gram_pairs_batch_input, y=skip_gram_pairs_batch_labels
                )
                avg_loss += loss
                steps += 1

                # Update progressbar
                progbar.update(sent_nr, values=[("loss", loss)])
            print()

            # Compute average loss
            avg_loss /= steps

            # Save intermediate model to file
            if verbose == 1:
                print("Saving model to file...")
            checkpoint_path = get_model_checkpoint_filepath(
                self._model_checkpoints_dir,
                self._model_name,
                dataset_name,
                epoch_nr,
                avg_loss,
            )
            self.save_model(checkpoint_path)
            if verbose == 1:
                print("Done!")

            # TODO: Generate UMAP embeddings?

    def save_model(self, target_filepath: str) -> None:
        """
        Saves the Word2vec instance to a file.

        Parameters
        ----------
        target_filepath : str
            Where to save the model.
        """
        if self._model is None:
            raise TypeError(
                "Model has not been built yet. Did you forget to call `build_model`?"
            )

        # Make a copy of this class' internal dictionary
        # and remove the `_model` key-value pair
        self_dict = self.__dict__.copy()
        self_dict.pop("_model")

        # Prepare dictionary for saving
        saved_model_dict = {
            "__dict__": self_dict,
            "_model_weights": self._model.get_weights(),
        }

        # Save model to file
        with open(target_filepath, "wb") as file:
            pickle.dump(saved_model_dict, file)

    def load_model(self, model_filepath: str) -> None:
        """
        Loads the Word2vec instance from file.

        Parameters
        ----------
        model_filepath : str
            Where to load the model from.
        """
        # Read saved model dictionary from file
        with open(model_filepath, "rb") as file:
            saved_model_dict = pickle.load(file)

        # Set internal variables
        self.__dict__.update(saved_model_dict["__dict__"])

        # Build model and load weights from file
        self.build_model()
        if self._model is not None:
            self._model.set_weights(saved_model_dict["_model_weights"])
