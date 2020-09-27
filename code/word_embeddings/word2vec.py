import os
import pickle
from typing import List, Optional

import numpy as np
import tensorflow as tf
from data_utils import Tokenizer, create_dataset
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar
from train_utils import create_model_checkpoint_filepath
from word2vec_model import Word2VecSGNSModel


class Word2vec:
    """
    Helper class for training a Word2vec model.
    """

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        embedding_dim: int = 300,
        learning_rate: float = 0.0025,
        min_learning_rate: float = 0.0001,
        batch_size: int = 256,
        sampling_window_size: int = 2,
        num_negative_samples: int = 15,
        unigram_exponent_negative_sampling: float = 3 / 4,
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
            Training learning rate (defaults to 0.0025).
        min_learning_rate : float, optional
            Minimum training learning rate (defaults to 0.0001).
        batch_size : int
            Size of batches during fitting/training.
        sampling_window_size : int
            Window size to use when generating skip-gram couples (defaults to 2).
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
        self._min_learning_rate = min_learning_rate
        self._batch_size = batch_size
        self._sampling_window_size = sampling_window_size
        self._num_negative_samples = num_negative_samples
        self._unigram_exponent_negative_sampling = unigram_exponent_negative_sampling
        self._model_name = model_name
        self._target_embedding_layer_name = target_embedding_layer_name
        self._model_checkpoints_dir = model_checkpoints_dir

        # Initialize model
        self._init_model()

        # Set train step signature
        inputs_spec = tf.TensorSpec(shape=(self._batch_size,), dtype="int64")
        labels_spec = tf.TensorSpec(shape=(self._batch_size,), dtype="int64")
        progress_spec = tf.TensorSpec(shape=(1,), dtype="float32")
        self._train_step_signature = [inputs_spec, labels_spec, progress_spec]

    def _init_model(self, weights: List[np.ndarray] = None) -> None:
        """
        Initializes the Word2vec model.

        Parameters
        ----------
        weights : list of np.ndarray
            List of Numpy arrays containing weights to initialize model with (defaults to None).
        """
        if self._tokenizer is None:
            self._model: Optional[Model] = None
        else:
            self._model = Word2VecSGNSModel(
                word_counts=self._tokenizer.word_counts,
                embedding_dim=self._embedding_dim,
                batch_size=self._batch_size,
                num_negative_samples=self._num_negative_samples,
                unigram_exponent_negative_sampling=self._unigram_exponent_negative_sampling,
                learning_rate=self._learning_rate,
                min_learning_rate=self._min_learning_rate,
                add_bias=True,
                name=self._model_name,
                target_embedding_layer_name=self._target_embedding_layer_name,
            )

            if weights is not None:
                self._model.set_weights(weights)

    def get_model(self) -> Optional[Model]:
        """
        Gets the internal Word2vec Keras model.

        Returns
        -------
        model : Model
            Word2vec Keras model
        """
        return self._model

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

        # Get target embedding weights
        target_embedding_weights = [
            weight
            for weight in self._model.weights
            if weight.name.startswith(self._target_embedding_layer_name)
        ][0].numpy()

        return target_embedding_weights

    def fit(
        self,
        text_data_filepath: str,
        num_texts: int,
        dataset_name: str,
        n_epochs: int,
        starting_epoch_nr: int = 1,
        verbose: int = 1,
    ) -> None:
        """
        Fits/trains the Word2vec model.

        Parameters
        ----------
        text_data_filepath : str
            Path of text data to fit/train the Word2vec model on.
        num_texts : int
            Number of texts (or sentences) of the content of `text_data_filepath`.
        dataset_name : str
            Name of the dataset we are fitting/training on.
        n_epochs : int
            Number of epochs to fit/train.
        starting_epoch_nr : int, optional
            Denotes the starting epoch number (defaults to 1).
        verbose : int, optional
            Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose).
            Defaults to 1 (verbose).
        """

        # Ensure checkpoints directory exists before training
        os.makedirs(self._model_checkpoints_dir, exist_ok=True)

        # Set up optimizer (SGD) with maximal learning rate.
        # The idea here is that `perform_train_step` will apply a decaying learning rate.
        optimizer = tf.keras.optimizers.SGD(1.0)

        @tf.function(input_signature=self._train_step_signature)
        def perform_train_step(
            input_targets: tf.Tensor,
            input_contexts: tf.Tensor,
            progress: float,
        ):
            """
            Performs a single training step on a batch of target/context pairs.

            Parameters
            ----------
            input_targets : tf.Tensor
                Input targets to train on
            input_contexts : tf.Tensor
                Input contexts to train on
            progress : float
                Current training progress
            Returns
            -------
            payload : tuple
                Tuple consisting of computed loss and learning rate
            """
            skip_gram_loss = self._model(input_targets, input_contexts)
            gradients = tf.gradients(skip_gram_loss, self._model.trainable_variables)

            decaying_learning_rate = tf.maximum(
                self._learning_rate * (1 - progress) + self._min_learning_rate * progress,
                self._min_learning_rate,
            )

            # Apply learning rate
            if hasattr(gradients[0], "_values"):
                gradients[0]._values *= decaying_learning_rate
            else:
                gradients[0] *= decaying_learning_rate

            if hasattr(gradients[1], "_values"):
                gradients[1]._values *= decaying_learning_rate
            else:
                gradients[1] *= decaying_learning_rate

            if hasattr(gradients[2], "_values"):
                gradients[2]._values *= decaying_learning_rate
            else:
                gradients[2] *= decaying_learning_rate

            optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

            return skip_gram_loss, decaying_learning_rate

        # Train model
        if verbose == 1:
            print("---")
            print(
                f"Fitting Word2vec on {dataset_name} with arguments:\n"
                f"- batch_size={self._batch_size}\n"
                f"- n_epochs={n_epochs}\n"
                f"- vocab_size={self._tokenizer.vocab_size}\n"
                f"- embedding_dim={self._embedding_dim}\n"
                f"- learning_rate={self._learning_rate}\n"
                f"- min_learning_rate={self._min_learning_rate}\n"
                f"- window_size={self._sampling_window_size}\n"
                f"- num_negative_samples={self._num_negative_samples}"
            )
            print("---")
        end_epoch_nr = n_epochs + starting_epoch_nr - 1
        for epoch_nr in range(starting_epoch_nr, end_epoch_nr + 1):
            if verbose >= 1:
                print(f"Epoch {epoch_nr}/{end_epoch_nr}")

            # Initialize progressbar
            progressbar = Progbar(
                num_texts,
                verbose=verbose,
                stateful_metrics=["learning_rate"],
            )
            progressbar.update(0)

            # Initialize new dataset per epoch
            train_dataset = create_dataset(
                text_data_filepath,
                num_texts,
                self._tokenizer,
                self._sampling_window_size,
                self._batch_size,
            )

            # Iterate over batches of data and perform training
            avg_loss = 0.0
            steps = 0
            for (
                input_targets_batch,
                input_contexts_batch,
                epoch_progress,
            ) in train_dataset:

                # Compute overall progress (over all epochs)
                overall_progress = tf.constant(
                    (epoch_nr - 1 + epoch_progress) / end_epoch_nr,
                    shape=(1,),
                    dtype=tf.float32,
                )

                # Train on batch
                loss, learning_rate = perform_train_step(
                    input_targets_batch, input_contexts_batch, overall_progress
                )

                # Add to average loss
                loss_np = loss.numpy().mean()
                avg_loss += loss_np
                steps += 1

                # Update progressbar
                sent_nr = int(epoch_progress.numpy() * num_texts)
                progressbar.update(
                    sent_nr,
                    values=[
                        ("loss", loss_np),
                        ("learning_rate", learning_rate),
                    ],
                )

                # TODO: Save every N'th% weights to file.
            print()

            # Compute average loss
            avg_loss /= steps

            # Save intermediate model to file
            if verbose == 1:
                print("Saving model to file...")
            checkpoint_path = create_model_checkpoint_filepath(
                self._model_checkpoints_dir,
                self._model_name,
                dataset_name,
                epoch_nr,
                avg_loss,
            )
            self.save_model(checkpoint_path)
            if verbose == 1:
                print("Done!")

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

        # Initialize model with weights from file
        self._init_model(saved_model_dict["_model_weights"])
