import os
import pickle
from configparser import ConfigParser
from time import time
from typing import List, Optional, TextIO

import numpy as np
import tensorflow as tf
from dataset import create_dataset
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar
from tokenizer import Tokenizer
from train_utils import (create_model_checkpoint_filepath,
                         create_model_intermediate_embedding_weights_filepath,
                         create_model_train_logs_filepath)
from word2vec_model import Word2VecSGNSModel


class Word2vec:
    """
    Helper class for training a word2vec model.
    """

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        embedding_dim: int = 300,
        learning_rate: float = 0.0025,
        min_learning_rate: float = 0.0000025,
        batch_size: int = 256,
        max_window_size: int = 2,
        num_negative_samples: int = 15,
        unigram_exponent_negative_sampling: float = 3 / 4,
        model_name: str = "word2vec_sgns",
        target_embedding_layer_name: str = "target_embedding",
        model_checkpoints_dir: str = "checkpoints",
    ) -> None:
        """
        Initializes a word2vec instance.

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
        max_window_size : int
            Maximum window size to use when generating skip-gram couples (defaults to 2).
        num_negative_samples : int
            Number of negative samples to use when generating skip-gram couples
            (defaults to 15).
        model_name : str, optional
            Name of the word2vec model (defaults to "word2vec_sgns").
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
        self._max_window_size = max_window_size
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

    def __getstate__(self):
        """
        Gets the internal state of the class.
        """
        state = self.__dict__.copy()

        # Remove unpickable static vocabulary table
        del state["_model"]

        # Extract model weights, if they exist, and create
        # new modified state
        model_weights = self._model.get_weights() if self._model is not None else None
        modified_state = {"state": state, "model_weights": model_weights}

        return modified_state

    def __setstate__(self, modified_state):
        """
        Sets the internal state of the class.
        """
        self.__dict__.update(modified_state["state"])

        # Initialize model with weights from file
        self._init_model(modified_state["model_weights"])

    def _init_model(self, weights: List[np.ndarray] = None) -> None:
        """
        Initializes the word2vec model.

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
        Gets the internal word2vec Keras model.

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
        text_data_filepaths: List[str],
        num_texts: int,
        dataset_name: str,
        n_epochs: int,
        starting_epoch_nr: int = 1,
        intermediate_embedding_weights_saves: int = 0,
        train_logs_to_file: bool = True,
        verbose: int = 1,
    ) -> None:
        """
        Fits/trains the word2vec model.

        Parameters
        ----------
        text_data_filepaths : list
            Paths of text data to generate skip-gram target/context pairs from.
        num_texts : int
            Number of texts (or sentences) of the contents of `text_data_filepaths`.
        dataset_name : str
            Name of the dataset we are fitting/training on.
        n_epochs : int
            Number of epochs to fit/train.
        starting_epoch_nr : int, optional
            Denotes the starting epoch number (defaults to 1).
        intermediate_embedding_weights_saves : int, optional
            Number of intermediate saves of embedding weights per epoch during training
            (defaults to 0).
        train_logs_to_file : bool, optional
            Whether or not to save logs from training to file.
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

        intermediate_saving_thresholds: Optional[float] = None
        if intermediate_embedding_weights_saves > 0:

            # Save words to file for later reference
            if verbose == 1:
                print("Saving words to file...")
            words_filepath = os.path.join(
                self._model_checkpoints_dir,
                f"{self._model_name}_{dataset_name}_words.txt",
            )
            self.save_words(words_filepath)
            print("Done!")

            # Set up thresholds for saving intermediate embedding weights
            intermediate_saving_thresholds = 1 / intermediate_embedding_weights_saves

        # Save model training configuration to file
        model_training_conf_filepath = os.path.join(
            self._model_checkpoints_dir,
            f"{self._model_name}_{dataset_name}.conf",
        )
        self.save_model_training_conf(model_training_conf_filepath, n_epochs)

        # Train model
        if verbose == 1:
            print("---")
            print(
                f"Fitting word2vec on {dataset_name} with arguments:\n"
                f"- batch_size={self._batch_size}\n"
                f"- n_epochs={n_epochs}\n"
                f"- vocab_size={self._tokenizer.vocab_size}\n"
                f"- embedding_dim={self._embedding_dim}\n"
                f"- learning_rate={self._learning_rate}\n"
                f"- min_learning_rate={self._min_learning_rate}\n"
                f"- max_window_size={self._max_window_size}\n"
                f"- num_negative_samples={self._num_negative_samples}"
            )
            print("---")
        end_epoch_nr = n_epochs + starting_epoch_nr - 1

        # Initialize train logs file
        train_logs_file: Optional[TextIO] = None
        if train_logs_to_file:
            train_logs_filepath = create_model_train_logs_filepath(
                self._model_checkpoints_dir,
                self._model_name,
                dataset_name,
            )
            train_logs_file = open(train_logs_filepath, "w")
            train_logs_file.write("epoch_nr,train_loss,time_spent")
            train_logs_file.flush()

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
                text_data_filepaths,
                num_texts,
                self._tokenizer,
                self._max_window_size,
                self._batch_size,
            )

            # Measure time spent per epoch
            time_epoch_start = time()

            # Iterate over batches of data and perform training
            avg_loss = 0.0
            steps = 0
            intermediate_embedding_progress = 0
            for (
                input_targets_batch,
                input_contexts_batch,
                epoch_progress,
            ) in train_dataset:

                # Perform intermediate saves of embedding weights to file
                if intermediate_embedding_weights_saves > 0:
                    if (
                        epoch_progress - intermediate_embedding_progress
                        >= intermediate_saving_thresholds
                    ):
                        intermediate_embedding_progress += intermediate_saving_thresholds

                        # Save to file
                        intermediate_embedding_weight_nr = int(
                            intermediate_embedding_progress
                            * intermediate_embedding_weights_saves
                        )
                        self.save_embedding_weights(
                            create_model_intermediate_embedding_weights_filepath(
                                self._model_checkpoints_dir,
                                self._model_name,
                                dataset_name,
                                epoch_nr,
                                intermediate_embedding_weight_nr,
                            )
                        )

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
            print()

            # Compute time spent on epoch
            time_spent_epoch = time() - time_epoch_start
            if verbose == 1:
                print(f"Spent {time_spent_epoch:.2f} seconds!")

            # Compute average loss
            avg_loss /= steps

            # Save last intermediate save of embedding weights to file
            if intermediate_embedding_weights_saves > 0:
                self.save_embedding_weights(
                    create_model_intermediate_embedding_weights_filepath(
                        self._model_checkpoints_dir,
                        self._model_name,
                        dataset_name,
                        epoch_nr,
                        intermediate_embedding_weights_saves,
                    )
                )

            # Write to train logs
            if train_logs_to_file:
                train_logs_file.write(f"\n{epoch_nr},{avg_loss},{time_spent_epoch}")
                train_logs_file.flush()

            # Save intermediate model to file
            if verbose == 1:
                print("Saving model to file...")
            checkpoint_path = create_model_checkpoint_filepath(
                self._model_checkpoints_dir,
                self._model_name,
                dataset_name,
                epoch_nr,
            )
            self.save_model(checkpoint_path)
            if verbose == 1:
                print("Done!")

        # Close train logs file handler
        if train_logs_to_file:
            train_logs_file.close()

    def save_model(self, target_filepath: str) -> None:
        """
        Saves the word2vec instance to a file.

        Parameters
        ----------
        target_filepath : str
            Where to save the model.
        """
        # Save model to file
        with open(target_filepath, "wb") as file:
            pickle.dump(self, file)

    def save_embedding_weights(self, target_filepath: str) -> None:
        """
        Saves (target) embedding weights to file using Numpy.

        Parameters
        ----------
        target_filepath : str
            Where to save the (target) embedding weights to.
        """
        np.save(target_filepath, self.embedding_weights)

    def save_words(self, target_filepath: str) -> None:
        """
        Saves words used during training to file, one word in each line.

        Parameters
        ----------
        target_filepath : str
            Where to save the words to.
        """
        # Create string with one word in each line
        words = list(self._tokenizer.words)
        words_lines = "\n".join(words)

        # Save to file
        with open(target_filepath, "w") as file:
            file.write(words_lines)

    def save_model_training_conf(self, target_filepath: str, n_epochs: int) -> None:
        """
        Saves model training configuration to file.

        Parameters
        ----------
        target_filepath : str
            Target filepath for saving configuration.
        n_epochs : int
            Number of epochs used during training.
        """
        # Create config parser and add key-value pairs
        model_train_config = ConfigParser()
        model_train_config["MODELCONFIG"] = {
            "vocab_size": str(self._tokenizer.vocab_size),
            "embedding_dim": str(self._embedding_dim),
        }
        model_train_config["TRAINCONFIG"] = {
            "batch_size": str(self._batch_size),
            "n_epochs": str(n_epochs),
            "learning_rate": str(self._learning_rate),
            "min_learning_rate": str(self._min_learning_rate),
            "max_window_size": str(self._max_window_size),
            "num_negative_samples": str(self._num_negative_samples),
        }

        # Save to file
        with open(target_filepath, "w") as file:
            model_train_config.write(file)
