import argparse
import os
from os.path import join as join_path

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from models import build_word2vec_model
from SGNSDataGenerator import SGNSDataGenerator


def parse_args() -> argparse.Namespace:
    """
    Parses arguments from the commandline
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_filepath",
        type=str,
        default="data/dracula.txt",  # TODO: Remove default
        help="Text filepath containing the text we wish to train on",
    )
    parser.add_argument(
        "--vocab_filepath",
        type=str,
        default="data/dracula_vocab.pickle",  # TODO: Remove default
        help="Vocabulary filepath containing the word vocabulary we want to use",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size used for training",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=20, help="Number of epochs to train our model on",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate to use when training",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="Vocabulary size to use when training. Defaults to use all words",
    )
    parser.add_argument(
        "--vector_dim",
        type=int,
        default=300,
        help="Number of latent dimensions to use in the embedding layers",
    )
    parser.add_argument(
        "--sampling_window_size",
        type=int,
        default=5,
        help="Window size to use when generating skipgram couples",
    )
    parser.add_argument(
        "--num_negative_samples",
        type=int,
        default=15,
        help="Number of negative samples to use when generating skipgram couples",
    )
    parser.add_argument(
        "--sampling_factor",
        type=float,
        default=1e-5,
        help="Sampling factor to use when generating skipgram couples",
    )
    parser.add_argument(
        "--model_checkpoint_dir",
        type=str,
        default="checkpoints/",
        help="Where to save the intermediate model after each epoch",
    )
    parser.add_argument(
        "--model_checkpoint_filename",
        type=str,
        default="model-sgns.{epoch:02d}-{loss}.h5",
        help="""Filename to use when saving intermediate models after each epoch.

        For more information, please refer to the documentation:
        https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
        """,
    )
    parser.add_argument(
        "--pretrained_model_filepath",
        type=str,
        default="",
        help="Load an already trained Word2vec model from file",
    )
    parser.add_argument(
        "--pretrained_model_epoch_nr",
        type=int,
        default=None,
        help="Epoch number of an already trained Word2vec model",
    )
    return parser.parse_args()


def train_word2vec_sgns(
    text_filepath: str,
    vocab_filepath: str,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    vocab_size: int,
    vector_dim: int,
    sampling_window_size: int,
    num_negative_samples: int,
    sampling_factor: float,
    model_checkpoint_dir: str,
    model_checkpoint_filename: str,
    pretrained_model_filepath: str,
    pretrained_model_epoch_nr: int,
) -> None:
    """
    Trains a Word2vec model using skipgram negative sampling
    """
    print(f"-- Training Word2vec on {text_filepath}... --")

    # Initialize data generator
    print("Initializing data generator...")
    train_generator = SGNSDataGenerator(
        text_filepath,
        vocab_filepath,
        batch_size,
        sampling_window_size,
        num_negative_samples,
        vocab_size,
        sampling_factor,
    )
    vocab_size = train_generator.vocab_size
    print("Done!")

    # Initialize model
    if pretrained_model_filepath == "":
        print("Initializing Word2vec model...")
        model = build_word2vec_model(vocab_size, vector_dim, learning_rate)
        model.summary()
    else:
        print("Loading Word2vec model...")
        model = load_model(pretrained_model_filepath)
    print("Done!")

    # Initialize callbacks
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    model_checkpoint_filepath = join_path(model_checkpoint_dir, model_checkpoint_filename)
    model_checkpoint = ModelCheckpoint(filepath=model_checkpoint_filepath, monitor="loss")
    model_callbacks = [model_checkpoint]

    if pretrained_model_filepath == "":
        epoch_range = range(n_epochs)
    else:
        epoch_range = range(
            pretrained_model_epoch_nr, pretrained_model_epoch_nr + n_epochs
        )

    # Fit model
    print(f"-- Training for {n_epochs} epochs... --")
    for epoch in epoch_range:
        # print(f"Epoch {epoch + 1}/{n_epochs}")
        model.fit(
            train_generator,
            epochs=epoch + 1,
            initial_epoch=epoch,
            callbacks=model_callbacks,
        )


if __name__ == "__main__":
    args = parse_args()

    # Perform training
    train_word2vec_sgns(
        args.text_filepath,
        args.vocab_filepath,
        args.batch_size,
        args.n_epochs,
        args.learning_rate,
        args.vocab_size,
        args.vector_dim,
        args.sampling_window_size,
        args.num_negative_samples,
        args.sampling_factor,
        args.model_checkpoint_dir,
        args.model_checkpoint_filename,
        args.pretrained_model_filepath,
        args.pretrained_model_epoch_nr,
    )
