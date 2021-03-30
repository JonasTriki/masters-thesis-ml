import argparse
import sys
from os.path import isdir, isfile, join
from pathlib import Path

import numpy as np

sys.path.append("..")

from approx_nn import ApproxNN  # noqa: E402
from word_embeddings.word2vec import load_model_training_output  # noqa: E402

rng_seed = 399
np.random.seed(rng_seed)


def parse_args() -> argparse.Namespace:
    """
    Parses arguments sent to the python script.

    Returns
    -------
    parsed_args : argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_training_output_dir",
        type=str,
        default="",
        help="Word2vec model training output directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Name of the trained model",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help="Name of the dataset used to train the model",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=-1,
        help="Size of the vocabulary to use, -1 denotes all words",
    )
    return parser.parse_args()


def postprocess_word2vec_embeddings(
    model_training_output_dir: str,
    model_name: str,
    dataset_name: str,
    vocab_size: int,
) -> None:
    """
    Applies post-processing to trained word2vec word embeddings:
    - Saves normalized word embeddings
    - Creates approximate nearest-neighbour index using Annoy

    Parameters
    ----------
    model_training_output_dir : str
        word2vec model training output directory.
    model_name : str
        Name of the trained model.
    dataset_name : str
        Name of the dataset the model is trained on.
    vocab_size : int
        Size of the vocabulary to use, -1 denotes all words
    """
    # Load output from training word2vec
    w2v_training_output = load_model_training_output(
        model_training_output_dir=model_training_output_dir,
        model_name=model_name,
        dataset_name=dataset_name,
    )
    last_embedding_weights = w2v_training_output["last_embedding_weights"]

    use_full_vocab = False
    if vocab_size == -1:
        vocab_size = last_embedding_weights.shape[0]
        use_full_vocab = True

    # Define filepaths
    last_embedding_weights_filepath = w2v_training_output[
        "last_embedding_weights_filepath"
    ]
    last_embedding_weights_filepath_no_ext = Path(last_embedding_weights_filepath).stem
    if use_full_vocab:
        last_embedding_weights_normalized_filepath = join(
            model_training_output_dir,
            f"{last_embedding_weights_filepath_no_ext}_normalized.npy",
        )
    else:
        last_embedding_weights_normalized_filepath = join(
            model_training_output_dir,
            f"{last_embedding_weights_filepath_no_ext}_{vocab_size}_normalized.npy",
        )
    if use_full_vocab:
        model_annoy_index_filepath = join(
            model_training_output_dir,
            f"{last_embedding_weights_filepath_no_ext}_annoy_index.ann",
        )
        model_scann_artifacts_dir = join(
            model_training_output_dir,
            f"{last_embedding_weights_filepath_no_ext}_scann_artifacts",
        )
    else:
        model_annoy_index_filepath = join(
            model_training_output_dir,
            f"{last_embedding_weights_filepath_no_ext}_{vocab_size}_annoy_index.ann",
        )
        model_scann_artifacts_dir = join(
            model_training_output_dir,
            f"{last_embedding_weights_filepath_no_ext}_{vocab_size}_scann_artifacts",
        )

    # Normalize word embeddings and save to file
    if not isfile(last_embedding_weights_normalized_filepath):
        print("Normalizing word embeddings and saving to file...")

        # Normalize word embeddings
        if use_full_vocab:
            last_embedding_weights_in_vocab = last_embedding_weights
        else:
            last_embedding_weights_in_vocab = last_embedding_weights[:vocab_size]
        last_embedding_weights_normalized = (
            last_embedding_weights_in_vocab
            / np.linalg.norm(last_embedding_weights_in_vocab, axis=1).reshape(-1, 1)
        )
        np.save(
            last_embedding_weights_normalized_filepath,
            last_embedding_weights_normalized,
        )
        print("Done!")
    else:
        last_embedding_weights_normalized = np.load(
            last_embedding_weights_normalized_filepath
        )

    annoy_index_created = isfile(model_annoy_index_filepath)
    scann_instance_created = isdir(model_scann_artifacts_dir)
    if not annoy_index_created or not scann_instance_created:

        # Add word embeddings to index and build it
        if use_full_vocab:
            last_embedding_weights_normalized_in_vocab = (
                last_embedding_weights_normalized
            )
        else:
            last_embedding_weights_normalized_in_vocab = (
                last_embedding_weights_normalized[:vocab_size]
            )

        if not isfile(model_annoy_index_filepath):
            ann_index_annoy = ApproxNN(ann_alg="annoy")
            ann_index_annoy.build(
                data=last_embedding_weights_normalized_in_vocab,
                annoy_n_trees=500,
                distance_measure="euclidean",
            )
            ann_index_annoy.save(model_annoy_index_filepath)

        if not isdir(model_scann_artifacts_dir):
            scann_instance = ApproxNN(ann_alg="scann")
            scann_instance.build(
                data=last_embedding_weights_normalized_in_vocab,
                distance_measure="dot_product",
                scann_num_leaves_scaling=5,
            )
            scann_instance.save(model_scann_artifacts_dir)


if __name__ == "__main__":
    args = parse_args()
    postprocess_word2vec_embeddings(
        model_training_output_dir=args.model_training_output_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        vocab_size=args.vocab_size,
    )
