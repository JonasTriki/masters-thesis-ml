import argparse
import os
from os.path import dirname, isfile, join
from pathlib import Path

import annoy
import numpy as np
from tqdm import tqdm
from word2vec import load_model_training_output


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
    parser.add_argument(
        "--annoy_index_n_trees",
        type=int,
        default="",
        help="Number of trees to pass to Annoy's build method. More trees => higher precision",
    )
    return parser.parse_args()


def postprocess_word2vec_embeddings(
    model_training_output_dir: str,
    model_name: str,
    dataset_name: str,
    vocab_size: int,
    annoy_index_n_trees: int,
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
    annoy_index_n_trees : int
        Number of trees to pass to Annoys build method. More trees => higher precision.
    """
    # Load output from training word2vec
    w2v_training_output = load_model_training_output(
        model_training_output_dir=model_training_output_dir,
        model_name=model_name,
        dataset_name=dataset_name,
    )
    last_embedding_weights = w2v_training_output["last_embedding_weights"]
    embedding_dim = last_embedding_weights.shape[1]

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
        model_ann_index_filepath = join(
            model_training_output_dir,
            f"{last_embedding_weights_filepath_no_ext}_annoy_index.ann",
        )
    else:
        model_ann_index_filepath = join(
            model_training_output_dir,
            f"{last_embedding_weights_filepath_no_ext}_{vocab_size}_annoy_index.ann",
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

    if not isfile(model_ann_index_filepath):

        # Add word embeddings to index and build it
        ann_index = annoy.AnnoyIndex(f=embedding_dim, metric="euclidean")
        print("Adding word embeddings to index...")
        for i in tqdm(range(vocab_size)):
            ann_index.add_item(i, last_embedding_weights_normalized[i])
        print("Done!")

        print("Building index...")
        ann_index.build(n_trees=annoy_index_n_trees, n_jobs=-1)
        print("Done!")

        print("Saving to file...")
        ann_index.save(model_ann_index_filepath)
        print("Done!")


if __name__ == "__main__":
    args = parse_args()
    postprocess_word2vec_embeddings(
        model_training_output_dir=args.model_training_output_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        vocab_size=args.vocab_size,
        annoy_index_n_trees=args.annoy_index_n_trees,
    )
