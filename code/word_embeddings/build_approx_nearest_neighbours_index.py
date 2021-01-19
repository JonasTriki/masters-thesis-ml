import argparse
from os import makedirs
from os.path import join

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
        "--annoy_index_n_trees",
        type=int,
        default="",
        help="Number of trees to pass to Annoy's build method. More trees => higher precision",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory to save processed data",
    )
    return parser.parse_args()


def build_approx_nearest_neighbours_index(
    model_training_output_dir: str,
    model_name: str,
    dataset_name: str,
    annoy_index_n_trees: int,
    output_dir: str,
) -> None:
    """
    Builds an approximate nearest neighbours index on the word embeddings
    of a trained word2vec model.

    Parameters
    ----------
    model_training_output_dir : str
        word2vec model training output directory.
    model_name : str
        Name of the trained model.
    dataset_name : str
        Name of the dataset the model is trained on.
    annoy_index_n_trees : int
        Number of trees to pass to Annoys build method. More trees => higher precision.
    output_dir : str
        Output directory.
    """
    # Load output from training word2vec
    w2v_training_output = load_model_training_output(
        model_training_output_dir=model_training_output_dir,
        model_name=model_name,
        dataset_name=dataset_name,
    )
    last_embedding_weights = w2v_training_output["last_embedding_weights"]
    vocab_size = last_embedding_weights.shape[0]
    embedding_dim = last_embedding_weights.shape[1]

    # Normalize word embeddings
    last_embedding_weights_normalized = last_embedding_weights / np.linalg.norm(
        last_embedding_weights, axis=1
    ).reshape(-1, 1)

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
    makedirs(output_dir)
    output_ann_index_filepath = join(
        output_dir, f"{model_name}_{dataset_name}_annoy_index.ann"
    )
    ann_index.save(output_ann_index_filepath)
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    build_approx_nearest_neighbours_index(
        model_training_output_dir=args.model_training_output_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        annoy_index_n_trees=args.annoy_index_n_trees,
        output_dir=args.output_dir,
    )
