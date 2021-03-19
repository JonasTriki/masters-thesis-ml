import argparse
import sys
from multiprocessing import Array, Pool, cpu_count
from os.path import join
from pathlib import Path

import numpy as np
from fastdist import fastdist
from tqdm import tqdm

sys.path.append("..")

from utils import batch_list_gen  # noqa: E402
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
    parser.add_argument(
        "--num_nearest_neighbours",
        type=int,
        default=100,
        help="Number of nearest neighbours to find for each word embedding",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=-1,
        help="Number of CPUs to use for multiprocessing",
    )
    return parser.parse_args()


# Multiprocessing variable dict
mp_var_dict = {}


def mp_init(word_embeddings: Array, word_embeddings_shape: tuple) -> None:
    mp_var_dict["word_embeddings"] = word_embeddings
    mp_var_dict["word_embeddings_shape"] = word_embeddings_shape


def find_nearest_neighbours(args: tuple) -> dict:
    """
    TODO: Docs

    Parameters
    ----------
    args : tuple

    Returns
    -------
    nearest_neighbour_indices : dict
    """
    # Parse args
    word_ints, num_nearest_neighbours = args

    # Load word embeddings from var dict
    word_embeddings = np.frombuffer(mp_var_dict["word_embeddings"]).reshape(
        mp_var_dict["word_embeddings_shape"]
    )

    # Find neighbours
    nearest_neighbour_indices = {}
    for i in tqdm(word_ints):

        # Find nearest neighbours using cosine similarity between word vectors
        word_vec = word_embeddings[i]
        word_vec_cosine_dims = fastdist.cosine_vector_to_matrix(
            u=word_vec, m=word_embeddings
        )
        word_vec_cosine_dims_sorted_indices = np.argsort(word_vec_cosine_dims)[::-1]

        # Set result
        word_vec_nearest_neighbour_indices = word_vec_cosine_dims_sorted_indices[
            1 : num_nearest_neighbours + 1
        ]
        nearest_neighbour_indices[i] = word_vec_nearest_neighbour_indices

    return nearest_neighbour_indices


def save_nearest_neighbours_word_embeddings(
    model_training_output_dir: str,
    model_name: str,
    dataset_name: str,
    vocab_size: int,
    num_nearest_neighbours: int,
    num_cpus: int,
) -> None:
    """
    Finds and saves exact nearest neighbours of word embeddings to disk.

    Parameters
    ----------
    model_training_output_dir : str
        word2vec model training output directory.
    model_name : str
        Name of the trained model.
    dataset_name : str
        Name of the dataset the model is trained on.
    vocab_size : int
        Size of the vocabulary to use, -1 denotes all words.
    num_nearest_neighbours : int
        Number of nearest neighbours to find for each word embedding.
    num_cpus : int
        Number of CPUs to use for multiprocessing.
    """
    # Load output from training word2vec
    print("Loading word2vec data...")
    w2v_training_output = load_model_training_output(
        model_training_output_dir=model_training_output_dir,
        model_name=model_name,
        dataset_name=dataset_name,
    )
    last_embedding_weights = w2v_training_output["last_embedding_weights"]
    use_full_vocab = False
    if vocab_size == -1:
        use_full_vocab = True
        vocab_size = last_embedding_weights.shape[0]
    print("Done!")

    # Prepare data for multiprocessing
    print("Preparing data for MP...")
    word_embeddings_shape = last_embedding_weights.shape
    word_embeddings_raw = Array(
        "d", word_embeddings_shape[0] * word_embeddings_shape[1], lock=False
    )
    word_embeddings_raw_np = np.frombuffer(word_embeddings_raw).reshape(
        word_embeddings_shape
    )
    np.copyto(word_embeddings_raw_np, last_embedding_weights)
    print("Done!")

    # Find nearest neighbours using multiprocessing
    if num_cpus == -1:
        num_cpus = cpu_count()
    num_word_ints_per_process = int(vocab_size // num_cpus)
    mp_args = [
        (word_int_chunk, num_nearest_neighbours)
        for word_int_chunk in batch_list_gen(
            np.arange(vocab_size), num_word_ints_per_process
        )
    ]

    # Run MP
    print("Running MP...")
    nearest_neighbour_indices = np.zeros((vocab_size, num_nearest_neighbours))
    with Pool(
        processes=num_cpus,
        initializer=mp_init,
        initargs=(word_embeddings_raw, word_embeddings_shape),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(find_nearest_neighbours, mp_args), total=num_cpus
        ):
            for word_int, nearest_neighbours in result.items():
                nearest_neighbour_indices[word_int] = nearest_neighbours

    # Save to file
    last_embedding_weights_filepath = w2v_training_output[
        "last_embedding_weights_filepath"
    ]
    last_embedding_weights_filepath_no_ext = Path(last_embedding_weights_filepath).stem
    if use_full_vocab:
        last_embedding_weights_neighbours_filepath = join(
            model_training_output_dir,
            f"{last_embedding_weights_filepath_no_ext}_{num_nearest_neighbours}_neighbours.npy",
        )
    else:
        last_embedding_weights_neighbours_filepath = join(
            model_training_output_dir,
            f"{last_embedding_weights_filepath_no_ext}_vocab_{vocab_size}_{num_nearest_neighbours}_neighbours.npy",
        )
    np.save(last_embedding_weights_neighbours_filepath, nearest_neighbour_indices)


if __name__ == "__main__":
    args = parse_args()
    save_nearest_neighbours_word_embeddings(
        model_training_output_dir=args.model_training_output_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        vocab_size=args.vocab_size,
        num_nearest_neighbours=args.num_nearest_neighbours,
        num_cpus=args.num_cpus,
    )
