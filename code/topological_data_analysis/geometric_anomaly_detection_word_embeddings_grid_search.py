import argparse
import sys
from os import makedirs
from os.path import join

import joblib
import numpy as np
from geometric_anomaly_detection import GeometricAnomalyDetection
from sklearn.metrics import euclidean_distances

sys.path.append("..")

from word_embeddings.word2vec import load_model_training_output  # noqa: E402


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
        "--model_dir",
        type=str,
        default="",
        help="Directory of the model to load",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="word2vec",
        help="Name of the trained word2vec model",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="enwiki",
        help="Name of the dataset the model is trained on",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="Size of the vocabulary to use",
    )
    parser.add_argument(
        "--annoy_index_filepath",
        type=str,
        help="Filepath of Annoy index fit on word embeddings",
    )
    parser.add_argument(
        "--manifold_dimension",
        type=int,
        help="Manifold dimension to be passed to geometric anomaly detection algorithm",
    )
    parser.add_argument(
        "--num_radii_to_use",
        type=int,
        help="Number of radii parameters to use at most (all for outer radius and (all - 1) for inner radius)",
    )
    parser.add_argument(
        "--max_annulus_radii_diff",
        type=float,
        default=np.inf,
        help="Maximal difference between outer and inner radii for annulus",
    )
    parser.add_argument(
        "--use_ripser_plus_plus",
        default=False,
        action="store_true",
        help="Whether or not to use Ripser++ and GPUs for computing Rips complices",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        help="Number of CPUs to use (defaults -1 = to all CPUs)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory to save data",
    )
    return parser.parse_args()


def geometric_anomaly_detection_grid_search(
    model_dir: str,
    model_name: str,
    dataset_name: str,
    annoy_index_filepath: str,
    vocab_size: int,
    manifold_dimension: int,
    num_radii_to_use: int,
    max_annulus_radii_diff: float,
    use_ripser_plus_plus: bool,
    num_cpus: int,
    output_dir: str,
) -> None:
    """
    Performs grid search to find best set of annulus radii (inner and outer)
    for computing geometric data anomaly detection on word embeddings.

    Parameters
    ----------
    model_dir : str
        Directory of the model to load.
    model_name : str
        Name of the trained word2vec model.
    dataset_name : str
        Name of the dataset the model is trained on.
    vocab_size : int
        Size of the vocabulary to use.
    annoy_index_filepath : str
        Filepath of Annoy index fit on word embeddings.
    manifold_dimension : int
        Manifold dimension to be passed to geometric anomaly detection algorithm.
    num_radii_to_use : int
        Number of radii parameters to use at most
        (all for outer radius and (all - 1) for inner radius).
    max_annulus_radii_diff : float
        Maximal difference between outer and inner radii for annulus
    use_ripser_plus_plus : bool
        Whether or not to use Ripser++ and GPUs for computing Rips complices.
    num_cpus : int
        Number of CPUs to use (defaults -1 = to all CPUs).
    output_dir : str
        Output directory to save data
    """
    # Ensure output directory exists
    makedirs(output_dir, exist_ok=True)

    # Load output from training word2vec
    print("Loading word2vec model...")
    w2v_training_output = load_model_training_output(
        model_training_output_dir=model_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        return_normalized_embeddings=True,
    )
    last_embedding_weights_normalized = w2v_training_output[
        "last_embedding_weights_normalized"
    ]
    model_id = f"{model_name}_{dataset_name}"
    print("Done!")

    # Compute pairwise distances for grid search using specified vocab size
    vocabulary_word_ints = list(range(vocab_size))
    word_embeddings_pairwise_dists_grid_search = euclidean_distances(
        last_embedding_weights_normalized[vocabulary_word_ints]
    )

    # Initialize GAD instance
    gad_instance = GeometricAnomalyDetection(
        word_embeddings=last_embedding_weights_normalized
    )

    # Do grid search
    (
        best_gad_result_idx,
        P_man_counts,
        gad_results,
        annulus_radii_grid,
    ) = gad_instance.grid_search_radii(
        word_ints=vocabulary_word_ints,
        manifold_dimension=manifold_dimension,
        num_radii_per_parameter=num_radii_to_use,
        annoy_index_filepath=annoy_index_filepath,
        outer_inner_radii_max_diff=max_annulus_radii_diff,
        word_embeddings_pairwise_dists=word_embeddings_pairwise_dists_grid_search,
        use_ripser_plus_plus=use_ripser_plus_plus,
        num_cpus=num_cpus,
    )
    grid_search_result = {
        "best_gad_result_idx": best_gad_result_idx,
        "P_man_counts": P_man_counts,
        "gad_results": gad_results,
        "annulus_radii_grid": annulus_radii_grid,
    }
    grid_search_result_filepath = join(
        output_dir, f"{model_id}_grid_search_result.joblib"
    )
    joblib.dump(grid_search_result, grid_search_result_filepath)


if __name__ == "__main__":
    args = parse_args()
    geometric_anomaly_detection_grid_search(
        model_dir=args.model_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        annoy_index_filepath=args.annoy_index_filepath,
        vocab_size=args.vocab_size,
        manifold_dimension=args.manifold_dimension,
        num_radii_to_use=args.num_radii_to_use,
        max_annulus_radii_diff=args.max_annulus_radii_diff,
        use_ripser_plus_plus=args.use_ripser_plus_plus,
        num_cpus=args.num_cpus,
        output_dir=args.output_dir,
    )
