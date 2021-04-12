import argparse
from os.path import join

import numpy as np

rng_seed = 399
np.random.seed(rng_seed)

# Clustering
from gudhi.clustering.tomato import Tomato  # noqa: E402
from hdbscan import HDBSCAN  # noqa: E402
from sklearn.cluster import (  # noqa: E402
    AgglomerativeClustering,
    KMeans,
    MiniBatchKMeans,
)
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn_extra.cluster import KMedoids  # noqa: E402

# Directory constants
analysis_of_embeddings_dir = ".."
root_code_dir = join(analysis_of_embeddings_dir, "..")

# Extend sys path for importing custom Python files
import sys  # noqa: E402

sys.path.extend([analysis_of_embeddings_dir, root_code_dir])

from analysis_of_embeddings.cluster_analysis_metrics import (  # noqa: E402
    calinski_harabasz_score_metric,
    davies_bouldin_score_metric,
    silhouette_score_metric,
)
from analysis_of_embeddings.cluster_analysis_utils import cluster_analysis  # noqa: E402
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
        help="Directory of the word2vec model to perform cluster analysis on",
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
        default=10000,
        help="Vocabulary size to use",
    )
    parser.add_argument(
        "--output_filepath_suffix",
        type=str,
        default="cluster_labels",
        help="Output filepath suffix to use when saving result to disk",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory to save processed data",
    )
    return parser.parse_args()


def cluster_analysis_all_algorithms(
    model_dir: str,
    model_name: str,
    dataset_name: str,
    vocab_size: int,
    output_filepath_suffix: str,
    output_dir: str,
) -> None:
    """
    Performs cluster analysis on word2vec word embeddings using a variety of clustering
    algorithms and internal cluster evaluation methods. Result is saved to file.

    Parameters
    ----------
    model_dir : str
        Directory of the word2vec model to perform cluster analysis on.
    model_name : str
        Name of the trained word2vec model.
    dataset_name : str
        Name of the dataset the model is trained on.
    vocab_size : int
        Vocabulary size to use.
    output_filepath_suffix : str
        Output filepath suffix to use when saving result to disk.
    output_dir : str
        Output directory to save processed data.
    """
    # Load output from training word2vec
    print("Loading word2vec model...")
    w2v_training_output = load_model_training_output(
        model_training_output_dir=model_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        return_normalized_embeddings=True,
    )
    last_embedding_weights = w2v_training_output["last_embedding_weights"]
    last_embedding_weights_normalized = w2v_training_output[
        "last_embedding_weights_normalized"
    ]
    word_to_int = w2v_training_output["word_to_int"]
    print("Done!")

    # Restrict vocabulary size for analysis
    vocabulary = list(range(vocab_size))

    # Define hyperparameters for cluster analysis
    n_clusters = [
        2,
        3,
        4,
        5,
        10,
        50,
        100,
        150,
        200,
        300,
        400,
        500,
        750,
        1000,
        1500,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
    ]
    general_eval_metrics = [
        ("silhouette_score", silhouette_score_metric),
        ("davies_bouldin_score", davies_bouldin_score_metric, True),
        ("calinski_harabasz_score", calinski_harabasz_score_metric, True),
    ]
    eval_metrics_grid = [
        general_eval_metrics,
        general_eval_metrics,
        general_eval_metrics,
        general_eval_metrics,
        general_eval_metrics,
        general_eval_metrics,
        general_eval_metrics,
    ]
    eval_metrics_params = {
        "silhouette_score": {"metric": "precomputed"},
    }
    clusterers = [
        ("ToMATo", Tomato, True),
        ("Agglomerative clustering", AgglomerativeClustering),
        ("GMM clustering", GaussianMixture, True),
        ("HDBSCAN", HDBSCAN),
        ("K-means clustering", KMeans, True),
        ("Mini-batch k-means clustering", MiniBatchKMeans, True),
        ("K-medoids clustering", KMedoids),
    ]
    hyperparameter_grids = [
        {
            "density_type": ["DTM", "logDTM", "KDE", "logKDE"],
            "k": [
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                20,
                30,
                40,
                50,
                100,
                150,
                200,
                250,
            ],
            "n_jobs": [-1],
        },
        {
            "n_clusters": n_clusters,
            "affinity": ["precomputed"],
            "linkage": ["single", "average", "complete", "ward"],
        },
        {"n_components": n_clusters, "random_state": [rng_seed]},
        {
            "min_cluster_size": [2, 4, 8, 16, 32, 64],
            "min_samples": [1, 2, 4, 8, 16, 32, 64],
            "metric": ["precomputed"],
            "core_dist_n_jobs": [-1],
        },
        {"n_clusters": n_clusters, "random_state": [rng_seed]},
        {
            "n_clusters": n_clusters,
            "random_state": [rng_seed],
            "batch_size": [100],
        },
        {
            "n_clusters": n_clusters,
            "random_state": [rng_seed],
            "metric": ["precomputed"],
        },
    ]

    print("Cluster analysis starting...")
    _ = cluster_analysis(
        clusterers=clusterers,
        hyperparameter_grids=hyperparameter_grids,
        eval_metrics_grid=eval_metrics_grid,
        eval_metrics_params=eval_metrics_params,
        word_embeddings=last_embedding_weights,
        words_vocabulary=vocabulary,
        word_to_int=word_to_int,
        word_embeddings_normalized=last_embedding_weights_normalized,
        compute_pairwise_word_distances=True,
        compute_pairwise_word_distances_normalized=True,
        return_word_vectors=True,
        save_result_to_disk=True,
        output_dir=output_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        output_filepath_suffix=output_filepath_suffix,
    )
    print("Done!")


if __name__ == "__main__":
    args = parse_args()

    # Perform evaluation
    cluster_analysis_all_algorithms(
        model_dir=args.model_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        vocab_size=args.vocab_size,
        output_filepath_suffix=args.output_filepath_suffix,
        output_dir=args.output_dir,
    )
