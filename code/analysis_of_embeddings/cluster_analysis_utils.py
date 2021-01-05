import sys
from os.path import join
from typing import Union

import joblib
import numpy as np
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

sys.path.append("..")
from utils import pairwise_cosine_distances, words_to_vectors


def create_linkage_matrix(clustering: AgglomerativeClustering) -> np.ndarray:
    """
    Creates a linkage matrix from an agglomerative clustering.

    Code snippet source:
    https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    Downloaded 7th of December 2020.

    Parameters
    ----------
    clustering : AgglomerativeClustering
        Agglomerative clustering

    Returns
    -------
    linkage_matrix : np.ndarray
        Linkage matrix
    """
    # Create the counts of samples under each node
    counts = np.zeros(clustering.children_.shape[0])
    n_samples = len(clustering.labels_)
    for i, merge in enumerate(clustering.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    # Create required linkage matrix
    linkage_matrix = np.column_stack(
        [clustering.children_, clustering.distances_, counts]
    ).astype(np.float)

    return linkage_matrix


def save_cluster_result_to_disk(
    cluster_result: dict,
    output_dir: str,
    model_name: str,
    dataset_name: str,
    output_filepath_suffix: str,
) -> None:
    """
    Saves cluster result to disk.

    Parameters
    ----------
    cluster_result : dict
        Dictionary containing result from clustering
    output_dir : str
        Output directory
    model_name : str
        Name of the model
    dataset_name : str
        Name of the dataset the model was trained on
    output_filepath_suffix : str
        Output filepath suffix
    """
    # Save result to output dir
    joblib.dump(
        cluster_result,
        join(output_dir, f"{model_name}-{dataset_name}-{output_filepath_suffix}.joblib"),
    )


def cluster_analysis(
    clusterers: list,
    hyperparameter_grids: list,
    eval_metrics_grid: list,
    eval_metrics_params: list,
    word_embeddings: np.ndarray,
    words_vocabulary: list,
    word_to_int: dict,
    compute_pairwise_word_distances: bool = False,
    agglomerative_clustering_optimized: bool = False,
    return_word_vectors: bool = False,
    save_result_to_disk: bool = False,
    output_dir: str = None,
    model_name: str = None,
    dataset_name: str = None,
    output_filepath_suffix: str = None,
) -> Union[dict, tuple]:
    """
    TODO: Docs
    """
    # Create word vectors from given words/vocabulary
    word_vectors = words_to_vectors(
        words_vocabulary=words_vocabulary,
        word_to_int=word_to_int,
        word_embeddings=word_embeddings,
    )
    if compute_pairwise_word_distances:
        word_vectors_pairwise_distances = pairwise_cosine_distances(word_vectors)

    # If we should do agglomerative clustering first (for faster clustering)
    agglomerative_clustering_idx = [
        i
        for i, (_, clusterer_cls) in enumerate(clusterers)
        if clusterer_cls is AgglomerativeClustering
    ]
    fast_agglomerative_clustering = (
        agglomerative_clustering_optimized and len(agglomerative_clustering_idx) > 0
    )
    if fast_agglomerative_clustering:
        agglomerative_clustering_idx = agglomerative_clustering_idx[0]
        param_grid = ParameterGrid(hyperparameter_grids[agglomerative_clustering_idx])
        for params in param_grid:
            params_copy = params.copy()
            params_copy.pop(
                "n_clusters", None
            )  # Ensure we don't override n_clusters=None

            # Do agglomerative clustering
            agglomerative_clustering_instance = AgglomerativeClustering(
                n_clusters=None, distance_threshold=0, **params_copy
            )
            if (
                compute_pairwise_word_distances
                and params_copy.get("affinity") == "precomputed"
            ):
                agglomerative_clustering_instance.fit(word_vectors_pairwise_distances)
            else:
                agglomerative_clustering_instance.fit(word_vectors)

            # Create required linkage matrix for fcluster function
            agglomerative_clustering_linkage_matrix = create_linkage_matrix(
                clustering=agglomerative_clustering_instance
            )

            # Set result
            clusterers[agglomerative_clustering_idx] = (
                clusterers[agglomerative_clustering_idx][0],
                {
                    "clustering": agglomerative_clustering_instance,
                    "linkage_matrix": agglomerative_clustering_linkage_matrix,
                },
            )

    # Perform cluster analysis
    clusterers_result = {}
    unique_cluster_metrics = set()
    for (clusterer_name, clusterer_cls), hyperparameter_grid, eval_metrics in zip(
        clusterers, hyperparameter_grids, eval_metrics_grid
    ):
        print(f"-- Clustering using {clusterer_name} --")
        clusterers_result[clusterer_name] = {
            "cluster_labels": [],
            "cluster_params": [],
            "cluster_metrics": {},
        }

        # Do clustering for each set of hyperparameters
        param_grid = ParameterGrid(hyperparameter_grid)
        for params_idx, params in enumerate(tqdm(param_grid)):
            clusterers_result[clusterer_name]["cluster_params"].append(params)
            if (
                fast_agglomerative_clustering
                and isinstance(clusterer_cls, dict)
                and "linkage_matrix" in clusterer_cls
            ):
                predicted_labels = fcluster(
                    Z=clusterer_cls["linkage_matrix"],
                    criterion="maxclust",
                    t=params["n_clusters"],
                )
                clusterer_instance = None
            else:
                clusterer_instance = clusterer_cls(**params)
                if (
                    compute_pairwise_word_distances
                    and params.get("affinity") == "precomputed"
                ):
                    clusterer_instance.fit(word_vectors_pairwise_distances)
                else:
                    clusterer_instance.fit(word_vectors)
                predicted_labels = clusterer_instance.labels_
            clusterers_result[clusterer_name]["cluster_labels"].append(predicted_labels)

            # Evaluate predicted cluster labels using internal evaluation metrics
            for eval_metric_tuple in eval_metrics:
                eval_metric_key, eval_metric = eval_metric_tuple
                eval_metric_params = eval_metrics_params.get(eval_metric_key, {})
                if (
                    compute_pairwise_word_distances
                    and eval_metric_params.get("metric") == "precomputed"
                ):
                    metric_name, metric_score, metric_obj_max = eval_metric(
                        word_embeddings=word_vectors_pairwise_distances,
                        cluster_labels=predicted_labels,
                        clusterer=clusterer_instance,
                        **eval_metric_params,
                    )
                else:
                    metric_name, metric_score, metric_obj_max = eval_metric(
                        word_embeddings=word_vectors,
                        cluster_labels=predicted_labels,
                        clusterer=clusterer_instance,
                        **eval_metric_params,
                    )
                unique_cluster_metrics.add(metric_name)

                # Initialize metric result
                if (
                    metric_name
                    not in clusterers_result[clusterer_name]["cluster_metrics"]
                ):
                    clusterers_result[clusterer_name]["cluster_metrics"][metric_name] = {
                        "metric_scores": [],
                        "metric_obj_max": metric_obj_max,
                        "best_metric_score_indices": [],
                    }

                clusterers_result[clusterer_name]["cluster_metrics"][metric_name][
                    "metric_scores"
                ].append(metric_score)

                # Set best metric score indices
                if params_idx == len(param_grid) - 1:
                    best_metric_score_indices = np.argsort(
                        clusterers_result[clusterer_name]["cluster_metrics"][metric_name][
                            "metric_scores"
                        ]
                    )
                    if metric_obj_max:
                        best_metric_score_indices = best_metric_score_indices[::-1]
                    clusterers_result[clusterer_name]["cluster_metrics"][metric_name][
                        "best_metric_score_indices"
                    ] = best_metric_score_indices

    # Find preferred clusterers for each cluster metric (from best to worst)
    metric_preferred_clusterers = {}
    for cluster_metric_name in unique_cluster_metrics:
        metric_obj_max = None
        metric_best_scores = []
        clusterer_names = []
        for clusterer_name, clusterer_result in clusterers_result.items():
            if cluster_metric_name in clusterer_result["cluster_metrics"]:
                clusterer_names.append(clusterer_name)
                metric_result = clusterer_result["cluster_metrics"][cluster_metric_name]
                if metric_obj_max is None:
                    metric_obj_max = metric_result["metric_obj_max"]
                best_metric_score = metric_result["metric_scores"][
                    metric_result["best_metric_score_indices"][0]
                ]
                metric_best_scores.append(best_metric_score)
        clusterer_names = np.array(clusterer_names)
        metric_best_scores = np.array(metric_best_scores)

        metric_best_scores_sorted_indices = np.argsort(metric_best_scores)
        if metric_obj_max:
            metric_best_scores_sorted_indices = metric_best_scores_sorted_indices[::-1]
        metric_preferred_clusterers[cluster_metric_name] = {
            "clusterer_names": clusterer_names[metric_best_scores_sorted_indices],
            "best_metric_scores": metric_best_scores[metric_best_scores_sorted_indices],
        }

    # Return result as dictionary
    cluster_analysis_result = {
        "clusterers": clusterers_result,
        "metric_preferred_clusterers": metric_preferred_clusterers,
    }

    # Save result to disk
    if save_result_to_disk:
        save_cluster_result_to_disk(
            cluster_result=cluster_analysis_result,
            output_dir=output_dir,
            model_name=model_name,
            dataset_name=dataset_name,
            output_filepath_suffix=output_filepath_suffix,
        )

    if return_word_vectors:
        if compute_pairwise_word_distances:
            return cluster_analysis_result, word_vectors, word_vectors_pairwise_distances
        else:
            return cluster_analysis_result, word_vectors
    else:
        return cluster_analysis_result
