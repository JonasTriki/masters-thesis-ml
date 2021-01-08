import sys
from os.path import join
from typing import Union

import joblib
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

sys.path.append("..")
from analysis_utils import plot_cluster_metric_scores

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
    eval_metrics_params: dict,
    word_embeddings: np.ndarray,
    words_vocabulary: list,
    word_to_int: dict,
    compute_pairwise_word_distances: bool = False,
    return_word_vectors: bool = False,
    save_result_to_disk: bool = False,
    output_dir: str = None,
    model_name: str = None,
    dataset_name: str = None,
    output_filepath_suffix: str = None,
) -> Union[dict, tuple]:
    """
    Performs cluster hyperparameter and algorithm search over a range of clusterers
    and set of hyperparameters. Uses internal cluster evaluation metrics to select
    best performing clusterer (with some set of hyperparameters).

    Parameters
    ----------
    clusterers : list
        List of clusterer classes, where each element in the list should be a tuple
        of the form (clusterer_name, clusterer_cls).
    hyperparameter_grids : list
        List of dictionaries with hyperparameters, sent to ParameterGrid for each
        respective clusterer.
    eval_metrics_grid : list
        List of internal cluster evaluation metrics used for each respective clusterer.
        Each element of the list is a tuple of the form (eval_metric_key, eval_metric_func).
    eval_metrics_params : dict
        Dictionary containing kwargs given to internal cluster evaluation metrics
        before computing metric score. The dictionary maps from eval_metric_key to
        some kwargs (dictionary).
    word_embeddings : np.ndarray
        Word embeddings
    words_vocabulary : list
        List of either words (str) or word integer representations (int), signalizing
        what part of the vocabulary we want to use.
    word_to_int : dict of str and int
        Dictionary mapping from word to its integer representation.
    compute_pairwise_word_distances : bool
        Whether or not to compute the pairwise distances between each word vector
        (defaults to False). Set this to true if you want to use the pairwise distances
        for fitting clusterers or in evaluation metrics.
    return_word_vectors : bool
        Whether or not to return word vectors and pairwise distances as well
        (if compute_pairwise_word_distances is set to True). Defaults to False.
    save_result_to_disk : bool
        Whether or not to save the cluster analysis result to disk (defaults to False).
        If true, then output_dir, model_name, dataset_name and output_filepath_suffix must
        be set accordingly.
    output_dir : str
        Output directory to save the result to (defaults to None).
    model_name : str
        Name of the trained model in which the word embeddings are from (defaults to None).
    dataset_name : str
        Name of the dataset the word embeddings model is trained on (defaults to None).
    output_filepath_suffix : str
        Output filepath suffix to use when saving result to disk (defaults to None).

    Returns
    -------
    cluster_analysis_result : dict or tuple
        Cluster analysis result as dictionary. If return_word_vectors is true,
        then word vectors are returned as well in addition to the cluster analysis result
        in a tuple. If both return_word_vectors and compute_pairwise_word_distances are true,
        then the pairwise word distances are returned as well.
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
    fast_agglomerative_clustering = len(agglomerative_clustering_idx) > 0
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


def visualize_cluster_analysis_result(
    cluster_analysis_result: dict,
    print_hyperparameters: bool = True,
    interactive: bool = False,
) -> None:
    """
    Visualizes cluster analysis results from `cluster_analysis` function.

    Parameters
    ----------
    cluster_analysis_result : dict
        Cluster analysis result as returned from `cluster_analysis`.
    print_hyperparameters : bool
        Whether or not to print out hyperparameters prior to plotting
        metric scores for each clusterer (defaults to True).
    interactive : bool
        Whether or not to use interactive plotting with Plotly
        (defaults to False).
    """
    for clusterer_name, clusterer_result in cluster_analysis_result["clusterers"].items():
        num_clusterer_metrics = len(clusterer_result["cluster_metrics"])
        if interactive:
            fig = make_subplots(
                rows=1,
                cols=num_clusterer_metrics,
            )
            fig.update_layout(
                title={
                    "text": clusterer_name,
                    "y": 0.9,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                }
            )
        else:
            _, axes = plt.subplots(
                nrows=1,
                ncols=num_clusterer_metrics,
                figsize=(5 * num_clusterer_metrics, 5),
            )
            plt.suptitle(clusterer_name)

        clusterer_params = clusterer_result["cluster_params"]
        if print_hyperparameters:
            print(f"Hyperparameters used for {clusterer_name}:")
            for i, hyperparams in enumerate(clusterer_params):
                print(f"{i}: {hyperparams}")
        xs = list(range(len(clusterer_params)))

        if interactive:
            for i, (clusterer_metric_name, clusterer_metric_result) in zip(
                range(1, num_clusterer_metrics + 1),
                clusterer_result["cluster_metrics"].items(),
            ):
                metric_scores = clusterer_metric_result["metric_scores"]
                best_metric_score_idx = clusterer_metric_result[
                    "best_metric_score_indices"
                ][0]

                fig.update_xaxes(title_text="Hyperparameter set (index)", row=1, col=i)
                fig.update_yaxes(
                    title_text=f"{clusterer_metric_name} score", row=1, col=i
                )
                fig.add_trace(
                    go.Scatter(x=xs, y=metric_scores, name=clusterer_metric_name),
                    row=1,
                    col=i,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[best_metric_score_idx],
                        y=[metric_scores[best_metric_score_idx]],
                        marker={
                            "size": 10,
                        },
                        name="Best score",
                        showlegend=False,
                    ),
                    row=1,
                    col=i,
                )
            fig.show()
        else:
            for ax, (clusterer_metric_name, clusterer_metric_result) in zip(
                axes, clusterer_result["cluster_metrics"].items()
            ):
                metric_scores = clusterer_metric_result["metric_scores"]
                best_metric_score_idx = clusterer_metric_result[
                    "best_metric_score_indices"
                ][0]

                ax.set_xticks(xs)
                plot_cluster_metric_scores(
                    metric_scores=metric_scores,
                    hyperparameters=clusterer_params,
                    best_score_idx=best_metric_score_idx,
                    metric_name=clusterer_metric_name,
                    xlabel="Hyperparameter set (index)",
                    set_xticks=False,
                    set_xtickslabels=False,
                    show_plot=False,
                    ax=ax,
                )
            plt.tight_layout()
            plt.show()

    total_num_clusterer_metrics = len(
        cluster_analysis_result["metric_preferred_clusterers"]
    )
    if interactive:
        fig = make_subplots(
            rows=1,
            cols=total_num_clusterer_metrics,
        )
        fig.update_layout(
            title={
                "text": "Most preferred clusterers (by metrics)",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            }
        )
        for i, (clusterer_metric_name, clusterer_metric_result) in zip(
            range(1, total_num_clusterer_metrics + 1),
            cluster_analysis_result["metric_preferred_clusterers"].items(),
        ):
            best_metric_scores = clusterer_metric_result["best_metric_scores"]
            clusterer_names = clusterer_metric_result["clusterer_names"]
            xs = list(range(len(clusterer_names)))
            print(best_metric_scores)

            fig.update_xaxes(title_text="Clusterer", row=1, col=i)
            fig.update_yaxes(title_text=f"{clusterer_metric_name} score", row=1, col=i)
            fig.update_layout(
                {
                    f"xaxis{i}": dict(
                        tickmode="array", tickvals=xs, ticktext=clusterer_names
                    )
                },
            )
            fig.add_trace(
                go.Scatter(x=xs, y=best_metric_scores, name=clusterer_metric_name),
                row=1,
                col=i,
            )
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[best_metric_scores[0]],
                    marker={
                        "size": 10,
                    },
                    name="Best score",
                    showlegend=False,
                ),
                row=1,
                col=i,
            )

        fig.show()
    else:
        _, axes = plt.subplots(
            nrows=1,
            ncols=total_num_clusterer_metrics,
            figsize=(5 * total_num_clusterer_metrics, 5),
        )
        plt.suptitle("Most preferred clusterers (by metrics)")
        for ax, (clusterer_metric_name, clusterer_metric_result) in zip(
            axes, cluster_analysis_result["metric_preferred_clusterers"].items()
        ):
            best_metric_scores = clusterer_metric_result["best_metric_scores"]
            clusterer_names = clusterer_metric_result["clusterer_names"]
            print(best_metric_scores)
            plot_cluster_metric_scores(
                metric_scores=best_metric_scores,
                hyperparameters=clusterer_names,
                best_score_idx=0,
                metric_name=clusterer_metric_name,
                xlabel="Clusterer",
                xtickslabels_rotation=45,
                show_plot=False,
                ax=ax,
            )
        plt.tight_layout()
        plt.show()
