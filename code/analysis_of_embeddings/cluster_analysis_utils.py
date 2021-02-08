import sys
from copy import deepcopy
from os.path import join
from typing import Union

import joblib
import numpy as np
import plotly.graph_objects as go
from hdbscan import HDBSCAN
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import cut_tree
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

sys.path.append("..")

import analysis_utils

from utils import pairwise_cosine_distances, words_to_vectors
from vis_utils import plot_word_vectors


def separate_noise_labels_into_clusters(
    labels: np.ndarray, noise_label: int = -1
) -> np.ndarray:
    """
    Separates noise labels into their own clusters

    Parameters
    ----------
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.  (`noise_label` - for noise)
    noise_label : int
        Noise label (defaults to -1)

    Returns
    -------
    new_labels : array-like, shape (n_samples,)
        Modified predicted labels for each sample, where each data point with label
        `noise_label` are separated into its own cluster.
    """
    new_labels = labels.copy()
    max_label = np.max(new_labels)
    j = max_label + 1
    for i in range(len(new_labels)):
        if new_labels[i] == noise_label:
            new_labels[i] = j
            j += 1
    return new_labels


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
    word_embeddings_normalized: np.ndarray = None,
    compute_pairwise_word_distances: bool = False,
    compute_pairwise_word_distances_normalized: bool = False,
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
        of the form (clusterer_name, clusterer_cls). A third element of the tuples
        can be set to indicate whether or not to use normalized word vectors.
    hyperparameter_grids : list
        List of dictionaries with hyperparameters, sent to ParameterGrid for each
        respective clusterer.
    eval_metrics_grid : list
        List of internal cluster evaluation metrics used for each respective clusterer.
        Each element of the list is a tuple of the form (eval_metric_key, eval_metric_func).
        An optional third element of the tuples can be set to denote whether
        or not to use normalized word vectors (boolean).
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
    word_embeddings_normalized : np.ndarray
        Normalized word embeddings. Used for "euclidean only" algorithms.
    compute_pairwise_word_distances : bool
        Whether or not to compute the pairwise distances between each word vector
        (defaults to False). Set this to true if you want to use the pairwise distances
        for fitting clusterers or in evaluation metrics.
    compute_pairwise_word_distances_normalized : bool
        Whether or not to compute the pairwise distances between each normalized
        word vector (defaults to False). Set this to true if you want to use the
        normalized pairwise distances for fitting clusterers or in evaluation metrics.

        If it is set to True, then word_embeddings_normalized must be specified as well.
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

    # Create normalized word vectors from given words/vocabulary if specified.
    word_vectors_normalized = None
    if word_embeddings_normalized is not None:
        word_vectors_normalized = words_to_vectors(
            words_vocabulary=words_vocabulary,
            word_to_int=word_to_int,
            word_embeddings=word_embeddings_normalized,
        )

    if compute_pairwise_word_distances:
        word_vectors_pairwise_distances = pairwise_cosine_distances(word_vectors)
    if compute_pairwise_word_distances_normalized and word_vectors_normalized is not None:
        normalized_word_vectors_pairwise_distances = euclidean_distances(
            word_vectors_normalized
        )

    # If we should do agglomerative clustering first (for faster clustering)
    agglomerative_clustering_idx = [
        i
        for i, clusterer_tuple in enumerate(clusterers)
        if clusterer_tuple[1] is AgglomerativeClustering
    ]
    fast_agglomerative_clustering = len(agglomerative_clustering_idx) > 0
    if fast_agglomerative_clustering:
        print("Pre-computing agglomerative clustering...")
        agglomerative_clustering_idx = agglomerative_clustering_idx[0]
        param_grid = hyperparameter_grids[agglomerative_clustering_idx]
        linkages = param_grid.get("linkage")
        affinity = param_grid.get("affinity", ["euclidean"])[0]

        clusterers = deepcopy(clusterers)
        agglomerative_clusterings = {}
        for linkage in linkages:

            # Do agglomerative clustering
            agglomerative_clustering_instance = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0,
                linkage=linkage,
                affinity=affinity if linkage != "ward" else "euclidean",
            )
            if linkage == "ward" and word_vectors_normalized is not None:
                agglomerative_clustering_instance.fit(word_vectors_normalized)
            else:
                if affinity == "precomputed" and compute_pairwise_word_distances:
                    agglomerative_clustering_instance.fit(word_vectors_pairwise_distances)
                else:
                    agglomerative_clustering_instance.fit(word_vectors)

            # Create required linkage matrix for cut_tree function
            agglomerative_clustering_linkage_matrix = create_linkage_matrix(
                clustering=agglomerative_clustering_instance
            )

            # Set result
            agglomerative_clusterings[linkage] = {
                "clustering": agglomerative_clustering_instance,
                "linkage_matrix": agglomerative_clustering_linkage_matrix,
            }

        # Set result
        clusterers[agglomerative_clustering_idx] = (
            clusterers[agglomerative_clustering_idx][0],
            agglomerative_clusterings,
        )
        print("Done!")

    # Perform cluster analysis
    clusterers_result = {}
    unique_cluster_metrics = set()
    for clusterer_tuple, hyperparameter_grid, eval_metrics in zip(
        clusterers, hyperparameter_grids, eval_metrics_grid
    ):
        if len(clusterer_tuple) == 3:
            (clusterer_name, clusterer_cls, clusterer_use_normalized) = clusterer_tuple
        else:
            clusterer_use_normalized = False
            (clusterer_name, clusterer_cls) = clusterer_tuple
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
            if fast_agglomerative_clustering and isinstance(clusterer_cls, dict):
                agglomerative_clustering = clusterer_cls[params["linkage"]]
                predicted_labels = cut_tree(
                    Z=agglomerative_clustering["linkage_matrix"],
                    n_clusters=params["n_clusters"],
                ).T[0]
                clusterer_instance = None
            else:
                clusterer_instance = clusterer_cls(**params)
                if (
                    (
                        clusterer_use_normalized
                        and compute_pairwise_word_distances_normalized
                    )
                    or compute_pairwise_word_distances
                ) and (
                    params.get("affinity") == "precomputed"
                    or params.get("metric") == "precomputed"
                ):
                    if (
                        clusterer_use_normalized
                        and compute_pairwise_word_distances_normalized
                    ):
                        fit_predict_X = normalized_word_vectors_pairwise_distances
                    elif compute_pairwise_word_distances:
                        fit_predict_X = word_vectors_pairwise_distances
                else:
                    if clusterer_use_normalized and word_vectors_normalized is not None:
                        fit_predict_X = word_vectors_normalized
                    else:
                        fit_predict_X = word_vectors

                # Use fit_predict if it is available.
                if getattr(clusterer_instance, "fit_predict", None) is not None:
                    predicted_labels = clusterer_instance.fit_predict(fit_predict_X)
                else:
                    clusterer_instance.fit(fit_predict_X)
                    predicted_labels = clusterer_instance.predict(fit_predict_X)

                # Separate noise labels into clusters
                if clusterer_cls is HDBSCAN:
                    predicted_labels = separate_noise_labels_into_clusters(
                        predicted_labels
                    )

            clusterers_result[clusterer_name]["cluster_labels"].append(predicted_labels)

            # Evaluate predicted cluster labels using internal evaluation metrics
            for eval_metric_tuple in eval_metrics:
                if len(eval_metric_tuple) == 3:
                    (
                        eval_metric_key,
                        eval_metric,
                        eval_metric_use_normalized,
                    ) = eval_metric_tuple
                else:
                    eval_metric_use_normalized = False
                    (eval_metric_key, eval_metric) = eval_metric_tuple
                eval_metric_params = eval_metrics_params.get(eval_metric_key, {})
                if (
                    compute_pairwise_word_distances
                    and eval_metric_params.get("metric") == "precomputed"
                ):
                    if (
                        eval_metric_use_normalized
                        and compute_pairwise_word_distances_normalized
                    ):
                        metric_name, metric_score, metric_obj_max = eval_metric(
                            word_embeddings=normalized_word_vectors_pairwise_distances,
                            cluster_labels=predicted_labels,
                            clusterer=clusterer_instance,
                            **eval_metric_params,
                        )
                    else:
                        metric_name, metric_score, metric_obj_max = eval_metric(
                            word_embeddings=word_vectors_pairwise_distances,
                            cluster_labels=predicted_labels,
                            clusterer=clusterer_instance,
                            **eval_metric_params,
                        )
                else:
                    if eval_metric_use_normalized and word_vectors_normalized is not None:
                        metric_name, metric_score, metric_obj_max = eval_metric(
                            word_embeddings=word_vectors_normalized,
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

    if return_word_vectors:
        if compute_pairwise_word_distances:
            cluster_analysis_result = (
                cluster_analysis_result,
                word_vectors,
                word_vectors_pairwise_distances,
            )
        else:
            cluster_analysis_result = (cluster_analysis_result, word_vectors)

    # Save result to disk
    if save_result_to_disk:
        save_cluster_result_to_disk(
            cluster_result=cluster_analysis_result,
            output_dir=output_dir,
            model_name=model_name,
            dataset_name=dataset_name,
            output_filepath_suffix=output_filepath_suffix,
        )

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
                fig.update_layout(
                    {f"xaxis{i}": dict(tickmode="array", tickvals=xs, ticktext=xs)},
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
                analysis_utils.plot_cluster_metric_scores(
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
            analysis_utils.plot_cluster_metric_scores(
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


def plot_word_embeddings_clustered(
    transformed_word_embeddings: dict,
    words: np.ndarray,
    cluster_labels: np.ndarray,
    embedder_labels=None,
    embedder_keys: list = None,
    print_words_in_clusters: bool = False,
    continuous_word_colors: bool = False,
) -> None:
    """
    Plots transformed word embeddings with some given cluster labels.

    Parameters
    ----------
    transformed_word_embeddings : dict
        Transformed word embeddings dictionary, as returned from `transform_word_embeddings`.
    words : np.ndarray
        List of words to plot.
    cluster_labels : np.ndarray
        Cluster labels to plot.
    embedder_labels : dict
        Dictionary containing x_label and y_label for each embedder (defaults to empty dict).
    embedder_keys : list
        List of embedders (as keys) to plot (defaults to all embedders)
    print_words_in_clusters : bool
        Whether or not to print words in clusters
    continuous_word_colors : bool
        Whether or not to make the word color continuous (defaults to False).
    """
    if embedder_labels is None:
        embedder_labels = {}
    if embedder_keys is None:
        embedder_keys = list(transformed_word_embeddings.keys())

    cluster_size = len(np.unique(cluster_labels))
    for embedder_key in embedder_keys:
        if embedder_key in embedder_labels:
            x_label = embedder_labels[embedder_key]["x_label"]
            y_label = embedder_labels[embedder_key]["y_label"]
        else:
            x_label = f"{embedder_key}1"
            y_label = f"{embedder_key}2"
        plot_word_vectors(
            transformed_word_embeddings=transformed_word_embeddings[embedder_key],
            words=words,
            title=f"Embedding of words in {embedder_key} coordinates with {cluster_size} clusters",
            x_label=x_label,
            y_label=y_label,
            word_colors=cluster_labels,
            interactive=True,
            continuous_word_colors=continuous_word_colors,
        )

    if print_words_in_clusters:
        cluster_words, _ = analysis_utils.words_in_clusters(cluster_labels, words)
        print("-- Words in clusters --")
        for word_cluster in cluster_words:
            print("Words", word_cluster)
