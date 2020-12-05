from os import makedirs
from os.path import join
from typing import Callable, Union

import joblib
import numpy as np
import seaborn as sns
from hdbscan import HDBSCAN
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm


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


def k_means_cluster_hyperparameter_search(
    param_grid: ParameterGrid,
    default_params: dict,
    word_embeddings: np.ndarray,
    word_embeddings_pairwise_dists: np.ndarray,
    output_dir: str,
    model_name: str,
    dataset_name: str,
    output_filepath_suffix: str,
    clusterer: Union[
        KMeans,
        MiniBatchKMeans,
        KMedoids,
        GaussianMixture,
    ] = KMeans,
    clusterer_name: str = "K-means clustering",
) -> dict:
    """
    Searches for the best set of hyperparameters using K-means clustering
    and mean Silhouette Coefficient as the internal cluster metric.

    Parameters
    ----------
    param_grid : ParameterGrid
        Parameter grid to search through
    default_params : dict
        Default parameters to use for the clusterer
    word_embeddings : np.ndarray
        Word embeddings to perform clustering on
    word_embeddings_pairwise_dists : np.ndarray
        Numpy matrix containing pairwise distances between word embeddings
    output_dir : str
        Output directory
    model_name : str
        Name of the model
    dataset_name : str
        Name of the dataset the model was trained on
    output_filepath_suffix : str
        Output filepath suffix
    clusterer : KMeans
        The clusterer to use (defaults to KMeans)
    clusterer_name : str
        Name of the clusterer (defaults to "K-means clustering")

    Returns
    -------
    result : dict
        Dictionary containing cluster labels and metric scores
    """
    # Ensure output directory exists
    makedirs(output_dir, exist_ok=True)

    # Perform clustering
    cluster_labels = []
    cluster_metric_values = []
    for params in tqdm(param_grid, desc=f"Performing clustering using {clusterer_name}"):
        cls = clusterer(**params, **default_params)
        cluster_labels_pred = cls.fit_predict(word_embeddings)
        cluster_labels.append(cluster_labels_pred)

        # Compute Silhouette Coefficient score
        cluster_metric_value = silhouette_score(
            X=word_embeddings_pairwise_dists,
            labels=cluster_labels_pred,
            metric="precomputed",
        )
        cluster_metric_values.append(cluster_metric_value)

    result = {
        "metric_name": "Silhouette Coefficient",
        "metric_scores": cluster_metric_values,
        "cluster_labels": cluster_labels,
        "best_cluster_labels_idx": np.argmax(cluster_metric_values),
    }

    # Save result to output dir
    save_cluster_result_to_disk(
        result, output_dir, model_name, dataset_name, output_filepath_suffix
    )

    return result


def k_means_mini_batch_cluster_hyperparameter_search(
    param_grid: ParameterGrid,
    default_params: dict,
    word_embeddings: np.ndarray,
    word_embeddings_pairwise_dists: np.ndarray,
    output_dir: str,
    model_name: str,
    dataset_name: str,
    output_filepath_suffix: str,
) -> dict:
    """
    Searches for the best set of hyperparameters using mini-batch K-means
    clustering and mean Silhouette Coefficient as the internal cluster metric.

    Parameters
    ----------
    param_grid : ParameterGrid
        Parameter grid to search through
    default_params : dict
        Default parameters to use for the clusterer
    word_embeddings : np.ndarray
        Word embeddings to perform clustering on
    word_embeddings_pairwise_dists : np.ndarray
        Numpy matrix containing pairwise distances between word embeddings
    output_dir : str
        Output directory
    model_name : str
        Name of the model
    dataset_name : str
        Name of the dataset the model was trained on
    output_filepath_suffix : str
        Output filepath suffix

    Returns
    -------
    result : dict
        Dictionary containing cluster labels and metric scores
    """
    return k_means_cluster_hyperparameter_search(
        param_grid=param_grid,
        default_params=default_params,
        word_embeddings=word_embeddings,
        word_embeddings_pairwise_dists=word_embeddings_pairwise_dists,
        output_dir=output_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        output_filepath_suffix=output_filepath_suffix,
        clusterer=MiniBatchKMeans,
        clusterer_name="Mini-batch K-means clustering",
    )


def k_medoids_cluster_hyperparameter_search(
    param_grid: ParameterGrid,
    default_params: dict,
    word_embeddings: np.ndarray,
    word_embeddings_pairwise_dists: np.ndarray,
    output_dir: str,
    model_name: str,
    dataset_name: str,
    output_filepath_suffix: str,
) -> dict:
    """
    Searches for the best set of hyperparameters using K-medoids clustering
    and mean Silhouette Coefficient as the internal cluster metric.

    Parameters
    ----------
    param_grid : ParameterGrid
        Parameter grid to search through
    default_params : dict
        Default parameters to use for the clusterer
    word_embeddings : np.ndarray
        Word embeddings to perform clustering on
    word_embeddings_pairwise_dists : np.ndarray
        Numpy matrix containing pairwise distances between word embeddings
    output_dir : str
        Output directory
    model_name : str
        Name of the model
    dataset_name : str
        Name of the dataset the model was trained on
    output_filepath_suffix : str
        Output filepath suffix

    Returns
    -------
    result : dict
        Dictionary containing cluster labels and metric scores
    """
    return k_means_cluster_hyperparameter_search(
        param_grid=param_grid,
        default_params=default_params,
        word_embeddings=word_embeddings,
        word_embeddings_pairwise_dists=word_embeddings_pairwise_dists,
        output_dir=output_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        output_filepath_suffix=output_filepath_suffix,
        clusterer=KMedoids,
        clusterer_name="K-medoids clustering",
    )


def gmm_cluster_hyperparameter_search(
    param_grid: ParameterGrid,
    default_params: dict,
    word_embeddings: np.ndarray,
    word_embeddings_pairwise_dists: np.ndarray,
    output_dir: str,
    model_name: str,
    dataset_name: str,
    output_filepath_suffix: str,
) -> dict:
    """
    Searches for the best set of hyperparameters using Gaussian mixture
    models (GMM) clustering and mean Silhouette Coefficient as the internal
    cluster metric.

    Parameters
    ----------
    param_grid : ParameterGrid
        Parameter grid to search through
    default_params : dict
        Default parameters to use for the clusterer
    word_embeddings : np.ndarray
        Word embeddings to perform clustering on
    word_embeddings_pairwise_dists : np.ndarray
        Numpy matrix containing pairwise distances between word embeddings
    output_dir : str
        Output directory
    model_name : str
        Name of the model
    dataset_name : str
        Name of the dataset the model was trained on
    output_filepath_suffix : str
        Output filepath suffix

    Returns
    -------
    result : dict
        Dictionary containing cluster labels and metric scores
    """
    return k_means_cluster_hyperparameter_search(
        param_grid=param_grid,
        default_params=default_params,
        word_embeddings=word_embeddings,
        word_embeddings_pairwise_dists=word_embeddings_pairwise_dists,
        output_dir=output_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        output_filepath_suffix=output_filepath_suffix,
        clusterer=GaussianMixture,
        clusterer_name="GMM clustering",
    )


def agglomerative_clustering(
    word_embeddings_pairwise_dists: np.ndarray,
    linkages: list = None,
) -> dict:
    """
    Performs agglomerative clustering and creates linkage matrices for each linkage

    Parameters
    ----------
    word_embeddings_pairwise_dists : np.ndarray
        Numpy matrix containing pairwise distances between word embeddings
    linkages : list
        List of agglomerative linkages (defaults to all linkages)

    Returns
    -------
    agglomerative_clusterings : dict
        Result of agglomerative clustering
    """
    if linkages is None:
        linkages = ["complete", "average", "single"]
    agglomerative_clusterings = {}
    for linkage in linkages:

        # Fit clustering on pairwise distances between word embeddings
        clustering = AgglomerativeClustering(
            n_clusters=None, affinity="precomputed", linkage=linkage, distance_threshold=0
        )
        clustering.fit(word_embeddings_pairwise_dists)

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

        # Create required linkage matrix for fcluster function
        agglomerative_clustering_linkage_matrix = np.column_stack(
            [clustering.children_, clustering.distances_, counts]
        ).astype(np.float)

        # Set result in dict
        agglomerative_clusterings[linkage] = {
            "clustering": clustering,
            "linkage_matrix": agglomerative_clustering_linkage_matrix,
        }
    return agglomerative_clusterings


def agglomerative_cluster_hyperparameter_search(
    cluster_numbers: list,
    linkages: list,
    agglomerative_clusterings: dict,
    word_embeddings_pairwise_dists: np.ndarray,
    output_dir: str,
    model_name: str,
    dataset_name: str,
    output_filepath_suffix: str,
) -> dict:
    """
    Searches for the best set of hyperparameters using agglomerative
    clustering and mean Silhouette Coefficient as the internal cluster metric.

    Parameters
    ----------
    cluster_numbers : list
        List of cluster numbers to evaluate.
    linkages : list
        List of linkages to evaluate
    agglomerative_clusterings : dict
        Dictionary containing result from `agglomerative_clustering`
        function.
    word_embeddings_pairwise_dists : np.ndarray
        Numpy matrix containing pairwise distances between word embeddings
    output_dir : str
        Output directory
    model_name : str
        Name of the model
    dataset_name : str
        Name of the dataset the model was trained on
    output_filepath_suffix : str
        Output filepath suffix

    Returns
    -------
    result : dict
        Dictionary containing cluster labels and metric scores
    """
    # Ensure output directory exists
    makedirs(output_dir, exist_ok=True)

    # Perform clustering
    clustering_result = {}
    print(f"-- Fitting and predicting cluster labels for agglomerative clustering --")
    for linkage in linkages:
        print(f"Linkage: {linkage}")
        clustering_result[linkage] = {
            "metric_name": "Silhouette Coefficient",
            "cluster_labels": [],
            "metric_scores": [],
            "best_cluster_labels_idx": -1,
        }
        for k in tqdm(cluster_numbers):
            linkage_matrix = agglomerative_clusterings[linkage]["linkage_matrix"]
            cluster_labels_pred = fcluster(Z=linkage_matrix, criterion="maxclust", t=k)
            clustering_result[linkage]["cluster_labels"].append(cluster_labels_pred)

            # Compute Silhouette Coefficient score
            cluster_metric_value = silhouette_score(
                X=word_embeddings_pairwise_dists,
                labels=cluster_labels_pred,
                metric="precomputed",
            )
            clustering_result[linkage]["metric_scores"].append(cluster_metric_value)

        clustering_result[linkage]["best_cluster_labels_idx"] = np.argmax(
            clustering_result[linkage]["metric_scores"]
        )

    # Save result to output dir
    save_cluster_result_to_disk(
        clustering_result, output_dir, model_name, dataset_name, output_filepath_suffix
    )

    return clustering_result


def hdbscan_cluster_hyperparameter_search(
    param_grid: ParameterGrid,
    default_params: dict,
    word_embeddings: np.ndarray,
    output_dir: str,
    model_name: str,
    dataset_name: str,
    output_filepath_suffix: str,
) -> dict:
    """
    Searches for the best set of hyperparameters using HDBSCAN and
    Density-Based Clustering Validation (DBCV) as the internal
    cluster metric.


    Parameters
    ----------
    param_grid : ParameterGrid
        Parameter grid to search through
    default_params : dict
        Default parameters for HDBSCAN
    word_embeddings : np.ndarray
        Word embeddings to perform clustering on
    output_dir : str
        Output directory
    model_name : str
        Name of the model
    dataset_name : str
        Name of the dataset the model was trained on
    output_filepath_suffix : str
        Output filepath suffix

    Returns
    -------
    result : dict
        Dictionary containing cluster labels and DBCV scores
    """
    # Ensure output directory exists
    makedirs(output_dir, exist_ok=True)

    # Perform clustering
    hdbscan_cluster_labels = []
    hdbscan_dbcv_scores = []
    for params in tqdm(param_grid, desc="Performing clustering using HDBSCAN"):
        hdbscan_clustering = HDBSCAN(**params, **default_params)
        cluster_labels_pred = hdbscan_clustering.fit_predict(word_embeddings)
        hdbscan_cluster_labels.append(cluster_labels_pred)

        # Use already computed DBCV score to compare parameters
        dbcv_score = hdbscan_clustering.relative_validity_
        hdbscan_dbcv_scores.append(dbcv_score)

    # Create result as dict
    result = {
        "metric_name": "DBCV",
        "metric_scores": hdbscan_dbcv_scores,
        "cluster_labels": hdbscan_cluster_labels,
        "best_cluster_labels_idx": np.argmax(hdbscan_dbcv_scores),
    }

    # Save result to output dir
    save_cluster_result_to_disk(
        result, output_dir, model_name, dataset_name, output_filepath_suffix
    )

    return result


def plot_cluster_metric_scores(
    hyperparameters: list,
    scores: list,
    best_score_idx: int,
    metric_name: str,
    scatter: bool = True,
    ax: plt.axis = None,
) -> None:
    """
    Plots internal cluster validation metric scores

    Parameters
    ----------
    hyperparameters : list
        List of hyperparameters used to compute the scores
    scores : list
        List of scores computes using metric
    best_score_idx : int
        Best score index
    metric_name : str
        Name of the internal cluster validation metric
    scatter : bool
        Whether or not to scatter points (defaults to True)
    ax : plt.axis
        Matplotlib axis (defaults to None)
    """
    if ax is None:
        _, ax = plt.subplots()
    xs = range(len(hyperparameters))
    plt.plot(xs, scores)
    if scatter:
        plt.scatter(xs, scores)
        plt.scatter(xs[best_score_idx], scores[best_score_idx], c="r")
    plt.xticks(xs, hyperparameters, rotation=90)
    plt.xlabel("Hyperparameters")
    plt.ylabel(f"{metric_name} score")
    plt.tight_layout()
    plt.show()


def plot_cluster_sizes(cluster_labels: list, ax: plt.axis = None) -> np.ndarray:
    """
    Plots cluster sizes using a histogram and returns a list of most frequent
    cluster sizes.

    Parameters
    ----------
    cluster_labels : list
        List of cluster labels
    ax : plt.axis
        Matplotlib axis (default None)

    Returns
    -------
    most_common_cluster_sizes : np.ndarray
        Numpy array containing the most common cluster sizes
    """
    if ax is None:
        _, ax = plt.subplots()

    # Print cluster size ratio (max / min)
    labels_unique, labels_counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes, cluster_size_counts = np.unique(labels_counts, return_counts=True)

    num_clusters = len(labels_unique)
    max_cluster_size = max(labels_counts)
    min_cluster_size = min(labels_counts)
    cluster_size_ratio = max_cluster_size / min_cluster_size
    print(
        f"{num_clusters} clusters: max={max_cluster_size}, min={min_cluster_size}, ratio={cluster_size_ratio}"
    )

    # Plot distribution of cluster sizes
    sns.histplot(labels_counts, bins=max_cluster_size, ax=ax)
    ax.set_xlabel("Cluster size")
    ax.set_ylabel("Number of words in cluster")
    plt.show()

    # Sort cluster sizes by frequency
    most_common_cluster_sizes = cluster_sizes[np.argsort(cluster_size_counts)[::-1]]

    return most_common_cluster_sizes


def words_in_clusters(cluster_labels: np.ndarray, words: np.ndarray) -> tuple:
    """
    Gets words in clusters from a list of cluster labels.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Numpy array containing cluster labels
    words : np.ndarray
        Numpy array containing all words from vocabulary.

    Returns
    -------
    result : tuple
        Tuple containing list of cluster words and sizes, respectively.
    """
    labels_unique, labels_counts = np.unique(cluster_labels, return_counts=True)
    cluster_words = []
    for cluster_label in labels_unique:
        words_in_cluster = words[cluster_labels == cluster_label]
        cluster_words.append(words_in_cluster)
    cluster_words = np.array(cluster_words, dtype=object)
    return cluster_words, labels_counts
