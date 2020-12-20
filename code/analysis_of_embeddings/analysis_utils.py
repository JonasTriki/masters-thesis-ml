import re
import sys
from os import makedirs
from os.path import join
from typing import Callable, Union

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from cdbw import CDbw
from hdbscan import HDBSCAN
from matplotlib import pyplot as plt
from s_dbw import S_Dbw
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from sklearn_extra.cluster import KMedoids
from tqdm.auto import tqdm

sys.path.append("..")

from text_preprocessing_utils import preprocess_text


def preprocess_name(name: str) -> str:
    """
    Preprocesses names by replacing brackets and combining words into
    a single word separated by underscore.

    Parameters
    ----------
    name : str
        Name to process

    Returns
    -------
    processed_name : str
        Processed name
    """
    remove_brackets_re = re.compile("^(.+?)[(\[].*?[)\]](.*?)$")
    name_no_brackets_results = re.findall(remove_brackets_re, name)
    if len(name_no_brackets_results) > 0:
        name = "".join(name_no_brackets_results[0]).strip()
    name = "_".join(preprocess_text(name.replace("'", "")))
    return name


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
    and various internal cluster metrics:
    - Silhouette Coefficient
    - S_Dbw validity index
    - CDbw validity index

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
    cluster_metrics = {
        "silhouette_coeff": {
            "name": "Silhouette Coefficient",
            "scores": [],
            "best_score_idx": -1,
        },
        "s_dbw": {
            "name": "S_Dbw validity index",
            "scores": [],
            "best_score_idx": -1,
        },
        "cdbw": {
            "name": "CDbw validity index",
            "scores": [],
            "best_score_idx": -1,
        },
    }
    for params in tqdm(param_grid, desc=f"Performing clustering using {clusterer_name}"):
        cls = clusterer(**params, **default_params)
        cluster_labels_pred = cls.fit_predict(word_embeddings)
        cluster_labels.append(cluster_labels_pred)

        # Compute metric scores
        silhouette_coeff_score = silhouette_score(
            X=word_embeddings_pairwise_dists,
            labels=cluster_labels_pred,
            metric="precomputed",
        )
        s_dbw_score = S_Dbw(
            X=word_embeddings, labels=cluster_labels_pred, metric="cosine"
        )
        cdbw_score = CDbw(
            X=word_embeddings, labels=cluster_labels_pred, metric="cosine", s=3
        )

        # Append metric scores
        cluster_metrics["silhouette_coeff"]["scores"].append(silhouette_coeff_score)
        cluster_metrics["s_dbw"]["scores"].append(s_dbw_score)
        cluster_metrics["cdbw"]["scores"].append(cdbw_score)

    # Find set score index for each metric
    cluster_metrics["silhouette_coeff"]["best_score_idx"] = np.argmax(
        cluster_metrics["silhouette_coeff"]["scores"]
    )
    cluster_metrics["s_dbw"]["best_score_idx"] = np.argmin(
        cluster_metrics["s_dbw"]["scores"]
    )
    cluster_metrics["cdbw"]["best_score_idx"] = np.argmax(
        cluster_metrics["cdbw"]["scores"]
    )

    result = {
        "cluster_labels": cluster_labels,
        "metrics": cluster_metrics,
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
    clustering and various internal cluster metrics:
    - Silhouette Coefficient
    - S_Dbw validity index
    - CDbw validity index

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
    and various internal cluster metrics:
    - Silhouette Coefficient
    - S_Dbw validity index
    - CDbw validity index

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
    models (GMM) clustering and various internal cluster metrics:
    - Silhouette Coefficient
    - S_Dbw validity index
    - CDbw validity index

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

        # Create required linkage matrix for fcluster function
        agglomerative_clustering_linkage_matrix = create_linkage_matrix(clustering)

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
    word_embeddings: np.ndarray,
    word_embeddings_pairwise_dists: np.ndarray,
    output_dir: str,
    model_name: str,
    dataset_name: str,
    output_filepath_suffix: str,
) -> dict:
    """
    Searches for the best set of hyperparameters using agglomerative
    clustering and various internal cluster metrics:
    - Silhouette Coefficient
    - S_Dbw validity index
    - CDbw validity index

    Parameters
    ----------
    cluster_numbers : list
        List of cluster numbers to evaluate.
    linkages : list
        List of linkages to evaluate
    agglomerative_clusterings : dict
        Dictionary containing result from `agglomerative_clustering`
        function.
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
    # Ensure output directory exists
    makedirs(output_dir, exist_ok=True)

    # Perform clustering
    clustering_result = {}
    print(f"-- Fitting and predicting cluster labels for agglomerative clustering --")
    for linkage in linkages:
        print(f"Linkage: {linkage}")

        cluster_labels = []
        cluster_metrics = {
            "silhouette_coeff": {
                "name": "Silhouette Coefficient",
                "scores": [],
                "best_score_idx": -1,
            },
            "s_dbw": {
                "name": "S_Dbw validity index",
                "scores": [],
                "best_score_idx": -1,
            },
            "cdbw": {
                "name": "CDbw validity index",
                "scores": [],
                "best_score_idx": -1,
            },
        }

        for k in tqdm(cluster_numbers):
            linkage_matrix = agglomerative_clusterings[linkage]["linkage_matrix"]
            cluster_labels_pred = fcluster(Z=linkage_matrix, criterion="maxclust", t=k)
            cluster_labels.append(cluster_labels_pred)

            # Compute metric scores
            silhouette_coeff_score = silhouette_score(
                X=word_embeddings_pairwise_dists,
                labels=cluster_labels_pred,
                metric="precomputed",
            )
            s_dbw_score = S_Dbw(
                X=word_embeddings, labels=cluster_labels_pred, metric="cosine"
            )
            cdbw_score = CDbw(
                X=word_embeddings, labels=cluster_labels_pred, metric="cosine", s=3
            )

            # Append metric scores
            cluster_metrics["silhouette_coeff"]["scores"].append(silhouette_coeff_score)
            cluster_metrics["s_dbw"]["scores"].append(s_dbw_score)
            cluster_metrics["cdbw"]["scores"].append(cdbw_score)

        # Find set score index for each metric
        cluster_metrics["silhouette_coeff"]["best_score_idx"] = np.argmax(
            cluster_metrics["silhouette_coeff"]["scores"]
        )
        cluster_metrics["s_dbw"]["best_score_idx"] = np.argmin(
            cluster_metrics["s_dbw"]["scores"]
        )
        cluster_metrics["cdbw"]["best_score_idx"] = np.argmax(
            cluster_metrics["cdbw"]["scores"]
        )
        clustering_result[linkage] = {
            "cluster_labels": cluster_labels,
            "cluster_metrics": cluster_metrics,
        }

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
    Searches for the best set of hyperparameters using HDBSCAN
    and various internal cluster metrics:
    - Density-Based Clustering Validation (DBCV)
    - S_Dbw validity index
    - CDbw validity index

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
    cluster_labels = []
    cluster_metrics = {
        "dbcv_relative": {
            "name": "Density-Based Clustering Validation (relative)",
            "scores": [],
            "best_score_idx": -1,
        },
        "s_dbw": {
            "name": "S_Dbw validity index",
            "scores": [],
            "best_score_idx": -1,
        },
        "cdbw": {
            "name": "CDbw validity index",
            "scores": [],
            "best_score_idx": -1,
        },
    }
    for params in tqdm(param_grid, desc="Performing clustering using HDBSCAN"):
        hdbscan_clustering = HDBSCAN(**params, **default_params)
        cluster_labels_pred = hdbscan_clustering.fit_predict(word_embeddings)
        cluster_labels.append(cluster_labels_pred)

        # Compute metric scores (DBCV already computed)
        dbcv_score = hdbscan_clustering.relative_validity_
        s_dbw_score = S_Dbw(
            X=word_embeddings, labels=cluster_labels_pred, metric="cosine"
        )
        cdbw_score = CDbw(
            X=word_embeddings, labels=cluster_labels_pred, metric="cosine", s=3
        )

        # Append metric scores
        cluster_metrics["dbcv_relative"]["scores"].append(dbcv_score)
        cluster_metrics["s_dbw"]["scores"].append(s_dbw_score)
        cluster_metrics["cdbw"]["scores"].append(cdbw_score)

    # Find set score index for each metric
    cluster_metrics["dbcv_relative"]["best_score_idx"] = np.argmax(
        cluster_metrics["dbcv_relative"]["scores"]
    )
    cluster_metrics["s_dbw"]["best_score_idx"] = np.argmin(
        cluster_metrics["s_dbw"]["scores"]
    )
    cluster_metrics["cdbw"]["best_score_idx"] = np.argmax(
        cluster_metrics["cdbw"]["scores"]
    )

    # Create result as dict
    result = {
        "cluster_labels": cluster_labels,
        "metrics": cluster_metrics,
    }

    # Save result to output dir
    save_cluster_result_to_disk(
        result, output_dir, model_name, dataset_name, output_filepath_suffix
    )

    return result


def plot_cluster_metric_scores(
    metric_scores: list,
    hyperparameters: list,
    best_score_idx: int,
    metric_name: str,
    scatter: bool = True,
    set_xticks: bool = True,
    ax: plt.axis = None,
    xlabel: str = "Hyperparameters",
    xrange: range = None,
) -> None:
    """
    Plots internal cluster validation metric scores

    Parameters
    ----------
    metric_scores : list
        List of scores computed using metric
    hyperparameters : list
        List of hyperparameters used to compute the scores
    best_score_idx : int
        Best score index
    metric_name : str
        Name of the internal cluster validation metric
    scatter : bool
        Whether or not to scatter points (defaults to True)
    set_xticks : bool
        Whether or not to set the ticks on the x-axis
    ax : plt.axis
        Matplotlib axis (defaults to None)
    xlabel : str
        X-axis label (defaults to "Hyperparameters")
    xrange : range
        Range to use for the x-axis (default starts from 0 and )
    """
    if ax is None:
        _, ax = plt.subplots()
    if xrange is None:
        xrange = range(len(hyperparameters))
    ax.plot(xrange, metric_scores)
    if scatter:
        plt.scatter(xrange, metric_scores)
    ax.scatter(xrange[best_score_idx], metric_scores[best_score_idx], c="r", s=72)
    if set_xticks:
        ax.set_xticks(xrange)
        ax.set_xticklabels(hyperparameters, rotation=90, ha="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{metric_name} score")
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


def inspect_word_clusters(
    cluster_labels: np.ndarray,
    words: np.ndarray,
    min_cluster_size: int,
    most_common_cluster_sizes: np.ndarray,
    num_words_in_clusters_print: int = 10,
) -> None:
    """
    Inspects words in clusters:
    - `num_words_in_clusters_print` largest/smallest clusters
    - Words from clusters whose cluster number is the most common

    Parameters
    ----------
    cluster_labels : np.ndarray
        Cluster labels to inspect.
    words : np.ndarray
        Words in vocabulary.
    min_cluster_size : int
        Minimum cluster size to investigate.
    most_common_cluster_sizes : np.ndarray
        Cluster sizes sorted by most common to least common
    num_words_in_clusters_print : int
        Number of words to print of each cluster (defaults to 10).
    """
    # Look at the words corresponding to the different clusters (biggest, smallest, etc.)
    cluster_words, cluster_sizes = words_in_clusters(
        cluster_labels=cluster_labels, words=words
    )

    # Only inspect clusters with at least `min_cluster_size` words in them
    filter_min_cluster_size_mask = cluster_sizes >= min_cluster_size
    cluster_sizes_filtered = cluster_sizes[filter_min_cluster_size_mask]
    cluster_words_filtered = cluster_words[filter_min_cluster_size_mask]

    # Print `num_words_in_clusters_print` largest/smallest clusters
    sorted_cluster_indices = np.argsort(cluster_sizes_filtered)[::-1]

    print(f"-- {num_words_in_clusters_print} largest clusters --")
    for i in range(num_words_in_clusters_print):
        print(cluster_words_filtered[sorted_cluster_indices[i]])
    print()

    print(f"-- {num_words_in_clusters_print} smallest clusters --")
    for i in range(1, num_words_in_clusters_print + 1):
        print(cluster_words_filtered[sorted_cluster_indices[-i]])
    print()

    # Inspect words from clusters whose cluster numbers is the most common
    most_common_cluster_size = most_common_cluster_sizes[0]
    print(
        f"-- {num_words_in_clusters_print} random words from clusters whose cluster number is the most common (i.e. clusters of {most_common_cluster_size} words) -- "
    )
    most_common_cluster_words = cluster_words[cluster_sizes == most_common_cluster_size]

    # Print random words
    rng_indices = np.random.choice(
        np.arange(len(most_common_cluster_words)),
        size=num_words_in_clusters_print,
        replace=False,
    )
    most_common_cluster_words_random = most_common_cluster_words[rng_indices]
    for cluster_words in most_common_cluster_words_random:
        print(cluster_words)


def load_word_cluster_group_words(data_dir: str, word_to_int: dict) -> dict:
    """
    Load word groups for clustering.

    Parameters
    ----------
    data_dir : str
        Data directory
    word_to_int : dict of str and int
        Dictionary mapping from word to its integer representation.

    Returns
    -------
    data : dict
        Word group data in dictionary
    """
    # Constants
    country_info_filepath = join(data_dir, "country-info.csv")
    forenames_filepath = join(data_dir, "forenames.csv")
    surnames_filepath = join(data_dir, "surnames.csv")
    numbers_filepath = join(data_dir, "numbers.txt")
    video_games_filepath = join(data_dir, "video_games.csv")

    # Filter words out of vocabulary
    word_in_vocab_filter = lambda word: word in word_to_int

    # Load country info
    country_info_df = pd.read_csv(country_info_filepath)
    country_info_df_vocab_mask = (
        country_info_df[["name", "capital"]].isin(word_to_int.keys()).apply(all, axis=1)
    )
    country_info_df = country_info_df[country_info_df_vocab_mask]
    countries = country_info_df["name"].values
    country_capitals = country_info_df["capital"].values

    # Load names
    forenames_df = pd.read_csv(forenames_filepath)
    forenames_df = forenames_df[forenames_df["name"].apply(word_in_vocab_filter)]
    surnames_df = pd.read_csv(surnames_filepath)
    surnames_df = surnames_df[surnames_df["name"].apply(word_in_vocab_filter)]
    forenames = forenames_df["name"].values
    forenames_male = forenames_df[forenames_df["gender"] == "M"]["name"].values
    forenames_female = forenames_df[forenames_df["gender"] == "F"]["name"].values
    surnames = surnames_df["name"].values

    # Load numbers
    with open(numbers_filepath, "r") as file:
        numbers = file.read().split("\n")
    numbers = np.array([num for num in numbers if word_in_vocab_filter(num)])

    # Load video games
    video_games_df = pd.read_csv(video_games_filepath)
    video_games_df = video_games_df[video_games_df["Name"].apply(word_in_vocab_filter)]
    video_games = video_games_df["Name"].values

    # Combine data into dictionary
    data = {
        "countries": countries,
        "country_capitals": country_capitals,
        "forenames": forenames,
        "forenames_male": forenames_male,
        "forenames_female": forenames_female,
        "surnames": surnames,
        "numbers": numbers,
        "video_games": video_games,
    }

    return data


def visualize_word_cluster_groups(
    transformed_word_embeddings: np.ndarray,
    words: np.ndarray,
    word_groups: dict,
    visualize_non_group_words: bool,
    xlabel: str,
    ylabel: str,
    non_group_words_color: str = "#c44e52",
    ax: plt.axis = None,
    show_plot: bool = True,
    alpha: float = 1,
    interactive: bool = False,
) -> None:
    """
    Visualizes word cluster groups.

    Parameters
    ----------
    transformed_word_embeddings : np.ndarray
        Transformed word embeddings.
    words : np.ndarray
        Numpy array containing all words from vocabulary.
    word_groups : dict
        Dictionary containing word groups to visualize.
    visualize_non_group_words : bool
        Whether or not to visualize words outside word groups
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    non_group_words_color : str
        Color for words outside groups (defaults to #c44e52)
    ax : plt.axis
        Matplotlib axis (defaults to None)
    show_plot : bool
        Whether or not to call plt.show() (defaults to True)
    alpha : float
        Scatter plot alpha value (defaults to 1)
    interactive : bool
        Whether or not to make the visualization interactive
        using Plotly (defaults to False).
    """
    if ax is None and not interactive:
        _, ax = plt.subplots(figsize=(12, 7))
    if interactive:
        fig = go.Figure(layout=dict(xaxis=dict(title=xlabel), yaxis=dict(title=ylabel)))

    if visualize_non_group_words:

        # Create boolean mask for words outside groups
        words_in_groups = []
        for group_name in word_groups.keys():
            words_in_groups.extend(word_groups[group_name]["words"])
        words_not_in_groups_mask = [word not in words_in_groups for word in words]
        words_not_in_groups_sorted = [
            word for word in words if word not in words_in_groups
        ]

        # Plot words outside word group
        if interactive:
            fig.add_trace(
                go.Scatter(
                    x=transformed_word_embeddings[words_not_in_groups_mask][:, 0],
                    y=transformed_word_embeddings[words_not_in_groups_mask][:, 1],
                    mode="markers",
                    marker=dict(color=non_group_words_color),
                    hovertext=words_not_in_groups_sorted,
                    hoverinfo="x+y+text",
                    name=f"Non group words",
                    opacity=alpha,
                )
            )
        else:
            ax.scatter(
                x=transformed_word_embeddings[words_not_in_groups_mask][:, 0],
                y=transformed_word_embeddings[words_not_in_groups_mask][:, 1],
                c=non_group_words_color,
                alpha=alpha,
            )

    # Visualize words in groups
    for group_name, word_group in word_groups.items():
        words_in_group = word_group["words"]
        words_in_group_mask = [word in words_in_group for word in words]
        words_in_group_sorted = [word for word in words if word in words_in_group]
        word_group_color = word_group["color"]

        # Plot words inside word group
        if interactive:
            fig.add_trace(
                go.Scatter(
                    x=transformed_word_embeddings[words_in_group_mask][:, 0],
                    y=transformed_word_embeddings[words_in_group_mask][:, 1],
                    mode="markers",
                    marker=dict(color=word_group_color),
                    hovertext=words_in_group_sorted,
                    hoverinfo="x+y+text",
                    name=f"Words in {group_name}",
                    opacity=alpha,
                )
            )
        else:
            ax.scatter(
                x=transformed_word_embeddings[words_in_group_mask][:, 0],
                y=transformed_word_embeddings[words_in_group_mask][:, 1],
                c=word_group_color,
                alpha=alpha,
            )

    if interactive:
        fig.show()
    else:
        ax_legends = ["Non group words"]
        ax_legends.extend(
            [f"Words which are {group_name}" for group_name in word_groups.keys()]
        )
        ax.legend(ax_legends)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if show_plot:
            plt.show()
