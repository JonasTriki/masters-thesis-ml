from os.path import join

import joblib
import numpy as np
import seaborn as sns
from hdbscan import HDBSCAN
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm


def k_means_cluster_hyperparameter_search():
    # TODO: Implement this
    pass


def k_means_mini_batch_cluster_hyperparameter_search():
    # TODO: Implement this
    pass


def k_medoids_cluster_hyperparameter_search():
    # TODO: Implement this
    pass


def gmm_cluster_hyperparameter_search():
    # TODO: Implement this
    pass


def agglomerative_cluster_hyperparameter_search():
    # TODO: Implement this
    pass


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
    Searches for the best set of hyperparameters when using the
    HDBSCAN clustering algorithm.

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
    # Perform clustering
    hdbscan_cluster_labels = []
    hdbscan_dbcv_scores = []
    for param_grid in tqdm(param_grid, desc="Performing HDBSCAN clustering"):
        hdbscan_clustering = HDBSCAN(**param_grid, **default_params)
        cluster_labels_pred = hdbscan_clustering.fit_predict(word_embeddings)
        hdbscan_cluster_labels.append(cluster_labels_pred)

        # Use already computed DBCV score to compare parameters
        dbcv_score = hdbscan_clustering.relative_validity_
        hdbscan_dbcv_scores.append(dbcv_score)

    # Create result as dict
    result = {
        "cluster_labels": hdbscan_cluster_labels,
        "dbcv_scores": hdbscan_dbcv_scores,
        "best_labels_idx": np.argmax(hdbscan_dbcv_scores),
    }

    # Save to output dir
    output_path = join(
        output_dir, f"{model_name}-{dataset_name}-{output_filepath_suffix}.joblib"
    )
    joblib.dump(result, output_path)

    return result


def create_linkage_matrix(clustering: AgglomerativeClustering) -> list:
    """
    Creates a linkage matrix for agglomerative clustering.

    Code from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html,
    downloaded 26th of October, 2020.

    Parameters
    ----------
    clustering : AgglomerativeClustering
        Agglomerative clustering instance

    Returns
    -------
    linkage_matrix : list
        Linkage matrix as a list
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

    linkage_matrix = np.column_stack(
        [clustering.children_, clustering.distances_, counts]
    ).astype(np.float)

    return linkage_matrix


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
