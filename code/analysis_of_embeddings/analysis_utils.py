import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering


def create_linkage_matrix(clustering: AgglomerativeClustering) -> list:
    """
    TODO: Docs

    Code from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html,
    downloaded 26th of October, 2020.
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


def plot_silhouette_scores(cluster_numbers: list, silhouette_scores: list) -> None:
    """
    TODO: Docs
    """
    xs = range(len(cluster_numbers))
    plt.plot(xs, silhouette_scores)
    plt.scatter(xs, silhouette_scores)
    plt.xticks(xs, cluster_numbers, rotation=90)
    plt.xlabel("Number of clusters")
    plt.ylabel("Average silhouette score")
    plt.show()


def words_in_clusters(cluster_labels: np.ndarray, words: np.ndarray) -> tuple:
    """
    TODO: Docs
    """
    labels_unique, labels_counts = np.unique(cluster_labels, return_counts=True)
    cluster_words = []
    for cluster_label in labels_unique:
        words_in_cluster = words[cluster_labels == cluster_label]
        cluster_words.append(words_in_cluster)
    cluster_words = np.array(cluster_words)
    return cluster_words, labels_counts
