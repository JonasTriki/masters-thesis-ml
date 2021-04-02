import re
import sys
from os.path import join
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, TextArea

sys.path.append("..")

from text_preprocessing_utils import preprocess_text  # noqa: E402
from utils import words_to_vectors  # noqa: E402


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
    remove_brackets_re = re.compile(r"^(.+?)[(\[].*?[)\]](.*?)$")
    name_no_brackets_results = re.findall(remove_brackets_re, name)
    if len(name_no_brackets_results) > 0:
        name = "".join(name_no_brackets_results[0]).strip()
    name = "_".join(preprocess_text(name.replace("'", "")))
    return name


def transform_word_embeddings(
    embedders: list,
    word_embeddings: np.ndarray,
    words_vocabulary: list,
    word_to_int: dict,
) -> dict:
    """
    Transforms word embeddings using dimensionality reduction techniques
    such as PCA or UMAP.

    Parameters
    ----------
    embedders : list
        List of embedder instances (e.g. PCA, UMAP)
    word_embeddings : np.ndarray
        Word embeddings
    words_vocabulary : list
        List of either words (str) or word integer representations (int), signalizing
        what part of the vocabulary we want to use.
    word_to_int : dict of str and int
        Dictionary mapping from word to its integer representation.

    Returns
    -------
    transformed_embeddings_result : dict
        Dictionary with transformed word embeddings
    """
    # Create word vectors from given words/vocabulary
    word_vectors = words_to_vectors(
        words_vocabulary=words_vocabulary,
        word_to_int=word_to_int,
        word_embeddings=word_embeddings,
    )

    # Create embeddings
    transformed_embeddings_result = {}
    for embedder_name, embedder_instance in embedders:
        transformed_embeddings_result[embedder_name] = embedder_instance.fit_transform(
            word_vectors
        )
    return transformed_embeddings_result


def plot_cluster_metric_scores(
    metric_scores: list,
    hyperparameters: list,
    best_score_idx: int,
    metric_name: str,
    scatter: bool = True,
    set_xticks: bool = True,
    set_xtickslabels: bool = True,
    xtickslabels_rotation: int = 90,
    ax: plt.axis = None,
    xlabel: str = "Hyperparameters",
    xrange: range = None,
    show_plot: bool = True,
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
    set_xtickslabels : bool
        Whether or not to set the labels on the x-axis
    xtickslabels_rotation : int
        Sets the xticks labels rotation (defaults to 90), set_xtickslabels
        must be set to True to have an effect.
    ax : plt.axis
        Matplotlib axis (defaults to None)
    xlabel : str
        X-axis label (defaults to "Hyperparameters")
    xrange : range
        Range to use for the x-axis (default starts from 0 to)
    show_plot : bool
        Whether or not to call plt.show() (defaults to True)
    """
    if ax is None:
        _, ax = plt.subplots()
    if xrange is None:
        xrange = range(len(hyperparameters))
    ax.plot(xrange, metric_scores)
    if scatter:
        ax.scatter(xrange, metric_scores)
    ax.scatter(xrange[best_score_idx], metric_scores[best_score_idx], c="r", s=72)
    if set_xticks:
        ax.set_xticks(xrange)
    if set_xtickslabels:
        ax.set_xticklabels(hyperparameters, rotation=xtickslabels_rotation, ha="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{metric_name} score")
    if show_plot:
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
    foods_filepath = join(data_dir, "foods.txt")
    vegan_foods_filepath = join(data_dir, "vegan_foods_categorized.csv")

    # Filter words out of vocabulary
    word_in_vocab_filter: Callable[[str], bool] = lambda word: word in word_to_int

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
    forenames_male_mask = forenames_df["gender"] == "M"
    forenames_female_mask = forenames_df["gender"] == "F"
    forenames_male = forenames_df[forenames_male_mask]["name"].values
    forenames_female = forenames_df[forenames_female_mask]["name"].values
    forenames_intersection = np.intersect1d(forenames_male, forenames_female)
    forenames_data = {
        "male": forenames_male,
        "female": forenames_female,
        "intersection": forenames_intersection,
        "all": forenames,
    }
    surnames = surnames_df["name"].values

    # Load numbers
    with open(numbers_filepath, "r") as numbers_file:
        numbers = numbers_file.read().split("\n")
    numbers = np.array([num for num in numbers if word_in_vocab_filter(num)])

    # Load video games
    video_games_df = pd.read_csv(video_games_filepath)
    video_games_df = video_games_df[video_games_df["Name"].apply(word_in_vocab_filter)]
    video_games = video_games_df["Name"].values
    video_games_genres = video_games_df["Genre"].values
    video_games_data = {"all": video_games}
    for video_game_genre in video_games_genres:
        video_games_data[video_game_genre] = video_games_df[
            video_games_df["Genre"] == video_game_genre
        ]["Name"].values

    # Load food data
    with open(foods_filepath, "r") as foods_file:
        foods = foods_file.read().split("\n")
    foods = np.array(
        [food_word for food_word in foods if word_in_vocab_filter(food_word)]
    )

    # Load categorized vegan food data
    vegan_foods_df = pd.read_csv(vegan_foods_filepath)
    vegan_foods_df = vegan_foods_df[vegan_foods_df["Name"].apply(word_in_vocab_filter)]
    vegan_food_categories = vegan_foods_df["Category"].values
    vegan_foods_data = {"all": vegan_foods_df["Name"].values}
    for vegan_food_category in vegan_food_categories:
        vegan_foods_data[vegan_food_category] = vegan_foods_df[
            vegan_foods_df["Category"] == vegan_food_category
        ]["Name"].values

    # Combine data into dictionary
    data = {
        "countries": countries,
        "country_capitals": country_capitals,
        "forenames": forenames_data,
        "surnames": surnames,
        "numbers": numbers,
        "video_games": video_games_data,
        "foods": foods,
        "vegan_foods": vegan_foods_data,
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
                    name="Non group words",
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


def word_group_visualization(
    transformed_word_embeddings: np.ndarray,
    words: np.ndarray,
    word_groups: dict,
    xlabel: str,
    ylabel: str,
    emphasis_words: list = None,
    alpha: float = 1,
    non_group_words_color: str = "#c44e52",
    show_plot: bool = True,
) -> None:
    """
    Visualizes one or more word groups by plotting its word embeddings in 2D.

    Parameters
    ----------
    transformed_word_embeddings : np.ndarray
        Transformed word embeddings.
    words : np.ndarray
        Numpy array containing all words from vocabulary.
    word_groups : dict
        Dictionary containing word groups to visualize.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    emphasis_words : list, optional
        List representing words to emphasize in the visualization (defaults to None).
        Entries can be either be strings (words) or tuples, consisting of the word, x-offset
        and y-offset.
    alpha : float
        Scatter plot alpha value (defaults to 1).
    non_group_words_color : str
        Color for words outside groups (defaults to #c44e52).
    show_plot : bool
        Whether or not to call plt.show() (defaults to True).
    """
    # Filter and restrict words in word groups
    word_group_words_restricted = {}
    for group_key, group_data in word_groups.items():
        group_words = group_data["words"]
        group_words = np.array([word for word in group_words if word in words])
        group_words_indices = np.array(
            [np.where(words == word)[0][0] for word in group_words]
        )
        group_word_embeddings = transformed_word_embeddings[group_words_indices]
        boundaries = group_data.get("boundaries", {})
        if boundaries.get("xmin") is None:
            boundaries["xmin"] = group_word_embeddings[:, 0].min()
        if boundaries.get("xmax") is None:
            boundaries["xmax"] = group_word_embeddings[:, 0].max()
        if boundaries.get("ymin") is None:
            boundaries["ymin"] = group_word_embeddings[:, 1].min()
        if boundaries.get("ymax") is None:
            boundaries["ymax"] = group_word_embeddings[:, 1].max()

        group_word_embeddings_boundaries_mask = [
            (boundaries["xmin"] <= word_vec[0] <= boundaries["xmax"])
            and (boundaries["ymin"] <= word_vec[1] <= boundaries["ymax"])
            for i, word_vec in enumerate(group_word_embeddings)
        ]
        word_group_words_restricted[group_key] = group_words[
            group_word_embeddings_boundaries_mask
        ]

    # Find words not in groups
    words_not_in_groups_mask = [
        i
        for i, word in enumerate(words)
        for group_words in word_group_words_restricted.values()
        if word not in group_words
    ]

    _, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Plot non-group words
    ax.scatter(
        x=transformed_word_embeddings[words_not_in_groups_mask][:, 0],
        y=transformed_word_embeddings[words_not_in_groups_mask][:, 1],
        s=10,
        alpha=alpha,
        c=non_group_words_color,
    )

    # Plot group words
    for group_key, group_words in word_group_words_restricted.items():
        group_words_indices = np.array(
            [np.where(words == word)[0][0] for word in group_words]
        )
        group_word_embeddings = transformed_word_embeddings[group_words_indices]

        ax.scatter(
            x=group_word_embeddings[:, 0],
            y=group_word_embeddings[:, 1],
            s=15,
            alpha=alpha,
            c=word_groups[group_key]["color"],
            label=word_groups[group_key]["label"],
        )

    # Visualize emphasized words
    if emphasis_words is not None:
        emphasis_words = [
            (entry, 0, 0) if type(entry) == str else entry for entry in emphasis_words
        ]
        for emphasis_word, x_offset, y_offset in emphasis_words:
            word_group_key = None
            for group_key, group_data in word_groups.items():
                if emphasis_word in group_data["words"]:
                    word_group_key = group_key
                    break
            if word_group_key is None:
                word_color = non_group_words_color
            else:
                word_color = word_groups[group_key]["color"]

            word_idx = [i for i, word in enumerate(words) if word == emphasis_word][0]
            ax.scatter(
                x=transformed_word_embeddings[word_idx, 0],
                y=transformed_word_embeddings[word_idx, 1],
                s=40,
                alpha=alpha,
                c=word_color,
            )

            # Annotate emphasis word with a text box
            offsetbox = TextArea(emphasis_word)
            ab = AnnotationBbox(
                offsetbox,
                tuple(transformed_word_embeddings[word_idx]),
                xybox=(x_offset, 40 + y_offset),
                xycoords="data",
                boxcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black", linewidth=2),
            )
            ax.add_artist(ab)

    plt.legend()
    if show_plot:
        plt.show()
