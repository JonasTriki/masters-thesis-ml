import pickle
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fastdist import fastdist
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
from umap import UMAP


def get_word_vec(
    target_word: str, word_to_int: Dict[str, int], weights: np.ndarray
) -> np.ndarray:
    """
    Gets the word vector of a word.

    Parameters
    ----------
    target_word : str
        Target word to find word vector of.
    word_to_int : dict of str and int
        Dictionary mapping from word to its integer representation.
    weights : np.ndarray
        Numpy matrix (vocabulary size, embedding dim) containing word vectors.

    Returns
    -------
    word_vec : np.ndarray
        Word vector of a word
    """
    return weights[word_to_int[target_word]]


def similar_words(
    weights: np.ndarray,
    word_to_int: Dict[str, int],
    words: np.ndarray,
    top_n: int = 10,
    positive_words: List[str] = None,
    negative_words: List[str] = None,
    vocab_size: int = -1,
    return_similarity_score: bool = True,
) -> List[Union[Tuple, str]]:
    """
    Finds the most similar words of a linear combination of positively and negatively
    contributing words.

    Parameters
    ----------
    weights : np.ndarray
        Numpy matrix (vocabulary size, embedding dim) containing word vectors.
    word_to_int : dict of str and int
        Dictionary mapping from word to its integer representation.
    words : np.ndarray
        Numpy array containing words from the vocabulary.
    top_n : int, optional
        Number of similar words (defaults to 10).
    positive_words : list of str, optional
        List of words contribution positively (defaults to empty list).
    negative_words : list of str, optional
        List of words contribution negatively (defaults to empty list).
    vocab_size : int, optional
        Vocabulary size to use, e.g., only most common `vocab_size` words to taken
        into account (defaults to -1 meaning all words).
    return_similarity_score : bool, optional
        Whether or not to return the cosine similarity score.

    Returns
    -------
    If return_similarity_score is True, then
        pairs : list of tuples of str and int
            List of `top_n` similar words and their cosine similarities.
    else:
        closest_words : list of str
            List of `top_n` similar words.
    """
    # Default values
    if positive_words is None:
        positive_words = []
    if negative_words is None:
        negative_words = []

    # Restrict vocabulary
    if vocab_size > 0:
        weights = weights[:vocab_size]
        words = words[:vocab_size]

    # Create query word vector
    query_word_vec = np.zeros((weights.shape[1],), dtype=np.float64)
    query_word_vec += np.array(
        [get_word_vec(pos_word, word_to_int, weights) for pos_word in positive_words]
    ).sum(axis=0)
    query_word_vec -= np.array(
        [get_word_vec(neg_word, word_to_int, weights) for neg_word in negative_words]
    ).sum(axis=0)

    # Create indices list of query words to exclude from search
    exclude_words_indices = [
        word_to_int[word] for word in positive_words + negative_words
    ]

    # Compute cosine similarity
    cos_sims = fastdist.cosine_vector_to_matrix(query_word_vec, weights)
    sorted_indices = cos_sims.argsort()[::-1]
    sorted_indices = [idx for idx in sorted_indices if idx not in exclude_words_indices]
    top_words = words[sorted_indices][:top_n]
    top_sims = cos_sims[sorted_indices][:top_n]

    # Create word similarity pairs
    if return_similarity_score:
        result = list(zip(top_words, top_sims))
    else:
        result = top_words

    return result


def create_embeddings_of_train_weight_checkpoints(
    model_weights_filepaths: list,
    vocab_size: int,
    embedding_dim: int,
    clusterer: ClusterMixin,
    transformer: Union[UMAP, TSNE, TransformerMixin],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates embeddings using a transformer over the course of multiple word2vec training
    checkpoints.

    Parameters
    ----------
    model_weights_filepaths : list
        List filepaths to model weights.
    vocab_size : int
        Vocabulary size to use.
    embedding_dim : int
        Embedding dimension used in models.
    clusterer : ClusterMixin
        Clusterer instance to use when labeling words. Applies clustering
        to the last set of checkpoints and uses them.
    transformer : Union[UMAP, TSNE, TransformerMixin]
        Transformer/dimensionality reduction instance applied to the embeddings.

    Returns
    -------
    results : tuple of np.ndarray
        A tuple consisting of the transformed embeddings and its respective
        clustering labels.
    """
    # Prepare matrix of word embeddings of all time steps
    num_checkpoints = len(model_weights_filepaths)
    embeddings = np.zeros((num_checkpoints * vocab_size, embedding_dim))
    cluster_labels = np.zeros(vocab_size)
    for i, model_weights_filepath in enumerate(model_weights_filepaths):

        # Load weights and restrict to vocabulary
        weights = np.load(model_weights_filepath, mmap_mode="r")
        weights = weights[:vocab_size]

        embeddings[i * vocab_size : (i + 1) * vocab_size] = weights

        # Use cluster labels from last embedding
        if i == num_checkpoints - 1:
            cluster_labels = clusterer.fit_predict(weights)

    # Create transformed embedding from all checkpoints
    transformed_embedding = transformer.fit_transform(embeddings)

    return transformed_embedding, cluster_labels


def visualize_embeddings_over_time(
    transformed_word_embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    vocab_size: int,
    words: np.ndarray,
    num_words: Optional[int] = None,
    title: str = "Word embeddings over time",
) -> None:
    """
    Visualizes transformed (e.g. into 2D or 3D) word embeddings over time.

    Parameters
    ----------
    transformed_word_embeddings : np.ndarray
        Transformed word embeddings.
    cluster_labels : np.ndarray
        Cluster labels of each word in the transformed word embeddings.
    vocab_size : int
        Vocabulary size of transformed word embeddings.
    words : np.ndarray
        Words in the vocabulary in a numpy array.
    num_words : int, optional
        Number of words to visualize (defaults to all words).
    title : str, optional
        Title to use for the plot (defaults to "Word embeddings over time").
    """
    is_3d = transformed_word_embeddings.shape[1] == 3
    num_time_steps = int(len(transformed_word_embeddings) / vocab_size)
    if num_words is None:
        num_words = vocab_size

    # Create Pandas DataFrame for Plotly animations
    word_embeddings_over_time_df_dict: dict = {
        "time_step": [],
        "x": [],
        "y": [],
        "cluster_label": [],
        "word": [],
    }
    if is_3d:
        word_embeddings_over_time_df_dict["z"] = []

    for time_step in range(1, num_time_steps + 1):
        weights = transformed_word_embeddings[
            (time_step - 1) * vocab_size : time_step * vocab_size
        ]

        # Add to df
        word_embeddings_over_time_df_dict["time_step"].extend(
            np.repeat(time_step, num_words)
        )
        word_embeddings_over_time_df_dict["x"].extend(weights[:num_words, 0])
        word_embeddings_over_time_df_dict["y"].extend(weights[:num_words, 1])
        if is_3d:
            word_embeddings_over_time_df_dict["z"].extend(weights[:num_words, 2])
        word_embeddings_over_time_df_dict["cluster_label"].extend(
            cluster_labels[:num_words]
        )
        word_embeddings_over_time_df_dict["word"].extend(words[:num_words])

    # Create df from dict
    word_embeddings_over_time_df = pd.DataFrame(word_embeddings_over_time_df_dict)

    # Visualize animation of transformed embeddings over time
    if is_3d:
        fig = px.scatter_3d(
            word_embeddings_over_time_df,
            x="x",
            y="y",
            z="z",
            range_x=[
                transformed_word_embeddings[:num_words, 0].min(),
                transformed_word_embeddings[:num_words, 0].max(),
            ],
            range_y=[
                transformed_word_embeddings[:num_words, 1].min(),
                transformed_word_embeddings[:num_words, 1].max(),
            ],
            range_z=[
                transformed_word_embeddings[:num_words, 2].min(),
                transformed_word_embeddings[:num_words, 2].max(),
            ],
            animation_frame="time_step",
            color="cluster_label",
            hover_name="word",
            title=title,
        )
    else:
        fig = px.scatter(
            word_embeddings_over_time_df,
            x="x",
            y="y",
            range_x=[
                transformed_word_embeddings[:num_words, 0].min(),
                transformed_word_embeddings[:num_words, 0].max(),
            ],
            range_y=[
                transformed_word_embeddings[:num_words, 1].min(),
                transformed_word_embeddings[:num_words, 1].max(),
            ],
            animation_frame="time_step",
            color="cluster_label",
            hover_name="word",
            title=title,
        )
    fig.update_scenes({"aspectmode": "cube"})
    fig.show()


def plot_word_relationships_2d(
    relationship_pairs: List[Tuple[str, str]],
    transformed_word_embeddings: np.ndarray,
    word_to_int: dict,
    title: str = "Plot of relationship pairs",
    x_label: str = "x1",
    y_label: str = "x2",
) -> None:
    """
    Plots relationships between words in 2D. Requires that the transformed word embeddings
    are transformed in 2D space.

    Parameters
    ----------
    relationship_pairs : list of tuples of str
        List of tuples of "from" (first entry) and "to" (second entry) words.
    transformed_word_embeddings : np.ndarray
        Transformed word embeddings.
    word_to_int : dict of str and int
        Dictionary mapping from word to its integer representation.
    title : str
        Title to use for the plot.
    x_label : str
        Label to use for the x-axis.
    y_label : str
        Label to use for the y-axis.
    """
    fig = go.Figure()
    for (from_word, to_word) in relationship_pairs:
        from_word_vec = get_word_vec(from_word, word_to_int, transformed_word_embeddings)
        to_word_vec = get_word_vec(to_word, word_to_int, transformed_word_embeddings)

        # Plot points in 2D
        fig.add_trace(
            go.Scatter(
                x=[from_word_vec[0], to_word_vec[0]],
                y=[from_word_vec[1], to_word_vec[1]],
                mode="markers+text",
                text=[from_word, to_word],
                textposition="bottom center",
                hovertext=[from_word, to_word],
            )
        )

        # Add title, x-label and y-label to plot
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)

        # Annotate points with arrows
        fig.add_annotation(
            ax=from_word_vec[0],
            ay=from_word_vec[1],
            axref="x",
            ayref="y",
            x=to_word_vec[0],
            y=to_word_vec[1],
            xref="x",
            yref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            opacity=0.5,
        )
    fig.update_layout(showlegend=False)
    fig.show()


def plot_word_vectors(
    words_to_plot: list,
    transformed_word_embeddings: np.ndarray,
    word_to_int: dict,
    word_colors: np.ndarray = None,
    title: str = "Plot of word vectors",
    x_label: str = "x1",
    y_label: str = "x2",
    z_label: str = "x3",
):
    """
    TODO: Docs
    Plot word vectors.
    """
    is_3d = transformed_word_embeddings.shape[1] == 3

    fig = go.Figure()
    word_vectors = np.array(
        [
            get_word_vec(word, word_to_int, transformed_word_embeddings)
            for word in words_to_plot
        ]
    )

    # Plot points
    if is_3d:
        fig.add_trace(
            go.Scatter3d(
                x=word_vectors[:, 0],
                y=word_vectors[:, 1],
                z=word_vectors[:, 2],
                mode="markers+text",
                hovertext=words_to_plot,
                text=words_to_plot,
                textposition="bottom center",
                marker=dict(color=word_colors),
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=word_vectors[:, 0],
                y=word_vectors[:, 1],
                mode="markers+text",
                hovertext=words_to_plot,
                text=words_to_plot,
                textposition="bottom center",
                marker=dict(color=word_colors),
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title=x_label),
            yaxis=dict(title=y_label),
            zaxis=dict(title=z_label) if is_3d else None,
        ),
        title=title,
    )

    fig.update_layout(showlegend=False)
    fig.show()


def filter_word_analogy_dataset(
    analogy_dataset: List[Tuple[str, ...]], word_to_int: dict, vocab_size: int
) -> list:
    """
    Filters a word analogy dataset such that it only contains words from the vocabulary.

    Parameters
    ----------
    analogy_dataset : list
        List of word analogies (list of tuple of words)
    word_to_int : dict
        Dictionary for mapping a word to its integer representation.
    vocab_size : int
        Vocabulary size.

    Returns
    -------
    analogies_filtered : list
        Filtered word analogies.
    """
    analogies_filtered = []
    for word_analogies in analogy_dataset:

        # Ensure all words are in vocabulary
        words_in_vocab = True
        for word in word_analogies:
            if word not in word_to_int or word_to_int[word] >= vocab_size:
                words_in_vocab = False
                break
        if words_in_vocab:
            analogies_filtered.append(word_analogies)
    return analogies_filtered


def load_analogies_test_dataset(
    analogies_filepath: str, word_to_int: dict, vocab_size: int
) -> dict:
    """
    Loads an analogies test dataset file and filters out out of vocabulary entries.

    Parameters
    ----------
    analogies_filepath : str
        Filepath of the analogies test dataset file.
    word_to_int : dict
        Dictionary for mapping a word to its integer representation.
    vocab_size : int
        Vocabulary size.

    Returns
    -------
    analogies_dict : dict
        Dictionary mapping from section name to list of tuples of word analogies
        from the word vocabulary.
    """
    # Load analogies dict from file
    with open(analogies_filepath, "rb") as file:
        analogies_dict_raw = pickle.load(file)

    # Initialize resulting dictionary
    analogies_dict = {key: [] for key in analogies_dict_raw.keys()}

    # Ensure analogies_dict only contain entries that are in the vocabulary.
    for section_name, analogies_pairs in analogies_dict_raw.items():
        analogies_dict[section_name] = filter_word_analogy_dataset(
            analogies_pairs, word_to_int, vocab_size
        )

    return analogies_dict


def evaluate_model_word_analogies(
    analogies_filepath: str,
    word_embeddings_filepath: str,
    word_to_int: dict,
    words: np.ndarray,
    vocab_size: int = -1,
    top_n: int = 1,
    verbose: int = 1,
) -> dict:
    """
    Evaluates a word2vec model on a word analogies test dataset.

    Parameters
    ----------
    analogies_filepath : str
        Filepath of the analogies test dataset file.
    word_embeddings_filepath : str
        Filepath of the word embeddings.
    word_to_int : dict mapping from str to int
        Dictionary for mapping a word to its integer representation
    words : np.ndarray
        Numpy array of words from the vocabulary.
    vocab_size : int, optional
        Vocabulary size to use (defaults to -1 meaning all words).
    top_n : int, optional
        Number of words to look at for computing accuracy. If the predicted word is in the
        `top_n` most similar words, it is flagged as a correct prediction. Defaults to
        1.
    verbose : int, optional
        Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose). Defaults to 1 (verbose).

    Returns
    -------
    analogies_accuracies : dict mapping from str to float
        Dictionary mapping from analogy section to its accuracy (percentage).
    """
    if vocab_size == -1:
        vocab_size = len(words)

    # Load analogies word pairs from file
    analogies = load_analogies_test_dataset(analogies_filepath, word_to_int, vocab_size)

    # Load embeddings
    word_embeddings = np.load(word_embeddings_filepath, mmap_mode="r").astype(np.float64)

    # Perform evaluation
    analogies_accuracies = {}
    for (section_name, analogies_word_pairs) in analogies.items():
        if verbose >= 1:
            print(f"-- Evaluating {section_name}... --")
        num_correct = 0
        total = len(analogies_word_pairs)
        for qw_pair in tqdm(analogies_word_pairs):
            (a_word, b_word, c_word, d_word) = qw_pair

            d_word_predictions = similar_words(
                positive_words=[b_word, c_word],
                negative_words=[a_word],
                weights=word_embeddings,
                words=words,
                word_to_int=word_to_int,
                vocab_size=vocab_size,
                top_n=top_n,
                return_similarity_score=False,
            )
            if d_word in d_word_predictions:
                num_correct += 1

        if total == 0:
            analogies_accuracies[section_name] = np.nan  # => no predictions made
            print(f"All word analogies in {section_name} missing from vocabulary")
        else:
            analogies_accuracies[section_name] = num_correct / total
            print(f"Accuracy: {(analogies_accuracies[section_name] * 100):.2f}%")

    # Compute average accuracy over all sections (ignore NaN's)
    analogies_accuracies["avg"] = np.nanmean(list(analogies_accuracies.values()))

    return analogies_accuracies
