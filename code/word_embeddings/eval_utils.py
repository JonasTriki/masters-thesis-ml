from pprint import pprint
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.manifold import TSNE
from text_preprocessing_utils import preprocess_text
from tqdm.auto import tqdm
from umap import UMAP
from word2vec import Word2vec


def similar_words_vec(
    target_word_vec: np.ndarray,
    weights: np.ndarray,
    words: np.ndarray,
    top_n: int = 10,
    skip_first: int = 0,
) -> list:
    """
    Finds the `top_n` words closest to a word vector.

    Parameters
    ----------
    target_word_vec : np.ndarray
        Target word vector to find closest words to.
    weights : np.ndarray
        Numpy matrix (vocabulary size, embedding dim) containing word vectors.
    words : np.ndarray
        Numpy array containing words from the vocabulary.
    top_n : int, optional
        Number of similar words (defaults to 10).
    skip_first : int, optional
        Number of similar words to skip (defaults to 0).

    Returns
    -------
    pairs : list of tuples of str and int
        List of `top_n` similar words and their cosine similarities.
    """
    word_vec_weights_dotted = target_word_vec @ weights.T
    word_vec_weights_norm = np.linalg.norm(target_word_vec) * np.linalg.norm(
        weights, axis=1
    )
    cos_sims = word_vec_weights_dotted / word_vec_weights_norm
    cos_sims = np.clip(cos_sims, 0, 1)
    sorted_indices = cos_sims.argsort()[::-1]
    # print(words.shape, weights.shape)
    top_words = words[sorted_indices][skip_first : skip_first + top_n]
    top_sims = cos_sims[sorted_indices][skip_first : skip_first + top_n]

    # Create word similarity pairs
    pairs = list(zip(top_words, top_sims))

    return pairs


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
    target_word: str,
    weights: np.ndarray,
    word_to_int: Dict[str, int],
    words: np.ndarray,
    top_n: int = 10,
) -> list:
    """
    Finds the most similar words of a given word

    Parameters
    ----------
    target_word : str
        Target word to find word vector of.
    weights : np.ndarray
        Numpy matrix (vocabulary size, embedding dim) containing word vectors.
    word_to_int : dict of str and int
        Dictionary mapping from word to its integer representation.
    words : np.ndarray
        Numpy array containing words from the vocabulary.
    top_n : int, optional
        Number of similar words (defaults to 10).

    Returns
    -------
    pairs : list of tuples of str and int
        List of `top_n` similar words and their cosine similarities.
    """
    # Get word vector of word
    word_vec = get_word_vec(target_word, word_to_int, weights)

    return similar_words_vec(word_vec, weights, words, top_n, skip_first=1)


def create_embeddings_of_train_checkpoints(
    model_checkpoint_filepaths: list,
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
    model_checkpoint_filepaths : list
        List of model checkpoints to load models from.
    vocab_size : int
        Vocabulary size of models
    embedding_dim : int
        Embedding dimension used in models
    clusterer : ClusterMixin
        Clusterer instance to use when labeling words. Applies clustering
        to the last set of checkpoints and uses them.
    transformer : Union[UMAP, TSNE, TransformerMixin]
        Transformer/dimensionality reduction instance applied to the embeddings

    Returns
    -------
    results : tuple of np.ndarray
        A tuple consisting of the transformed embeddings and its respective
        clustering labels.
    """

    # Prepare matrix of word embeddings of all epochs
    # epochs_lst = np.arange(1, epochs + 1)
    # model_checkpoint_paths = [
    #    f"{checkpoints_dir}/word2vec-model-epoch-{epoch_nr}.model"
    #    for epoch_nr in epochs_lst
    # ]
    num_checkpoints = len(model_checkpoint_filepaths)
    embeddings = np.zeros((num_checkpoints * vocab_size, embedding_dim))
    cluster_labels = np.zeros(vocab_size)
    for i, checkpoint_path in enumerate(model_checkpoint_filepaths):

        # Load model and get weights
        word2vec = Word2vec()
        word2vec.load_model(checkpoint_path)
        weights = word2vec.embedding_weights

        embeddings[i * vocab_size : (i + 1) * vocab_size] = weights

        # Use cluster labels from last embedding
        if i == num_checkpoints - 1:
            # KMeans(n_clusters=10)
            cluster_labels = clusterer.fit_predict(weights)

    # Create transformed embedding from all checkpoints
    transformed_embedding = transformer.fit_transform(embeddings)
    # umap_embedding_all_epochs = umap.UMAP(
    #    n_components=3, random_state=rng_seed
    # ).fit_transform(embeddings)

    return transformed_embedding, cluster_labels


def visualize_embeddings_over_time(
    transformed_word_embeddings: np.ndarray,
    cluster_labels: np.ndarray,
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
    words : np.ndarray
        Words in the vocabulary in a numpy array.
    num_words : int, optional
        Number of words to visualize (defaults to all words).
    title : str, optional
        Title to use for the plot (defaults to "Word embeddings over time").
    """
    is_3d = transformed_word_embeddings.shape[1] == 3
    vocab_size = len(words)
    num_epochs = int(len(transformed_word_embeddings) / vocab_size)
    if num_words is None:
        num_words = vocab_size

    # Create Pandas DataFrame for Plotly animations
    word_embeddings_over_time_df_dict: dict = {
        "epoch": [],
        "x": [],
        "y": [],
        "cluster_label": [],
        "word": [],
    }
    if is_3d:
        word_embeddings_over_time_df_dict["z"] = []

    for epoch_num in range(1, num_epochs + 1):
        weights = transformed_word_embeddings[
            (epoch_num - 1) * vocab_size : epoch_num * vocab_size
        ]

        # Add to df
        word_embeddings_over_time_df_dict["epoch"].extend(np.repeat(epoch_num, num_words))
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
            animation_frame="epoch",
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
            animation_frame="epoch",
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
    title: str = "Plot of word vectors",
    x_label: str = "x1",
    y_label: str = "x2",
    z_label: str = "x3",
):
    """
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


def evaluate_model_questions_words(
    questions_words: dict,
    word_embeddings: np.ndarray,
    word_to_int: dict,
    words: np.ndarray,
    top_n: int = 5,
    verbose: int = 1,
) -> dict:
    """
    Evaluates a word2vec mode using top-n accuracy on questions-words from Mikolov et. al.

    Parameters
    ----------
    questions_words : dict
        Dictionary mapping from section to a list of word pairs
    word_embeddings : np.ndarray
        Word embeddings in a numpy matrix
    word_to_int : dict mapping from str to int
        Dictionary for mapping a word to its integer representation
    words : np.ndarray
        Numpy array of words from the vocabulary.
    top_n : int, optional
        Number of words to look at for computing accuracy. If the predicted word is in the
        `top_n` most similar words, it is flatted as a correct prediction. Defaults to
        5.
    verbose : int, optional
        Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose). Defaults to 1 (verbose).

    Returns
    -------
    question_words_accuracies : dict mapping from str to float
        Dictionary mapping from questions-word section to its accuracy (percentage).
    """
    num_sections = len(questions_words.keys())
    question_words_accuracies: Dict[str, float] = {}
    for i, (section_name, question_word_pairs) in zip(
        range(num_sections), questions_words.items()
    ):
        if verbose >= 1:
            print(f"-- Evaluating {section_name}... --")
        num_correct = 0
        total = len(question_word_pairs)
        for word_pairs in tqdm(question_word_pairs):

            # Clean words (same as training) before evaluation
            word_pairs_clean = [preprocess_text(word)[0] for word in word_pairs]

            # Ensure all words are in vocabulary
            words_in_vocab = True
            for word in word_pairs_clean:
                if word not in word_to_int:
                    words_in_vocab = False
                    break
            if not words_in_vocab:
                total = total - 1
                continue

            # Evaluate prediction
            a_word, b_word, c_word, d_word = word_pairs_clean
            a_vec = get_word_vec(a_word, word_to_int, word_embeddings)
            b_vec = get_word_vec(b_word, word_to_int, word_embeddings)
            c_vec = get_word_vec(c_word, word_to_int, word_embeddings)

            d_word_predictions = similar_words_vec(
                a_vec - b_vec + c_vec, word_embeddings, words, top_n=top_n
            )
            d_word_predictions = [word for word, _ in d_word_predictions]
            if d_word in d_word_predictions:
                num_correct = num_correct + 1
            # else:
            #    if verbose == 1:
            #    print(
            #       f"Incorrect prediction: {a_word} is to {b_word} as {c_word} is to {d_word} (predicted: {d_word_predictions[0]})"
            #   )

        if total == 0:
            question_words_accuracies[section_name] = -1  # Meaning no predictions made
            print("All questions words missing from vocabulary")
        else:
            question_words_accuracies[section_name] = num_correct / total
            print(f"Accuracy: {(question_words_accuracies[section_name] * 100):.2f}%")

    return question_words_accuracies
