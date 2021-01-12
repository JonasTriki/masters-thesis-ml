import numpy as np
from gudhi.persistence_graphical_tools import \
    plot_persistence_diagram as gd_plot_persistence_diagram
from gudhi.rips_complex import RipsComplex
from gudhi.wasserstein import wasserstein_distance
from matplotlib import pyplot as plt

from utils import pairwise_cosine_distances, words_to_vectors


def plot_persistence_diagram(
    pairwise_distances: np.ndarray, simplex_tree_max_dims: int = 2, show_plot: bool = True
) -> None:
    """
    Plots a persistence diagram using Vietoris-Rips complex.

    Parameters
    ----------
    pairwise_distances : np.ndarray
        Pairwise distances between vectors.
    simplex_tree_max_dims : int
        Maximal dimension to use when creating the simplex tree (defaults to 2).
    show_plot : bool
        Whether or not to call plt.show() (defaults to True).
    """
    # Build Vietoris-Rips complex
    skeleton_word2vec = RipsComplex(distance_matrix=pairwise_distances)

    # Plot persistence diagram
    simplex_tree = skeleton_word2vec.create_simplex_tree(
        max_dimension=simplex_tree_max_dims
    )
    barcodes = simplex_tree.persistence()
    gd_plot_persistence_diagram(barcodes)

    if show_plot:
        plt.show()


def punctured_neighbourhood(
    target_word: str,
    word_to_int: dict,
    word_embeddings: np.ndarray,
    word_embeddings_pairwise_dists: np.ndarray,
    neighbourhood_size: int,
) -> np.ndarray:
    """
    TODO: Docs
    """
    # Find neighbouring words (excluding the target word itself)
    target_word_idx = word_to_int[target_word]
    neighbourhood_distances = word_embeddings_pairwise_dists[target_word_idx]
    neighbourhood_sorted_indices = np.argsort(neighbourhood_distances)[
        1 : neighbourhood_size + 1
    ]
    neighbouring_word_embeddings = word_embeddings[neighbourhood_sorted_indices]
    return neighbouring_word_embeddings


def tps(
    target_word: str,
    word_embeddings: np.ndarray,
    words_vocabulary: list,
    word_to_int: dict,
    neighbourhood_size: int,
    sanity_check: bool = False,
) -> float:
    """
    Computes the topological polysemy (TPS) [1] of a word with respect
    to some word embeddings and neighbourhood size.

    Parameters
    ----------
    target_word : str
        Target word (w)
    word_embeddings : np.ndarray
        Word embeddings
    words_vocabulary : list
        List of either words (str) or word integer representations (int), signalizing
        what part of the vocabulary we want to use.
    word_to_int : dict of str and int
        Dictionary mapping from word to its integer representation.
    neighbourhood_size : int
        Neighbourhood size (n)
    sanity_check : bool, optional
        Whether or not to run sanity checks (defaults to False).

    Returns
    -------
    tps : float
        TPS of `target_word` w.r.t. word_embeddings and neighbourhood_size.

    References
    ----------
    .. [1] Alexander Jakubowski, Milica Gašić, & Marcus Zibrowius. (2020).
       Topology of Word Embeddings: Singularities Reflect Polysemy.
    """
    # Create word vectors from given words/vocabulary
    word_vectors = words_to_vectors(
        words_vocabulary=words_vocabulary,
        word_to_int=word_to_int,
        word_embeddings=word_embeddings,
    )

    # Compute pairwise distances between each word vector
    pairwise_word_vector_distances = pairwise_cosine_distances(word_vectors)

    # Normalize all word vectors to have L2-norm
    word_vectors_norm = word_vectors / np.linalg.norm(word_vectors)

    # Compute punctured neighbourhood
    target_word_punct_neigh = punctured_neighbourhood(
        target_word=target_word,
        word_to_int=word_to_int,
        word_embeddings=word_vectors_norm,
        word_embeddings_pairwise_dists=pairwise_word_vector_distances,
        neighbourhood_size=neighbourhood_size,
    )

    # Project word vectors in punctured neighbourhood to the unit sphere
    target_word_punct_neigh_sphere = np.zeros(target_word_punct_neigh.shape)
    target_word_vector_w = word_vectors_norm[word_to_int[target_word]]
    for i, v in enumerate(target_word_punct_neigh):
        w_v_diff = v - target_word_vector_w
        target_word_punct_neigh_sphere[i] = w_v_diff / np.linalg.norm(w_v_diff)

    # Compute the degree zero persistence diagram of target_word_punct_neigh_sphere
    target_degree = 0
    rips_complex = RipsComplex(points=target_word_punct_neigh_sphere)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=target_degree)
    barcodes = simplex_tree.persistence()
    if sanity_check:
        gd_plot_persistence_diagram(barcodes)

    zero_degree_diagram_points = np.array(
        [
            [birth, death]
            for dim, (birth, death) in barcodes
            if dim == target_degree and death != np.inf
        ]
    )
    empty_degree_diagram_points = np.zeros(zero_degree_diagram_points.shape)
    wasserstein_norm = wasserstein_distance(
        X=zero_degree_diagram_points, Y=empty_degree_diagram_points
    )

    return wasserstein_norm
