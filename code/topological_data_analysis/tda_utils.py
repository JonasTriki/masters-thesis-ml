import sys
from typing import Union

import annoy
import numpy as np
from gudhi.persistence_graphical_tools import \
    plot_persistence_diagram as gd_plot_persistence_diagram
from gudhi.rips_complex import RipsComplex
from gudhi.wasserstein import wasserstein_distance
from matplotlib import pyplot as plt

sys.path.append("..")

from utils import (cosine_vector_to_matrix_distance, pairwise_cosine_distances,
                   words_to_vectors)


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
    word_embeddings_norm: np.ndarray,
    neighbourhood_size: int,
    word_embeddings_pairwise_dists: np.ndarray,
    annoy_index: annoy.AnnoyIndex,
) -> np.ndarray:
    """
    Finds a punctured neighbourhood around a target word using
    cosine distances.

    Parameters
    ----------
    target_word : str
        Target word (w)
    word_to_int : dict of str and int
        Dictionary mapping from word to its integer representation.
    word_embeddings : np.ndarray
        Word embeddings
    word_embeddings_norm : np.ndarray
        Normalized word embeddings
    neighbourhood_size : int
        Neighbourhood size (n)
    word_embeddings_pairwise_dists : np.ndarray
        Pairwise distances between word embeddings
    annoy_index : annoy.AnnoyIndex
        Annoy index built on the word embeddings (defaults to None).
        If specified, the approximate nearest neighbour index is used to find
        punctured neighbourhoods.

    Returns
    -------
    neighbouring_word_embeddings : np.ndarray
        Neighbouring word embeddings of `target_word`, excluding
        the word itself
    """
    # Find neighbouring words (excluding the target word itself)
    target_word_int = word_to_int[target_word]
    if annoy_index is not None:
        neighbourhood_sorted_indices = annoy_index.get_nns_by_item(
            i=target_word_int, n=neighbourhood_size + 1
        )[1:]
    else:
        if word_embeddings_pairwise_dists is not None:
            neighbourhood_distances = word_embeddings_pairwise_dists[target_word_int]
        else:
            neighbourhood_distances = cosine_vector_to_matrix_distance(
                x=word_embeddings[target_word_int], y=word_embeddings
            )
        neighbourhood_sorted_indices = np.argsort(neighbourhood_distances)[
            1 : neighbourhood_size + 1
        ]
    neighbouring_word_embeddings = word_embeddings_norm[neighbourhood_sorted_indices]
    return neighbouring_word_embeddings


def tps(
    target_word: str,
    word_embeddings: np.ndarray,
    words_vocabulary: list,
    word_to_int: dict,
    neighbourhood_size: int,
    word_embeddings_normalized: np.ndarray = None,
    word_embeddings_pairwise_dists: np.ndarray = None,
    annoy_index: annoy.AnnoyIndex = None,
    sanity_check: bool = False,
    return_persistence_diagram: bool = False,
) -> Union[float, tuple]:
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
        what part of the vocabulary we want to use. Set to none to use whole vocabulary.
    word_to_int : dict of str and int
        Dictionary mapping from word to its integer representation.
    neighbourhood_size : int
        Neighbourhood size (n)
    word_embeddings_normalized : np.ndarray, optional
        Normalized word embeddings (defaults to None).
    word_embeddings_pairwise_dists : np.ndarray, optional
        Numpy matrix containing pairwise distances between word embeddings
        (defaults to None).
    annoy_index : annoy.AnnoyIndex, optional
        Annoy index built on the word embeddings (defaults to None).
        If specified, the approximate nearest neighbour index is used to find
        punctured neighbourhoods.
    sanity_check : bool, optional
        Whether or not to run sanity checks (defaults to False).
    return_persistence_diagram : bool, optional
        Whether or not to return persistence diagram (defaults to False).

    Returns
    -------
    result : float or tuple
        TPS of `target_word` w.r.t. word_embeddings and neighbourhood_size.
        If return_persistence_diagram is set to true, then a tuple is returned
        with the TPS as the first value and the zero degree persistence diagram
        as the second value.

    References
    ----------
    .. [1] Alexander Jakubowski, Milica Gašić, & Marcus Zibrowius. (2020).
       Topology of Word Embeddings: Singularities Reflect Polysemy.
    """
    # Create word vectors from given words/vocabulary
    if words_vocabulary is not None:
        word_vectors = words_to_vectors(
            words_vocabulary=words_vocabulary,
            word_to_int=word_to_int,
            word_embeddings=word_embeddings,
        )
    else:
        word_vectors = word_embeddings

    if word_embeddings_normalized is None:

        # Normalize all word vectors to have L2-norm
        word_embeddings_normalized = word_vectors / np.linalg.norm(
            word_vectors, axis=1
        ).reshape(-1, 1)

    # Compute punctured neighbourhood
    target_word_punct_neigh = punctured_neighbourhood(
        target_word=target_word,
        word_to_int=word_to_int,
        word_embeddings=word_vectors,
        word_embeddings_norm=word_embeddings_normalized,
        neighbourhood_size=neighbourhood_size,
        word_embeddings_pairwise_dists=word_embeddings_pairwise_dists,
        annoy_index=annoy_index,
    )

    # Project word vectors in punctured neighbourhood to the unit sphere
    target_word_punct_neigh_sphere = np.zeros(target_word_punct_neigh.shape)
    target_word_vector_w = word_embeddings_normalized[word_to_int[target_word]]
    for i, v in enumerate(target_word_punct_neigh):
        w_v_diff = v - target_word_vector_w
        target_word_punct_neigh_sphere[i] = w_v_diff / np.linalg.norm(w_v_diff)

    # Compute the degree zero persistence diagram of target_word_punct_neigh_sphere
    target_dim = 0
    rips_complex = RipsComplex(points=target_word_punct_neigh_sphere)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=target_dim)
    barcodes = simplex_tree.persistence()
    if sanity_check:
        gd_plot_persistence_diagram(barcodes)

    zero_degree_diagram_points = np.array(
        [
            [birth, death]
            for dim, (birth, death) in barcodes
            if dim == target_dim and death != np.inf
        ]
    )
    empty_degree_diagram_points = np.zeros(zero_degree_diagram_points.shape)
    wasserstein_norm = wasserstein_distance(
        X=zero_degree_diagram_points, Y=empty_degree_diagram_points
    )

    if return_persistence_diagram:
        return wasserstein_norm, barcodes
    else:
        return wasserstein_norm


def geometric_anomaly_detection(
    word_embeddings: np.ndarray,
    words_vocabulary: list,
    word_to_int: dict,
    r: float,
    s: float,
    word_embeddings_pairwise_dists: np.ndarray = None,
    annoy_index: annoy.AnnoyIndex = None,
    sanity_check: bool = False,
) -> dict:
    """
    Computes geometric anomaly detection Procedure 1 from [1].

    Parameters
    ----------
    word_embeddings : np.ndarray
        Word embeddings
    words_vocabulary : list, optional
        List of either words (str) or word integer representations (int), signalizing
        what part of the vocabulary we want to use. Set to None to use whole vocabulary.
    word_to_int : dict of str and int
        Dictionary mapping from word to its integer representation.
    r : float
        Lower pairwise distance parameter.
    s : float
        Upper pairwise distance parameter.
    word_embeddings_pairwise_dists : np.ndarray, optional
        Numpy matrix containing pairwise distances between word embeddings
    annoy_index : annoy.AnnoyIndex, optional
        Annoy index built on the word embeddings (defaults to None).
        If specified, the approximate nearest neighbour index is used to compute
        distance between two word vectors.
    sanity_check : bool, optional
        Whether or not to print sanity checks (defaults to False).

    Returns
    -------
    result : dict
        Result as a dict, containing three subsets P_man (k-manifold points),
        P_bnd (boundary points) and P_int (desired intersection points).

    References
    ----------
    .. [1] Bernadette J Stolz, Jared Tanner, Heather A Harrington, & Vidit Nanda.
       (2019). Geometric anomaly detection in data.
    """
    # Create word vectors from given words/vocabulary
    if words_vocabulary is not None:
        word_vectors = words_to_vectors(
            words_vocabulary=words_vocabulary,
            word_to_int=word_to_int,
            word_embeddings=word_embeddings,
        )
    else:
        word_vectors = word_embeddings
    n, k = word_vectors.shape

    # Create lambda function for computing distance between word vectors efficiently.
    if word_embeddings_pairwise_dists is not None:
        word_vector_distance = lambda word_i, word_j: word_embeddings_pairwise_dists[
            word_i, word_j
        ]
    elif annoy_index is not None:
        word_vector_distance = lambda word_i, word_j: annoy_index.get_distance(
            word_i, word_j
        )
    else:
        word_vector_distance = lambda word_i, word_j: np.linalg.norm(
            word_vectors[word_i] - word_vectors[word_j]
        )

    # Initialize result
    P_man = []
    P_bnd = []
    P_int = []

    target_barcode_dim = k - 1
    high_low_pairwise_distance_diff = s - r
    for i, y in enumerate(range(n)):

        # Find A_y ⊂ word_vectors containing all word vectors in word_vectors
        # which satisfy r ≤ ||x − y|| ≤ s.
        A_y_indices = np.array(
            [j for j in range(n) if r <= word_vector_distance(i, j) <= s]
        )
        if sanity_check:
            print(f"A_y_indices: {A_y_indices}")
        if len(A_y_indices) == 0:
            continue
        A_y = word_vectors[A_y_indices]

        # Compute (k-1) Vietoris-Rips barcode of A_y
        rips_complex = RipsComplex(points=A_y)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=target_barcode_dim)
        A_y_barcodes = simplex_tree.persistence()

        # Calculate number of intervals in A_y_barcodes of length > (s - r).
        N_y = 0
        for _, (birth, death) in A_y_barcodes:
            if death == np.inf:
                continue
            life_time = death - birth
            if life_time > high_low_pairwise_distance_diff:
                N_y += 1
        if sanity_check:
            print(f"N_y: {N_y}")

        # Add result
        if N_y == 0:
            P_bnd.append(i)
        elif N_y == 1:
            P_man.append(i)
        else:
            P_int.append(i)

    return {
        "P_man": P_man,
        "P_bnd": P_bnd,
        "P_int": P_int,
    }
