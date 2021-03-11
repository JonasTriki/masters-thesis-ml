import sys
from typing import Optional, Union

import annoy
import numpy as np
from fastdist import fastdist
from fastdist.fastdist import vector_to_matrix_distance
from gudhi.persistence_graphical_tools import (
    plot_persistence_diagram as gd_plot_persistence_diagram,
)
from gudhi.rips_complex import RipsComplex
from gudhi.wasserstein import wasserstein_distance

sys.path.append("..")

from approx_nn import ApproxNN  # noqa: E402
from utils import words_to_vectors  # noqa: E402


def punctured_neighbourhood(
    target_word: str,
    word_to_int: dict,
    word_embeddings_norm: np.ndarray,
    neighbourhood_size: int,
    word_embeddings_pairwise_dists: np.ndarray,
    ann_instance: ApproxNN,
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
    word_embeddings_norm : np.ndarray
        Normalized word embeddings
    neighbourhood_size : int
        Neighbourhood size (n)
    word_embeddings_pairwise_dists : np.ndarray
        Pairwise distances between word embeddings
    ann_instance : ApproxNN
        Approximate nearest neighbour (ANN) instance, built on the word embeddings
        If specified, the ANN index is used to find punctured neighbourhoods.

    Returns
    -------
    neighbouring_word_embeddings : np.ndarray
        Neighbouring word embeddings of `target_word`, excluding
        the word itself
    """
    # Find neighbouring words (excluding the target word itself)
    target_word_int = word_to_int[target_word]
    if ann_instance is not None:
        neighbourhood_sorted_indices = ann_instance.search(
            query_vector=word_embeddings_norm[target_word_int],
            k_neighbours=neighbourhood_size,
            excluded_neighbour_indices=[target_word_int],
        )
    else:
        if word_embeddings_pairwise_dists is not None:
            neighbourhood_distances = word_embeddings_pairwise_dists[target_word_int]
        else:
            neighbourhood_distances = vector_to_matrix_distance(
                u=word_embeddings_norm[target_word_int],
                m=word_embeddings_norm,
                metric=fastdist.euclidean,
                metric_name="euclidean",
            )
        neighbourhood_sorted_indices = np.argsort(neighbourhood_distances)[
            1 : neighbourhood_size + 1
        ]
    neighbouring_word_embeddings = word_embeddings_norm[neighbourhood_sorted_indices]
    return neighbouring_word_embeddings


def tps(
    target_word: str,
    word_to_int: dict,
    neighbourhood_size: int,
    words_vocabulary: Optional[list] = None,
    word_embeddings: np.ndarray = None,
    word_embeddings_normalized: np.ndarray = None,
    word_embeddings_pairwise_dists: np.ndarray = None,
    ann_instance: ApproxNN = None,
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
    word_to_int : dict of str and int
        Dictionary mapping from word to its integer representation.
    neighbourhood_size : int
        Neighbourhood size (n)
    words_vocabulary : list, optional
        List of either words (str) or r word integer representations (int), signalizing
        what part of the vocabulary we want to use (defaults to None, i.e., whole vocabulary).
    word_embeddings : np.ndarray
        Word embeddings; either word_embeddings or word_embeddings_normalized
        must be specified (defaults to None).
    word_embeddings_normalized : np.ndarray, optional
        Normalized word embeddings; either word_embeddings_normalized or word_embeddings
        must be specified (defaults to None).
    word_embeddings_pairwise_dists : np.ndarray, optional
        Numpy matrix containing pairwise distances between word embeddings
        (defaults to None).
    ann_instance : ApproxNN, optional
        Approximate nearest neighbour (ANN) instance, built on the word embeddings
        (defaults to None). If specified, the ANN index is used to find punctured
        neighbourhoods.
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
    if word_embeddings is not None and word_embeddings_normalized is None:
        if words_vocabulary is not None:
            word_vectors = words_to_vectors(
                words_vocabulary=words_vocabulary,
                word_to_int=word_to_int,
                word_embeddings=word_embeddings,
            )
        else:
            word_vectors = word_embeddings

        # Normalize all word vectors to have L2-norm
        word_embeddings_normalized = word_vectors / np.linalg.norm(
            word_vectors, axis=1
        ).reshape(-1, 1)
    elif word_embeddings is None and word_embeddings_normalized is None:
        raise ValueError(
            "Either word embeddings or normalized word embeddings must be specifed."
        )

    # Compute punctured neighbourhood
    target_word_punct_neigh = punctured_neighbourhood(
        target_word=target_word,
        word_to_int=word_to_int,
        word_embeddings_norm=word_embeddings_normalized,
        neighbourhood_size=neighbourhood_size,
        word_embeddings_pairwise_dists=word_embeddings_pairwise_dists,
        ann_instance=ann_instance,
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


def tps_point_cloud(
    point_index: int,
    neighbourhood_size: int,
    point_cloud: np.ndarray = None,
    point_cloud_indices: Optional[list] = None,
    point_cloud_normalized: np.ndarray = None,
    point_cloud_pairwise_dists: np.ndarray = None,
    annoy_index: annoy.AnnoyIndex = None,
    sanity_check: bool = False,
    return_persistence_diagram: bool = False,
) -> Union[float, tuple]:
    """
    Computes the topological polysemy (TPS) [1] of a point with respect
    to some point cloud and neighbourhood size.

    Parameters
    ----------
    point_index : int
        Index of target point
    neighbourhood_size : int
        Neighbourhood size (n)
    point_cloud : np.ndarray, optional
        Point cloud. Either point_cloud or point_cloud_normalized must be specified.
        (Defaults to None).
    point_cloud_indices : list, optional
        List of indices of point cloud to use (defaults to None, i.e., all points).
    point_cloud_normalized : np.ndarray, optional
        Normalized point cloud. Either point_cloud or point_cloud_normalized must be specified.
        (Defaults to None).
    point_cloud_pairwise_dists : np.ndarray, optional
        Pairwise distances between points in point cloud (defaults to None).
    annoy_index : annoy.AnnoyIndex, optional
        Annoy index built on the point cloud (defaults to None).
        If specified, the approximate nearest neighbour index is used to find
        punctured neighbourhoods.
    sanity_check : bool, optional
        Whether or not to run sanity checks (defaults to False).
    return_persistence_diagram : bool, optional
        Whether or not to return persistence diagram (defaults to False).

    Returns
    -------
    result : float or tuple
        TPS of `point_index` w.r.t. point_cloud and neighbourhood_size.
        If return_persistence_diagram is set to true, then a tuple is returned
        with the TPS as the first value and the zero degree persistence diagram
        as the second value.

    References
    ----------
    .. [1] Alexander Jakubowski, Milica Gašić, & Marcus Zibrowius. (2020).
       Topology of Word Embeddings: Singularities Reflect Polysemy.
    """
    if point_cloud is not None:
        num_points = len(point_cloud)
    elif point_cloud_normalized is not None:
        num_points = len(point_cloud_normalized)
    else:
        raise ValueError(
            "Either point_cloud or point_cloud_normalized must be specified."
        )
    return tps(
        target_word=str(point_index),
        word_to_int={str(i): i for i in range(num_points)},
        neighbourhood_size=neighbourhood_size,
        words_vocabulary=point_cloud_indices,
        word_embeddings=point_cloud,
        word_embeddings_normalized=point_cloud_normalized,
        word_embeddings_pairwise_dists=point_cloud_pairwise_dists,
        annoy_index=annoy_index,
        sanity_check=sanity_check,
        return_persistence_diagram=return_persistence_diagram,
    )
