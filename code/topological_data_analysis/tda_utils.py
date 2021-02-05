import sys
from typing import Union

import annoy
import numpy as np
from fastdist import fastdist
from fastdist.fastdist import vector_to_matrix_distance
from gudhi.persistence_graphical_tools import (
    plot_persistence_diagram as gd_plot_persistence_diagram,
)
from gudhi.rips_complex import RipsComplex
from gudhi.wasserstein import wasserstein_distance
from matplotlib import pyplot as plt

sys.path.append("..")

from utils import cosine_vector_to_matrix_distance, words_to_vectors


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
            neighbourhood_distances = vector_to_matrix_distance(
                u=word_embeddings_norm[target_word_int],
                m=word_embeddings_norm,
                metric=fastdist.euclidean,
            )
        neighbourhood_sorted_indices = np.argsort(neighbourhood_distances)[
            1 : neighbourhood_size + 1
        ]
    neighbouring_word_embeddings = word_embeddings_norm[neighbourhood_sorted_indices]
    return neighbouring_word_embeddings


def tps(
    target_word: str,
    word_embeddings: np.ndarray,
    word_to_int: dict,
    neighbourhood_size: int,
    words_vocabulary: list = None,
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
    word_to_int : dict of str and int
        Dictionary mapping from word to its integer representation.
    neighbourhood_size : int
        Neighbourhood size (n)
    words_vocabulary : list, optional
        List of either words (str) or r word integer representations (int), signalizing
        what part of the vocabulary we want to use (defaults to None, i.e., whole vocabulary).
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


def generate_points_in_spheres(
    num_points: int,
    sphere_dimensionality: int,
    sphere_means: tuple,
    space_dimensionality: int = None,
    create_intersection_point: bool = False,
    random_state: int = 0,
) -> tuple:
    """
    Generates points laying in two d-dimensional spheres. Spheres can be overlapping
    by setting sphere_means accordingly (e.g. [0, 0.75]).

    Parameters
    ----------
    num_points : int
        Number of points to generate per sphere.
    sphere_dimensionality : int
        Dimensionality (d) to use when generating points in d-dimensional spheres.
    sphere_means : tuple
        Tuple containing two floats indicating the mean of each sphere.
    space_dimensionality : int, optional
        Dimensionality to use for the point space (must be equal or greater than
        sphere_dimensionality). Can be used to increase the dimensionality for the points.
        Defaults to None (or sphere_dimensionality).
    create_intersection_point : bool, optional
        Whether or not to add intersection point between spheres (defaults to False).
    random_state : int, optional
        Random state to use when generating points (defaults to 0).

    Returns
    -------
    result : tuple
        Tuple containing randomly generated sphere points and which sphere the point
        corresponds to (2 indicates intersection between spheres).
    """
    # Set random seed
    np.random.seed(random_state)

    # Generate points in spheres
    sphere_means_in_space_dim = [
        np.repeat(mean, sphere_dimensionality) for mean in sphere_means
    ]
    total_num_points = 2 * num_points
    if create_intersection_point:
        total_num_points += 1
    if space_dimensionality is not None:
        sphere_means_in_space_dim = [
            np.concatenate(
                (sphere_mean, np.zeros(space_dimensionality - sphere_dimensionality))
            )
            for sphere_mean in sphere_means_in_space_dim
        ]

        sphere_points = np.zeros((total_num_points, space_dimensionality))
    else:
        sphere_points = np.zeros((total_num_points, sphere_dimensionality))
    sphere_point_labels = np.zeros(total_num_points)

    for i, loc in enumerate(sphere_means_in_space_dim):
        for j in range(num_points):
            sphere_point_idx = i * num_points + j

            # Method 20 from (accessed 31th of January 2021):
            # http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
            u = np.random.normal(loc=0, scale=1, size=sphere_dimensionality)
            u /= np.linalg.norm(u)
            r = np.random.random() ** (1.0 / sphere_dimensionality)
            x = r * u
            if space_dimensionality is not None:
                x = np.concatenate(
                    (x, np.zeros(space_dimensionality - sphere_dimensionality))
                )

            x += loc  # Shift point by adding mean
            sphere_points[sphere_point_idx] = x

            # Compute distance to sphere origos
            point_in_spheres = []
            for k, sphere_loc in enumerate(sphere_means_in_space_dim):
                dist_to_sphere = np.linalg.norm(x - sphere_loc)
                point_in_sphere = dist_to_sphere <= 1
                point_in_spheres.append(point_in_sphere)

            # If point is between spheres, we label it as 2, indicating overlapping.
            if all(point_in_spheres):
                sphere_point_labels[sphere_point_idx] = 2
            else:

                # Else, use sphere index as label.
                sphere_point_labels[sphere_point_idx] = point_in_spheres.index(True)

    if create_intersection_point:
        sphere_points[total_num_points - 1] = np.concatenate(
            (
                np.repeat(sphere_means[1] / 2, sphere_dimensionality),
                np.zeros(space_dimensionality - sphere_dimensionality),
            )
        )
        sphere_point_labels[total_num_points - 1] = 2

    return sphere_points, sphere_point_labels
