import gudhi as gd
import numpy as np
from matplotlib import pyplot as plt


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
    skeleton_word2vec = gd.RipsComplex(distance_matrix=pairwise_distances)

    # Plot persistence diagram
    simplex_tree = skeleton_word2vec.create_simplex_tree(
        max_dimension=simplex_tree_max_dims
    )
    barcodes = simplex_tree.persistence()
    gd.plot_persistence_diagram(barcodes)

    if show_plot:
        plt.show()
