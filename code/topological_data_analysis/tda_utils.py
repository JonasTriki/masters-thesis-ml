import numpy as np
from gudhi.persistence_graphical_tools import \
    plot_persistence_diagram as gd_plot_persistence_diagram
from gudhi.rips_complex import RipsComplex
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
    skeleton_word2vec = RipsComplex(distance_matrix=pairwise_distances)

    # Plot persistence diagram
    simplex_tree = skeleton_word2vec.create_simplex_tree(
        max_dimension=simplex_tree_max_dims
    )
    barcodes = simplex_tree.persistence()
    gd_plot_persistence_diagram(barcodes)

    if show_plot:
        plt.show()


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
