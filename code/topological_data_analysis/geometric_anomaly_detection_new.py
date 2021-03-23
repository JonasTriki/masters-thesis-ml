import sys
from multiprocessing import Array, Pool, cpu_count
from typing import Callable

import annoy
import numpy as np
from ripser import ripser
from sklearn.metrics import euclidean_distances
from tqdm.auto import tqdm

sys.path.append("..")

from approx_nn import ApproxNN  # noqa: E402
from topological_data_analysis.ripser_utils import run_ripser_plus_plus  # noqa: E402
from utils import batch_list_gen  # noqa: E402

# Multiprocessing variable dict
mp_dict = {}


def compute_gad_mp_init(point_cloud: Array, point_cloud_shape: tuple) -> None:
    """
    Initializes multiprocessing variable dict for GAD.

    Parameters
    ----------
    point_cloud: Array
        Multiprocessing array representing the point cloud.
    point_cloud_shape : tuple
        Shape of the point cloud.
    """
    mp_dict["point_cloud"] = point_cloud
    mp_dict["point_cloud_shape"] = point_cloud_shape


def get_point_distance_func(
    point_cloud: np.ndarray = None,
    pairwise_distances: np.ndarray = None,
    approx_nn: ApproxNN = None,
) -> Callable[[int, int], float]:
    """
    Gets function for computing distance between two points by specifying its indices.
    One of the parameters has to be specified to yield a distance function.

    Parameters
    ----------
    pairwise_distances : np.ndarray, optional
        Pairwise distances between data points.
    approx_nn : ApproxNN, optional
        ApproxNN instance (algorithm must be "annoy").
    point_cloud : np.ndarray, optional
        Point cloud.

    Returns
    -------
    distance_func : Callable[[int, int], float]
        Distance function, taking in two point indices i and j and returns the distance
        between the points.
    """
    if point_cloud is not None:
        return lambda point_idx_i, point_idx_j: np.linalg.norm(
            point_cloud[point_idx_i] - point_cloud[point_idx_j]
        )
    elif pairwise_distances is not None:
        return lambda point_idx_i, point_idx_j: pairwise_distances[
            point_idx_i, point_idx_j
        ]
    elif approx_nn is not None:
        return lambda point_idx_i, point_idx_j: approx_nn.get_distance(
            point_idx_i, point_idx_j
        )
    else:
        raise ValueError("One of the parameters has to be specified.")


def compute_gad(
    point_cloud: np.ndarray,
    manifold_dimension: int,
    distance_func: Callable[[int, int], float],
    annulus_inner_radius: float,
    annulus_outer_radius: float,
    point_cloud_ints: list = None,
    return_annlus_persistence_diagrams: bool = False,
    progressbar_enabled: bool = False,
) -> tuple:
    """
    Computes geometric anomaly detection Procedure 1 from [1].

    Parameters
    ----------

    Returns
    -------
    P_man : list
        List of point indices of k-manifold points.
    P_bnd : list
        List of point indices of boundary points.
    P_int : list
        List of point indices of intersection points.
    annlus_persistence_diagrams : list
        List of persistence diagrams of annulus points, if
        return_annlus_persistence_diagrams is set to True.

    References
    ----------
    .. [1] Bernadette J Stolz, Jared Tanner, Heather A Harrington, & Vidit Nanda.
        (2019). Geometric anomaly detection in data.
    """
    if point_cloud_ints is None:
        point_cloud_ints = np.arange(len(point_cloud))

    # Initialize result
    P_bnd = []
    P_man = []
    P_int = []
    annulus_pds = None
    if return_annlus_persistence_diagrams:
        annulus_pds = []
    persistence_threshold = abs(annulus_outer_radius - annulus_inner_radius)

    if return_annlus_persistence_diagrams:
        return P_man, P_bnd, P_int, annulus_pds
    else:
        return P_man, P_bnd, P_int
