import sys
from multiprocessing import Array, Pool, cpu_count
from typing import Callable

import numpy as np
from ripser import ripser
from sklearn.metrics import euclidean_distances
from tqdm.auto import tqdm

sys.path.append("..")

from approx_nn import ApproxNN  # noqa: E402
from topological_data_analysis.ripser_utils import run_ripser_plus_plus  # noqa: E402
from utils import batch_list_gen  # noqa: E402

# Multiprocessing variable dict
mp_var_dict = {}


def compute_gad_mp_init(
    data_points: Array,
    data_points_shape: tuple,
    distance_func: Callable[[int, int], float],
) -> None:
    """
    Initializes multiprocessing variable dict for GAD.

    Parameters
    ----------
    data_points: Array
        Multiprocessing array representing the data points.
    data_points_shape : tuple
        Shape of the data points.
    distance_func : Callable[[int, int], float]
        Distance function.
    """
    mp_var_dict["data_points"] = data_points
    mp_var_dict["data_points_shape"] = data_points_shape
    mp_var_dict["distance_func"] = distance_func


def get_point_distance_func(
    data_points: np.ndarray = None,
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
    data_points : np.ndarray, optional
        Data points

    Returns
    -------
    distance_func : Callable[[int, int], float]
        Distance function, taking in two point indices i and j and returns the distance
        between the points.
    """
    if data_points is not None:
        return lambda point_idx_i, point_idx_j: np.linalg.norm(
            data_points[point_idx_i] - data_points[point_idx_j]
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


def compute_gad_point_indices(
    data_point_indices: list,
    data_points: np.ndarray,
    data_point_ints: list,
    annulus_inner_radius: float,
    annulus_outer_radius: float,
    distance_func: Callable[[int, int], float],
    target_homology_dim: int,
    use_ripser_plus_plus: bool,
    ripser_plus_plus_threshold: int,
    return_annlus_persistence_diagrams: bool,
    progressbar_enabled: bool,
) -> dict:
    """
    Computes geometric anomaly detection (GAD) Procedure 1 from [1], for data point
    indices.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO

    References
    ----------
    .. [1] Bernadette J Stolz, Jared Tanner, Heather A Harrington, & Vidit Nanda.
       (2019). Geometric anomaly detection in data.
    """
    # Initialize result
    result = {
        "P_bnd": [],
        "P_man": [],
        "P_int": [],
    }
    if return_annlus_persistence_diagrams:
        result["annulus_pds"] = {}

    for data_point_index in tqdm(data_point_indices, disable=not progressbar_enabled):

        # Find A_y ⊂ data_points containing all points in data_points
        # which satisfy r ≤ ||x − y|| ≤ s (*).
        A_y_indices = np.array(
            [
                j
                for j in data_point_ints
                if annulus_inner_radius
                <= distance_func(j, data_point_index)
                <= annulus_outer_radius
            ]
        )

        # Return already if there are no points satisfying condition in (*).
        N_y = 0
        if len(A_y_indices) == 0:
            if return_annlus_persistence_diagrams:
                return N_y, []
            else:
                return N_y

        # Compute (k-1) Vietoris-Rips barcode of A_y
        A_y = data_points[A_y_indices]
        if use_ripser_plus_plus and len(A_y) > ripser_plus_plus_threshold:
            diagrams_dict = run_ripser_plus_plus(
                data_points=A_y, max_dim=target_homology_dim
            )
            diagrams = list(diagrams_dict.values())
        else:
            rips_complex = ripser(
                X=euclidean_distances(A_y),
                maxdim=target_homology_dim,
                distance_matrix=True,
            )
            diagrams = rips_complex["dgms"]
        target_homology_dim_diagram = diagrams[target_homology_dim]

        # Calculate number of intervals in A_y_barcodes of length
        # (death - birth) > abs(annulus_outer_radius - annulus_inner_radius).
        N_y = 0
        for birth, death in target_homology_dim_diagram:
            if (death - birth) > abs(annulus_outer_radius - annulus_inner_radius):
                N_y += 1

        # Add result
        if N_y == 0:
            result["P_bnd"].append(data_point_index)
        elif N_y == 1:
            result["P_man"].append(data_point_index)
        else:
            result["P_int"].append(data_point_index)
        if return_annlus_persistence_diagrams:
            result["annulus_pds"][data_point_index] = target_homology_dim_diagram

    return result


def compute_gad_point_indices_mp(args: tuple) -> dict:
    """
    Computes geometric anomaly detection (GAD) Procedure 1 from [1], for data point
    indices, taking in args for multiprocessing purposes.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO

    References
    ----------
    .. [1] Bernadette J Stolz, Jared Tanner, Heather A Harrington, & Vidit Nanda.
       (2019). Geometric anomaly detection in data.
    """
    # Parse args
    (
        data_point_indices,
        data_point_ints,
        annulus_inner_radius,
        annulus_outer_radius,
        target_homology_dim,
        use_ripser_plus_plus,
        ripser_plus_plus_threshold,
        return_annlus_persistence_diagrams,
    ) = args

    # Get data_points and distance_func from MP dict
    data_points = np.frombuffer(mp_var_dict["data_points"]).reshape(
        mp_var_dict["data_points_shape"]
    )
    distance_func = mp_var_dict["distance_func"]

    # Compute GAD and return
    return compute_gad_point_indices(
        data_point_indices=data_point_indices,
        data_points=data_points,
        data_point_ints=data_point_ints,
        annulus_inner_radius=annulus_inner_radius,
        annulus_outer_radius=annulus_outer_radius,
        distance_func=distance_func,
        target_homology_dim=target_homology_dim,
        use_ripser_plus_plus=use_ripser_plus_plus,
        ripser_plus_plus_threshold=ripser_plus_plus_threshold,
        return_annlus_persistence_diagrams=return_annlus_persistence_diagrams,
        progressbar_enabled=False,
    )


def compute_gad(
    data_points: np.ndarray,
    manifold_dimension: int,
    distance_func: Callable[[int, int], float],
    annulus_inner_radius: float,
    annulus_outer_radius: float,
    data_point_ints: list = None,
    use_ripser_plus_plus: bool = False,
    ripser_plus_plus_threshold: int = 200,
    return_annlus_persistence_diagrams: bool = False,
    progressbar_enabled: bool = False,
    n_jobs: int = 1,
    verbose: int = 1,
) -> dict:
    """
    Computes geometric anomaly detection (GAD) Procedure 1 from [1].

    Parameters
    ----------
    TODO

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
    if data_point_ints is None:
        data_point_ints = np.arange(len(data_points))

    target_homology_dim = manifold_dimension - 1
    if n_jobs == -1:
        n_jobs = cpu_count()
    if n_jobs > 1:

        # Initialize MP results
        results = {
            "P_bnd": [],
            "P_man": [],
            "P_int": [],
        }
        if return_annlus_persistence_diagrams:
            results["annulus_pds"] = {}

        # Prepare data for multiprocessing
        print("Preparing data for multiprocessing...")
        data_points_shape = (len(data_point_ints), data_points.shape[1])
        data_points_raw = Array(
            "d", data_points_shape[0] * data_points_shape[1], lock=False
        )
        data_points_raw_np = np.frombuffer(data_points_raw).reshape(data_points_shape)
        np.copyto(data_points_raw_np, data_points[data_point_ints])
        print("Done!")

        # Prepare arguments
        num_data_points_per_process = int(data_point_ints.shape[0] // n_jobs)
        grid_search_args = [
            (
                data_point_ints_chunk,
                data_point_ints,
                annulus_inner_radius,
                annulus_outer_radius,
                target_homology_dim,
                use_ripser_plus_plus,
                ripser_plus_plus_threshold,
                return_annlus_persistence_diagrams,
            )
            for data_point_ints_chunk in batch_list_gen(
                data_point_ints, num_data_points_per_process
            )
        ]

        # Run MP
        if verbose == 1:
            print(f"Computing GAD using {n_jobs} processes...")
        with Pool(
            processes=n_jobs,
            initializer=compute_gad_mp_init,
            initargs=(data_points, data_points_shape, distance_func),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(compute_gad_point_indices_mp, grid_search_args),
                total=n_jobs,
                disable=not progressbar_enabled,
            ):
                results["P_bnd"].extend(result["P_bnd"])
                results["P_man"].extend(result["P_man"])
                results["P_int"].extend(result["P_int"])
                if return_annlus_persistence_diagrams:
                    results["annulus_pds"].update(result["annulus_pds"])
    else:

        # Compute GAD using only one processor
        if verbose == 1:
            print("Computing GAD...")
        results = compute_gad_point_indices(
            data_point_indices=data_point_ints,
            data_points=data_points,
            data_point_ints=data_point_ints,
            annulus_inner_radius=annulus_inner_radius,
            annulus_outer_radius=annulus_outer_radius,
            distance_func=distance_func,
            target_homology_dim=target_homology_dim,
            use_ripser_plus_plus=use_ripser_plus_plus,
            ripser_plus_plus_threshold=ripser_plus_plus_threshold,
            return_annlus_persistence_diagrams=return_annlus_persistence_diagrams,
            progressbar_enabled=progressbar_enabled,
        )

    return results
