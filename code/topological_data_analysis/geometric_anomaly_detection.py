import sys
from multiprocessing import Array, Pool, cpu_count
from typing import Callable, Tuple

import numpy as np
from fastdist import fastdist
from ripser import ripser
from sklearn.metrics import euclidean_distances
from tqdm.auto import tqdm

sys.path.append("..")
from approx_nn import ApproxNN  # noqa: E402
from topological_data_analysis.ripser_utils import run_ripser_plus_plus  # noqa: E402
from utils import batch_list_gen  # noqa: E402

# Multiprocessing variable dict
mp_var_dict = {}

# Type aliases
DistanceFunc = Callable[[int, int], float]
KnnFunc = Callable[[int, int], Tuple[np.ndarray, np.ndarray]]


def compute_gad_mp_init(
    data_points: Array,
    data_points_shape: tuple,
    distance_func: DistanceFunc,
    knn_func: KnnFunc = None,
) -> None:
    """
    Initializes multiprocessing variable dict for GAD.

    Parameters
    ----------
    data_points: Array
        Multiprocessing array representing the data points.
    data_points_shape : tuple
        Shape of the data points.
    distance_func : DistanceFunc
        Distance function.
    knn_func : KnnFunc
        K-nearest neighbour function.
    """
    mp_var_dict["data_points"] = data_points
    mp_var_dict["data_points_shape"] = data_points_shape
    mp_var_dict["distance_func"] = distance_func
    if knn_func is not None:
        mp_var_dict["knn_func"] = knn_func


def get_point_distance_func(
    data_points: np.ndarray,
    pairwise_distances: np.ndarray = None,
    approx_nn: ApproxNN = None,
) -> DistanceFunc:
    """
    Gets function for computing distance between two points by specifying its indices.
    Either pairwise distances or ApproxNN instance has to be specified, else, we
    default to the L2 norm of data points.

    Parameters
    ----------
    pairwise_distances : np.ndarray
        Pairwise distances between data points.
    approx_nn : ApproxNN, optional
        ApproxNN instance (algorithm must be "annoy").
    data_points : np.ndarray, optional
        Data points

    Returns
    -------
    distance_func : DistanceFunc
        Distance function, taking in two point indices i and j and returns the distance
        between the points.
    """
    if approx_nn is not None:
        return lambda point_idx_i, point_idx_j: approx_nn.get_distance(
            point_idx_i, point_idx_j
        )
    elif pairwise_distances is not None:
        return lambda point_idx_i, point_idx_j: pairwise_distances[
            point_idx_i, point_idx_j
        ]
    else:
        return lambda point_idx_i, point_idx_j: np.linalg.norm(
            data_points[point_idx_i] - data_points[point_idx_j]
        )


def get_nearest_neighbours(
    distances: np.ndarray,
    k_neighbours: int,
) -> Tuple[float, float]:
    """
    Gets nearest K neighbours from an array of distances.

    Parameters
    ----------
    distances : np.ndarray
        Array of distances.
    k_neighbours : int
        Number of neighbours to find.

    Returns
    -------
    sorted_k_distances_indices : np.ndarray
        Indices of K distances, similarly sorted as `sorted_k_distances`.
    sorted_k_distances : np.ndarray
        K distances, sorted from smallest to largest.
    """
    sorted_k_distances_indices = np.argsort(distances)[1 : k_neighbours + 1]
    sorted_k_distances = distances[sorted_k_distances_indices]
    return sorted_k_distances_indices, sorted_k_distances


def get_knn_func_data_points(
    data_points: np.ndarray,
    pairwise_distances: np.ndarray = None,
    approx_nn: ApproxNN = None,
    metric: Callable = fastdist.euclidean,
    metric_name: str = "euclidean",
) -> KnnFunc:
    """
    Gets a K-nearest neighbour callable for data points, used in `compute_gad`.

    Parameters
    ----------
    data_points : np.ndarray
        Data points.
    pairwise_distances : np.ndarray, optional
        Pairwise distances of data points (defaults to None).
    approx_nn : ApproxNN, optional
        ApproxNN instance.
    metric : Callable, optional
        fastdist metric; only required if `pairwise_distances` and `approx_nn` are None
        (defaults to fastdist.euclidean).
    metric_name : str, optional
        String name of the `metric` callable (defaults to "euclidean").

    Returns
    -------
    knn_func : KnnFunc
        K-nearest neighbour callable for data points.
    """
    if approx_nn is not None:
        return lambda point_idx, k_neighbours: approx_nn.search(
            query_vector=data_points[point_idx],
            k_neighbours=k_neighbours,
            excluded_neighbour_indices=[point_idx],
            return_distances=True,
        )
    elif pairwise_distances is not None:
        return lambda point_idx, k_neighbours: get_nearest_neighbours(
            distances=pairwise_distances[point_idx],
            k_neighbours=k_neighbours,
        )
    else:
        return lambda point_idx, k_neighbours: get_nearest_neighbours(
            distances=fastdist.vector_to_matrix_distance(
                u=data_points[point_idx],
                m=data_points,
                metric=metric,
                metric_name=metric_name,
            ),
            k_neighbours=k_neighbours,
        )


def compute_gad_point_indices(
    data_point_indices: list,
    data_points: np.ndarray,
    data_point_ints: list,
    annulus_inner_radius: float,
    annulus_outer_radius: float,
    distance_func: DistanceFunc,
    use_knn_annulus: bool,
    knn_func: KnnFunc,
    knn_annulus_inner: int,
    knn_annulus_outer: int,
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
    data_point_indices : list
        List consising of indices of data points to compute GAD for.
    data_points : np.ndarray
        All data points.
    data_point_ints : np.ndarray
        Array specifying which data point indices are used from all the data points.
    annulus_inner_radius : float
        Inner annulus radius.
    annulus_outer_radius : float
        Outer annulus radius.
    distance_func : DistanceFunc
        Distance function to measure distances between any two data points.
    use_knn_annulus : bool
        Whether or not to use the KNN verison of GAD.
    knn_func : KnnFunc
        K-nearest neighbour function to find K nearest neighbour of any data point.
    knn_annulus_inner : int
        Number of neighbours to determine inner annulus radius.
    knn_annulus_outer : int
        Number of neighbours to determine outer annulus radius.
    target_homology_dim : int
        Target homology dimension (k parameter in [1]).
    use_ripser_plus_plus : bool
        Whether or not to use Ripser++ (GPU acceleration).
    ripser_plus_plus_threshold : int
        The least number of data points in order to use Ripser++, only has an effect
        if `use_ripser_plus_plus` is set to True.
    return_annlus_persistence_diagrams : bool
        Whether or not to return annulus persistence diagrams.
    progressbar_enabled : bool
        Whether or not the tqdm progressbar is enabled.

    Returns
    -------
    result : dict
        Result dictionary consisting of:
            "P_man" : list
                List of point indices of k-manifold points.
            "P_bnd" : list
                List of point indices of boundary points.
            "P_int" : list
                List of point indices of intersection points.
            "annlus_persistence_diagrams" : list
                List of persistence diagrams of annulus points, if
                `return_annlus_persistence_diagrams` is set to True.

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
        if use_knn_annulus:
            annulus_outer_indices, annulus_outer_distances = knn_func(
                data_point_index, knn_annulus_outer
            )

            # Set annulus inner and outer radii and A_y_indices
            annulus_inner_radius = annulus_outer_distances[knn_annulus_inner]
            annulus_outer_radius = annulus_outer_distances[-1]
            A_y_indices = annulus_outer_indices[knn_annulus_inner:]
        else:
            A_y_indices = np.array(
                [
                    j
                    for j in data_point_ints
                    if annulus_inner_radius
                    <= distance_func(j, data_point_index)
                    <= annulus_outer_radius
                ],
                dtype=int,
            )

        # Return already if there are no points satisfying condition in (*).
        N_y = 0
        if len(A_y_indices) == 0:
            result["P_bnd"].append(data_point_index)
            if return_annlus_persistence_diagrams:
                result["annulus_pds"][data_point_index] = []
            continue

        # Compute (k-1) Vietoris-Rips barcode of A_y
        A_y = data_points[A_y_indices]
        if use_ripser_plus_plus and len(A_y) > ripser_plus_plus_threshold:
            diagrams_dict = run_ripser_plus_plus(
                point_cloud=A_y, max_dim=target_homology_dim
            )
            diagrams = list(diagrams_dict.values())
        else:
            # rips_complex = RipsComplex(points=A_y)
            # simplex_tree = rips_complex.create_simplex_tree(
            #     max_dimension=target_homology_dim
            # )
            # barcodes = simplex_tree.persistence()
            # target_homology_dim_diagram = np.array(
            #     [
            #         (birth, death)
            #         for dim, (birth, death) in barcodes
            #         if dim == target_homology_dim
            #     ]
            # )

            rips_complex = ripser(
                X=euclidean_distances(A_y),
                maxdim=target_homology_dim,
                distance_matrix=True,
            )
            diagrams = rips_complex["dgms"]
        target_homology_dim_diagram = diagrams[target_homology_dim]
        # print(target_homology_dim_diagram.shape)

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
    args : tuple
        Multiprocessing argument tuple:
            data_point_indices : list
                List consising of indices of data points to compute GAD for.
            data_point_ints : np.ndarray
                Array specifying which data point indices are used from all the data points.
            annulus_inner_radius : float
                Inner annulus radius.
            annulus_outer_radius : float
                Outer annulus radius.
            use_knn_annulus : bool
                Whether or not to use the KNN verison of GAD.
            knn_annulus_inner : int
                Number of neighbours to determine inner annulus radius.
            knn_annulus_outer : int
                Number of neighbours to determine outer annulus radius.
            target_homology_dim : int
                Target homology dimension (k parameter in [1]).
            use_ripser_plus_plus : bool
                Whether or not to use Ripser++ (GPU acceleration).
            ripser_plus_plus_threshold : int
                The least number of data points in order to use Ripser++, only has an effect
                if `use_ripser_plus_plus` is set to True.
            return_annlus_persistence_diagrams : bool
                Whether or not to return annulus persistence diagrams.

    Returns
    -------
    result : dict
        Result dictionary consisting of:
            "P_man" : list
                List of point indices of k-manifold points.
            "P_bnd" : list
                List of point indices of boundary points.
            "P_int" : list
                List of point indices of intersection points.
            "annlus_persistence_diagrams" : list
                List of persistence diagrams of annulus points, if
                `return_annlus_persistence_diagrams` is set to True.

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
        use_knn_annulus,
        knn_annulus_inner,
        knn_annulus_outer,
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
    knn_func = None
    if use_knn_annulus:
        knn_func = mp_var_dict["knn_func"]

    # Compute GAD and return
    return compute_gad_point_indices(
        data_point_indices=data_point_indices,
        data_points=data_points,
        data_point_ints=data_point_ints,
        annulus_inner_radius=annulus_inner_radius,
        annulus_outer_radius=annulus_outer_radius,
        distance_func=distance_func,
        use_knn_annulus=use_knn_annulus,
        knn_func=knn_func,
        knn_annulus_inner=knn_annulus_inner,
        knn_annulus_outer=knn_annulus_outer,
        target_homology_dim=target_homology_dim,
        use_ripser_plus_plus=use_ripser_plus_plus,
        ripser_plus_plus_threshold=ripser_plus_plus_threshold,
        return_annlus_persistence_diagrams=return_annlus_persistence_diagrams,
        progressbar_enabled=True,
    )


def compute_gad(
    data_points: np.ndarray,
    manifold_dimension: int,
    annulus_inner_radius: float = None,
    annulus_outer_radius: float = None,
    data_point_ints: list = None,
    data_points_pairwise_distances: np.ndarray = None,
    data_points_approx_nn: ApproxNN = None,
    use_ripser_plus_plus: bool = False,
    ripser_plus_plus_threshold: int = 200,
    use_knn_annulus: bool = False,
    knn_annulus_inner: int = None,
    knn_annulus_outer: int = None,
    knn_annulus_metric: Callable = fastdist.euclidean,
    knn_annulus_metric_name: str = "euclidean",
    return_annlus_persistence_diagrams: bool = False,
    progressbar_enabled: bool = False,
    n_jobs: int = 1,
    verbose: int = 1,
) -> dict:
    """
    Computes geometric anomaly detection (GAD) Procedure 1 from [1].

    Parameters
    ----------
    data_points : np.ndarray
        All data points.
    manifold_dimension : int
        Manifold homology dimension (k parameter in [1]).
    annulus_inner_radius : float
        Inner annulus radius.
    annulus_outer_radius : float
        Outer annulus radius.
    data_point_ints : np.ndarray
        Array specifying which data point indices are used from all the data points.
    data_points_pairwise_distances : np.ndarray, optional
        Pairwise distances of data points (defaults to None).
    data_points_approx_nn : ApproxNN, optional
        ApproxNN instance (defaults to None).
    use_ripser_plus_plus : bool
        Whether or not to use Ripser++ (GPU acceleration).
    ripser_plus_plus_threshold : int
        The least number of data points in order to use Ripser++, only has an effect
        if `use_ripser_plus_plus` is set to True.
    use_knn_annulus : bool
        Whether or not to use the KNN verison of GAD.
    knn_annulus_inner : int
        Number of neighbours to determine inner annulus radius.
    knn_annulus_outer : int
        Number of neighbours to determine outer annulus radius.
    knn_annulus_metric : Callable
        fastdist metric; only required if `data_points_pairwise_distances` and
        `data_points_approx_nn` are None (defaults to fastdist.euclidean).
    knn_annulus_metric_name : str
        String name of the `knn_annulus_metric` callable (defaults to "euclidean").
    return_annlus_persistence_diagrams : bool
        Whether or not to return annulus persistence diagrams.
    progressbar_enabled : bool
        Whether or not the tqdm progressbar is enabled.
    n_jobs : int, optional
        Number of processes to use (defaults 1, -1 denotes all processes).
    verbose : int, optional
        Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose). Defaults to 1 (verbose).

    Returns
    -------
    result : dict
        Result dictionary consisting of:
            "P_man" : list
                List of point indices of k-manifold points.
            "P_bnd" : list
                List of point indices of boundary points.
            "P_int" : list
                List of point indices of intersection points.
            "annlus_persistence_diagrams" : list
                List of persistence diagrams of annulus points, if
                `return_annlus_persistence_diagrams` is set to True.

    References
    ----------
    .. [1] Bernadette J Stolz, Jared Tanner, Heather A Harrington, & Vidit Nanda.
       (2019). Geometric anomaly detection in data.
    """
    if data_point_ints is None:
        data_point_ints = np.arange(len(data_points))

    # Get distance function
    distance_func = get_point_distance_func(
        data_points=data_points,
        pairwise_distances=data_points_pairwise_distances,
        approx_nn=data_points_approx_nn,
    )

    # Get KNN annulus function, use_knn_annulus is True
    knn_func = None
    if use_knn_annulus:
        knn_func = get_knn_func_data_points(
            data_points=data_points,
            pairwise_distances=data_points_pairwise_distances,
            approx_nn=data_points_approx_nn,
            metric=knn_annulus_metric,
            metric_name=knn_annulus_metric_name,
        )

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
        if verbose == 1:
            print("Preparing data for multiprocessing...")
        data_points_shape = (len(data_point_ints), data_points.shape[1])
        data_points_raw = Array(
            "d", data_points_shape[0] * data_points_shape[1], lock=False
        )
        data_points_raw_np = np.frombuffer(data_points_raw).reshape(data_points_shape)
        np.copyto(data_points_raw_np, data_points[data_point_ints])
        if verbose == 1:
            print("Done!")

        # Prepare arguments
        num_data_points_per_process = int(len(data_point_ints) // n_jobs)
        grid_search_args = [
            (
                data_point_ints_chunk,
                data_point_ints,
                annulus_inner_radius,
                annulus_outer_radius,
                use_knn_annulus,
                knn_annulus_inner,
                knn_annulus_outer,
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
            initargs=(data_points_raw_np, data_points_shape, distance_func, knn_func),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(compute_gad_point_indices_mp, grid_search_args),
                total=n_jobs,
                disable=not progressbar_enabled,
            ):
                results["P_man"].extend(result["P_man"])
                results["P_bnd"].extend(result["P_bnd"])
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
            use_knn_annulus=use_knn_annulus,
            knn_func=knn_func,
            knn_annulus_inner=knn_annulus_inner,
            knn_annulus_outer=knn_annulus_outer,
            target_homology_dim=target_homology_dim,
            use_ripser_plus_plus=use_ripser_plus_plus,
            ripser_plus_plus_threshold=ripser_plus_plus_threshold,
            return_annlus_persistence_diagrams=return_annlus_persistence_diagrams,
            progressbar_enabled=progressbar_enabled,
        )

    return results


def grid_search_gad_annulus_radii(
    data_points: np.ndarray,
    manifold_dimension: int,
    search_size: int,
    use_knn_annulus: bool,
    search_params_max_diff: float = np.inf,
    min_annulus_parameter: float = 0,
    max_annulus_parameter: float = -1,
    data_point_ints: list = None,
    data_points_pairwise_distances: np.ndarray = None,
    data_points_approx_nn: ApproxNN = None,
    use_ripser_plus_plus: bool = False,
    ripser_plus_plus_threshold: int = 200,
    return_annlus_persistence_diagrams: bool = False,
    progressbar_enabled: bool = False,
    n_jobs: int = 1,
    verbose: int = 1,
) -> tuple:
    """
    Performs hyperparameter search to find the best set of inner and outer
    annulus radii for the geometric anomaly detection (GAD) Procedure 1 from [1].

    Parameters
    ----------
    data_points : np.ndarray
        All data points.
    manifold_dimension : int
        Manifold homology dimension (k parameter in [1]).
    search_size : int
        Number of radii parameters to use at most (all for outer radius and (all - 1)
        for inner radius).
    use_knn_annulus : bool
        Whether or not to use the KNN verison of GAD.
    search_params_max_diff : float
        Maximal difference between outer and inner radii for annulus.
    min_annulus_parameter : float
        Minimal annulus radius to search over.
    max_annulus_parameter : float
        Maximal annulus radius to search over.
    data_point_ints : np.ndarray
        Array specifying which data point indices are used from all the data points.
    data_points_pairwise_distances : np.ndarray, optional
        Pairwise distances of data points (defaults to None).
    data_points_approx_nn : ApproxNN, optional
        ApproxNN instance (defaults to None).
    use_ripser_plus_plus : bool
        Whether or not to use Ripser++ (GPU acceleration).
    ripser_plus_plus_threshold : int
        The least number of data points in order to use Ripser++, only has an effect
        if `use_ripser_plus_plus` is set to True.
    return_annlus_persistence_diagrams : bool
        Whether or not to return annulus persistence diagrams.
    progressbar_enabled : bool
        Whether or not the tqdm progressbar is enabled.
    n_jobs : int, optional
        Number of processes to use (defaults 1, -1 denotes all processes).
    verbose : int, optional
        Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose). Defaults to 1 (verbose).

    Returns
    -------
    result : tuple
        Tuple containing best result index, P_man counts in a list, results
        from geometric anomaly detection and annulus radii grid.

    References
    ----------
    .. [1] Bernadette J Stolz, Jared Tanner, Heather A Harrington, & Vidit Nanda.
       (2019). Geometric anomaly detection in data.
    """
    if max_annulus_parameter == -1:
        if use_knn_annulus:
            max_annulus_parameter = len(data_points) - 1
        else:
            if data_points_pairwise_distances is not None:
                max_annulus_parameter = np.max(data_points_pairwise_distances)
            else:
                raise ValueError("Maximum pairwise distance must be specified.")

    # Find values for radii to use during search
    radii_space = np.linspace(
        start=min_annulus_parameter,
        stop=max_annulus_parameter,
        num=search_size + 1,
        dtype=int if use_knn_annulus else None,
    )[1:]

    # Grid-search best set of annulus radii to optimize number of P_man data points
    annulus_radii_grid = []
    for inner_idx in range(search_size):
        for outer_idx in range(inner_idx + 1, search_size):
            inner_param = radii_space[inner_idx]
            outer_param = radii_space[outer_idx]

            if outer_param - inner_param <= search_params_max_diff:
                annulus_radii_grid.append((inner_param, outer_param))

    if verbose == 1:
        print("Grid searching...")
    gad_results = []
    P_man_counts = []
    for inner_param, outer_param in tqdm(
        annulus_radii_grid, disable=not progressbar_enabled
    ):
        if use_knn_annulus:
            if verbose == 1:
                print(
                    f"Inner radius neighbours: {inner_param}, outer radius neighbours: {outer_param}"
                )
            gad_params = {
                "knn_annulus_inner": inner_param,
                "knn_annulus_outer": outer_param,
            }
        else:
            if verbose == 1:
                print(
                    f"Inner radius: {inner_param:.3f}, outer radius: {outer_param:.3f}"
                )
            gad_params = {
                "annulus_inner_radius": inner_param,
                "annulus_outer_radius": outer_param,
            }
        gad_result = compute_gad(
            data_points=data_points,
            manifold_dimension=manifold_dimension,
            data_point_ints=data_point_ints,
            data_points_pairwise_distances=data_points_pairwise_distances,
            data_points_approx_nn=data_points_approx_nn,
            use_ripser_plus_plus=use_ripser_plus_plus,
            ripser_plus_plus_threshold=ripser_plus_plus_threshold,
            use_knn_annulus=use_knn_annulus,
            return_annlus_persistence_diagrams=return_annlus_persistence_diagrams,
            progressbar_enabled=progressbar_enabled,
            n_jobs=n_jobs,
            verbose=verbose,
            **gad_params,
        )
        print(
            "P_man:",
            len(gad_result["P_man"]),
            "P_int:",
            len(gad_result["P_int"]),
            "P_bnd:",
            len(gad_result["P_bnd"]),
        )
        P_man_counts.append(len(gad_result["P_man"]))
        gad_results.append(gad_result)

    # Find best result
    best_gad_result_idx = np.argmax(P_man_counts)

    return best_gad_result_idx, P_man_counts, gad_results, annulus_radii_grid
