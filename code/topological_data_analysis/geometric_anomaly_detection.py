import subprocess
import sys
from multiprocessing import Array, Pool, cpu_count
from os.path import join
from typing import Callable, Dict

import annoy
import numpy as np
from ripser import ripser
from ripser_utils import run_ripser_plus_plus
from sklearn.metrics import euclidean_distances
from tqdm.auto import tqdm

sys.path.append("..")

from utils import batch_list_gen  # noqa: E402


def grid_search_prepare_word_ints_within_radii(
    word_ints: list,
    num_radii_per_parameter: int,
    word_vector_distance: Callable[[int, int], float],
    min_outer_annulus_radius: float = 0,
    max_outer_annulus_radius: float = -1,
    word_embeddings_pairwise_dists: np.ndarray = None,
) -> tuple:
    """
    Prepares dictionary with words within each radii for grid search in
    GeometricAnomalyDetection class.

    Parameters
    ----------
    word_ints : list
        List word integer representations, signalizing what part of the
        vocabulary we want to use.
    num_radii_per_parameter : int
        Number of inner/outer radii to search over.
    word_vector_distance : callable
        Callable which takes in two indices i and j and returns the distance
        between word vector i and j.
    min_outer_annulus_radius : float
        Minimal outer annulus radius to search over.
    max_outer_annulus_radius : float
        Maximal outer annulus radius to search over. Must
        be specified if word_embeddings_pairwise_dists is None.
    word_embeddings_pairwise_dists : np.ndarray
        Numpy matrix containing pairwise distances between word embeddings
        (defaults to None).

    Returns
    -------
    result : tuple
        Tuple containing dictionary with word ints within reach radii for grid search
        and radii space.
    """
    # Find largest pairwise distance between word embeddings
    if max_outer_annulus_radius == -1:
        if word_embeddings_pairwise_dists is not None:
            max_outer_annulus_radius = np.max(word_embeddings_pairwise_dists)
        else:
            raise ValueError("Maximum pairwise distance must be specified.")

    # Find values for radii to use during search
    radii_space = np.linspace(
        start=min_outer_annulus_radius,
        stop=max_outer_annulus_radius,
        num=num_radii_per_parameter + 1,
    )[1:]

    # Prepare words within radii dictionary
    word_ints_within_radii: dict = {}
    for i in word_ints:
        word_ints_within_radii[i] = []
        for _ in range(num_radii_per_parameter):
            word_ints_within_radii[i].append([])

    # Precompute words within radii to speed up search
    print("Compute words within radii...")
    for i in tqdm(word_ints):
        for j in word_ints[i + 1 :]:
            dist = word_vector_distance(i, j)
            for k, radius in enumerate(radii_space):
                if dist <= radius:
                    word_ints_within_radii[i][k].append(j)
                    word_ints_within_radii[j][k].append(i)

    return word_ints_within_radii, radii_space


# Multiprocessing variable dict for grid-search
mp_var_dict = {}


def compute_gad_mp_init(word_embeddings: Array, word_embeddings_shape: tuple) -> None:
    mp_var_dict["word_embeddings"] = word_embeddings
    mp_var_dict["word_embeddings_shape"] = word_embeddings_shape


def compute_gad_multiprocessing(args: tuple) -> dict:
    """
    TODO: Docs

    Parameters
    ----------
    args : tuple

    Returns
    -------
    result : dict
    """
    # Parse args
    (
        word_ints,
        annulus_inner_radius,
        annulus_outer_radius,
        annoy_index_filepath,
        target_homology_dim,
        use_ripser_plus_plus,
    ) = args
    persistence_threshold = abs(annulus_outer_radius - annulus_inner_radius)

    # Load word embeddings from var dict
    word_embeddings = np.frombuffer(mp_var_dict["word_embeddings"]).reshape(
        mp_var_dict["word_embeddings_shape"]
    )

    annoy_index = annoy.AnnoyIndex(f=word_embeddings.shape[1], metric="euclidean")
    annoy_index.load(fn=annoy_index_filepath)

    # Initialize result
    P_bnd = []
    P_man = []
    P_int = []

    for i in word_ints:

        # Find A_y ⊂ word_vectors containing all word vectors in word_vectors
        # which satisfy r ≤ ||x − y|| ≤ s.
        A_y_indices = np.array(
            [
                j
                for j in word_ints
                if annulus_inner_radius
                <= annoy_index.get_distance(j, i)
                <= annulus_outer_radius
            ]
        )
        if len(A_y_indices) == 0:
            P_bnd.append(i)
            continue

        # Compute (k-1) Vietoris-Rips barcode of A_y
        A_y = word_embeddings[A_y_indices]
        if use_ripser_plus_plus and len(A_y) > 200:
            diagrams_dict = run_ripser_plus_plus(
                point_cloud=A_y, max_dim=target_homology_dim
            )
            diagrams = list(diagrams_dict.values())
        else:
            A_y_pairwise_dists = euclidean_distances(A_y)
            rips_complex = ripser(
                X=A_y_pairwise_dists,
                maxdim=target_homology_dim,
                distance_matrix=True,
            )
            diagrams = rips_complex["dgms"]

        # Calculate number of intervals in A_y_barcodes of length
        # (death - birth) > (annulus_outer_radius - annulus_inner_radius).
        N_y = 0
        for birth, death in diagrams[target_homology_dim]:
            if (death - birth) > persistence_threshold:
                N_y += 1

        # Add result
        if N_y == 0:
            P_bnd.append(i)
        elif N_y == 1:
            P_man.append(i)
        else:
            P_int.append(i)

    return {
        "P_bnd": P_bnd,
        "P_man": P_man,
        "P_int": P_int,
    }


class GeometricAnomalyDetection:
    """
    Class for computing geometric anomaly detection Procedure 1 from [1].

    References
    ----------
    .. [1] Bernadette J Stolz, Jared Tanner, Heather A Harrington, & Vidit Nanda.
       (2019). Geometric anomaly detection in data.
    """

    def __init__(
        self,
        word_embeddings: np.ndarray,
    ) -> None:
        """
        Initializes a geometric anomaly detection instance.

        Parameters
        ----------
        word_embeddings : np.ndarray
            Word embeddings
        """
        self._word_embeddings = word_embeddings

    def _compute_gad(
        self,
        manifold_dimension: int,
        word_vector_distance: Callable[[int, int], float],
        annulus_inner_radius: float = None,
        annulus_outer_radius: float = None,
        annulus_inner_idx: int = -1,
        annulus_outer_idx: int = -1,
        word_ints_within_radii: dict = None,
        radii_space: list = None,
        word_ints: list = None,
        tqdm_enabled: bool = False,
        use_ripser_plus_plus: bool = False,
    ) -> dict:
        """
        Computes geometric anomaly detection Procedure 1 from [1]. Either
        annulus_inner_radius/annulus_outer_radius or annulus_inner_idx/annulus_outer_idx
        and word_ints_within_radii must be specified.

        Parameters
        ----------
        manifold_dimension : int
            Manifold dimension to detect intersections with (k).
        word_vector_distance : callable
            Callable which takes in two indices i and j and returns the distance
            between word vector i and j.
        annulus_inner_radius : float
            Inner radius parameter (r) for annulus.
        annulus_outer_radius : float
            Outer radius parameter (s) for annulus.
        annulus_inner_idx : int
            Index for the inner annulus (s).
        annulus_outer_idx : int
            Index for the inner annulus (s).
        word_ints_within_radii : dict
            Dictionary mapping from word integer to word integers for each radius index.
        radii_space : list
            List of radii used if annulus_inner_idx and annulus_outer_idx are specified.
        word_ints : list, optional
            List word integer representations, signalizing what part of the
            vocabulary we want to use. Set to None to use whole vocabulary.
        tqdm_enabled : bool, optional
            Whether or not to show the progress using tqdm (defaults to False).
        use_ripser_plus_plus : bool, optional
            Whether or not to use Ripser++, speeding up the computation

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
        # Verify input
        annulus_radius_specified = (
            annulus_inner_radius is not None and annulus_outer_radius is not None
        )
        precomputed_word_ints_within_radii_specified = (
            annulus_inner_idx != -1
            and annulus_outer_idx != -1
            and word_ints_within_radii is not None
            and radii_space is not None
        )
        if not (
            annulus_radius_specified or precomputed_word_ints_within_radii_specified
        ):
            raise ValueError(
                "Either annulus inner/outer radius or word ints within radii needs to be specified."
            )

        if word_ints is None:
            word_ints = np.arange(len(self._word_embeddings))

        # Initialize result
        P_bnd = []
        P_man = []
        P_int = []

        if annulus_radius_specified:
            persistence_threshold = abs(annulus_outer_radius - annulus_inner_radius)
        else:
            persistence_threshold = abs(
                radii_space[annulus_outer_idx] - radii_space[annulus_inner_idx]
            )

        target_homology_dim = manifold_dimension - 1
        for i in tqdm(word_ints, disable=not tqdm_enabled):

            # Find A_y ⊂ word_vectors containing all word vectors in word_vectors
            # which satisfy r ≤ ||x − y|| ≤ s.
            if annulus_inner_radius is not None and annulus_outer_radius is not None:
                A_y_indices = np.array(
                    [
                        j
                        for j in word_ints
                        if annulus_inner_radius
                        <= word_vector_distance(j, i)
                        <= annulus_outer_radius
                    ]
                )
            else:
                word_ints_within_inner_radii = word_ints_within_radii[i][
                    annulus_inner_idx
                ]
                word_ints_within_outer_radii = word_ints_within_radii[i][
                    annulus_outer_idx
                ]
                A_y_indices = np.setdiff1d(
                    word_ints_within_outer_radii, word_ints_within_inner_radii
                )
            if len(A_y_indices) == 0:
                P_bnd.append(i)
                continue

            # Compute (k-1) Vietoris-Rips barcode of A_y
            A_y = self._word_embeddings[A_y_indices]
            if use_ripser_plus_plus and len(A_y) > 200:
                diagrams_dict = run_ripser_plus_plus(
                    point_cloud=A_y, max_dim=target_homology_dim
                )
                diagrams = list(diagrams_dict.values())
            else:
                A_y_pairwise_dists = euclidean_distances(A_y)
                rips_complex = ripser(
                    X=A_y_pairwise_dists,
                    maxdim=target_homology_dim,
                    distance_matrix=True,
                )
                diagrams = rips_complex["dgms"]

            # Calculate number of intervals in A_y_barcodes of length
            # (death - birth) > (annulus_outer_radius - annulus_inner_radius).
            N_y = 0
            for birth, death in diagrams[target_homology_dim]:
                if (death - birth) > persistence_threshold:
                    N_y += 1

            # Add result
            if N_y == 0:
                P_bnd.append(i)
            elif N_y == 1:
                P_man.append(i)
            else:
                P_int.append(i)

        return {
            "P_bnd": P_bnd,
            "P_man": P_man,
            "P_int": P_int,
        }

    def compute(
        self,
        word_ints: list,
        manifold_dimension: int,
        annulus_inner_radius: float,
        annulus_outer_radius: float,
        word_embeddings_pairwise_dists: np.ndarray = None,
        annoy_index: annoy.AnnoyIndex = None,
        tqdm_enabled: bool = False,
    ) -> dict:
        """
        Computes geometric anomaly detection Procedure 1 from [1].

        Parameters
        ----------
        word_ints : list
            List word integer representations, signalizing what part of the
            vocabulary we want to use. Set to None to use whole vocabulary.
        manifold_dimension : int
            Manifold dimension to detect intersections with (k).
        annulus_inner_radius : float
            Inner radius parameter (r) for annulus.
        annulus_outer_radius : float
            Outer radius parameter (s) for annulus.
            Dictionary mapping from word integer to word integers for each radius index.
        word_embeddings_pairwise_dists : np.ndarray, optional
            Numpy matrix containing pairwise distances between word embeddings
            (defaults to None).
        annoy_index : annoy.AnnoyIndex, optional
            Annoy index built on the word embeddings (defaults to None).
            If specified, the approximate nearest neighbour index is used to compute
            distance between word vectors.
        tqdm_enabled : bool, optional
            Whether or not to show the progress using tqdm (defaults to False).

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
        # Get word vector distance callable
        if word_embeddings_pairwise_dists is not None:
            word_vector_distance = (
                lambda word_i, word_j: word_embeddings_pairwise_dists[word_i, word_j]
            )
        elif annoy_index is not None:
            word_vector_distance = lambda word_i, word_j: annoy_index.get_distance(
                word_i, word_j
            )
        else:
            word_vector_distance = lambda word_i, word_j: np.linalg.norm(
                self._word_embeddings[word_i] - self._word_embeddings[word_j]
            )

        return self._compute_gad(
            manifold_dimension=manifold_dimension,
            word_vector_distance=word_vector_distance,
            annulus_inner_radius=annulus_inner_radius,
            annulus_outer_radius=annulus_outer_radius,
            word_ints=word_ints,
            tqdm_enabled=tqdm_enabled,
        )

    def _compute_gad_mp(
        self,
        manifold_dimension: int,
        annulus_inner_radius: float,
        annulus_outer_radius: float,
        annoy_index_filepath: str,
        word_ints: list = None,
        use_ripser_plus_plus: bool = False,
        num_cpus: int = -1,
    ) -> dict:
        """
        Computes geometric anomaly detection Procedure 1 from [1] using
        multiprocessing.

        Parameters
        ----------
        manifold_dimension : int
            Manifold dimension to detect intersections with (k).
        annulus_inner_radius : float
            Inner radius parameter (r) for annulus.
        annulus_outer_radius : float
            Outer radius parameter (s) for annulus.
        annoy_index_filepath : str
            Filepath of Annoy index
        word_ints : list, optional
            List word integer representations, signalizing what part of the
            vocabulary we want to use. Set to None to use whole vocabulary.
        use_ripser_plus_plus : bool
            Whether or not to use Ripser++ and GPUs for computing Rips complices.
        num_cpus : int
            Number of CPUs to use (defaults -1 = to all CPUs).

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
        if word_ints is None:
            word_ints = np.arange(len(self._word_embeddings))

        # Prepare data for multiprocessing
        word_embeddings_shape = (len(word_ints), self._word_embeddings.shape[1])
        word_embeddings_raw = Array(
            "d", word_embeddings_shape[0] * word_embeddings_shape[1], lock=False
        )
        word_embeddings_raw_np = np.frombuffer(word_embeddings_raw).reshape(
            word_embeddings_shape
        )
        np.copyto(word_embeddings_raw_np, self._word_embeddings[word_ints])

        # Prepare arguments for multiprocessing
        if num_cpus == -1:
            num_cpus = cpu_count()
        num_word_ints_per_process = int(word_embeddings_shape[0] // num_cpus)
        grid_search_args = [
            (
                word_int_chunk,
                annulus_inner_radius,
                annulus_outer_radius,
                annoy_index_filepath,
                manifold_dimension - 1,
                use_ripser_plus_plus,
            )
            for word_int_chunk in batch_list_gen(word_ints, num_word_ints_per_process)
        ]

        # Initialize result
        P_bnd = []
        P_man = []
        P_int = []

        # Run MP
        with Pool(
            processes=num_cpus,
            initializer=compute_gad_mp_init,
            initargs=(word_embeddings_raw, word_embeddings_shape),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(compute_gad_multiprocessing, grid_search_args),
                total=num_cpus,
            ):
                P_bnd.extend(result["P_bnd"])
                P_man.extend(result["P_man"])
                P_int.extend(result["P_int"])

        return {
            "P_bnd": P_bnd,
            "P_man": P_man,
            "P_int": P_int,
        }

    def grid_search_radii(
        self,
        word_ints: list,
        manifold_dimension: int,
        num_radii_per_parameter: int,
        annoy_index_filepath: str,
        outer_inner_radii_max_diff: float = np.inf,
        min_outer_annulus_radius: float = 0,
        max_outer_annulus_radius: float = -1,
        word_embeddings_pairwise_dists: np.ndarray = None,
        use_ripser_plus_plus: bool = False,
        num_cpus: int = -1,
    ) -> tuple:
        """
        Performs grid search to find the best pair of inner/outer annulus radii.
        The objective of the grid search is to maximize the number of words categorized
        as P_man.

        Parameters
        ----------
        word_ints : list, optional
            List word integer representations, signalizing what part of the
            vocabulary we want to use. Set to None to use whole vocabulary.
        manifold_dimension : int
            Manifold dimension to detect intersections with (k).
        num_radii_per_parameter : int
            Number of inner/outer radii to search over.
        annoy_index_filepath : str
            Filepath of Annoy index built on the word embeddings
        outer_inner_radii_max_diff : float
            Maximal difference between outer and inner radii (defaults to np.inf => unbounded).
        min_outer_annulus_radius : float
            Minimal outer annulus radius to search over.
        max_outer_annulus_radius : float
            Maximal outer annulus radius to search over. Must
            be specified if word_embeddings_pairwise_dists is None.
        word_embeddings_pairwise_dists : np.ndarray, optional
            Numpy matrix containing pairwise distances between word embeddings
            (defaults to None).
        use_ripser_plus_plus : bool
            Whether or not to use Ripser++ and GPUs for computing Rips complices.
        num_cpus : int
            Number of CPUs to use (defaults -1 = to all CPUs).

        Returns
        -------
        result : tuple
            Tuple containing best result index, P_man counts in a list, results
            from geometric anomaly detection and annulus radii grid.
        """
        if max_outer_annulus_radius == -1:
            if word_embeddings_pairwise_dists is not None:
                max_outer_annulus_radius = np.max(word_embeddings_pairwise_dists)
            else:
                raise ValueError("Maximum pairwise distance must be specified.")

        # Find values for radii to use during search
        radii_space = np.linspace(
            start=min_outer_annulus_radius,
            stop=max_outer_annulus_radius,
            num=num_radii_per_parameter + 1,
        )[1:]

        # Grid-search best set of annulus radii to optimize number of P_man words
        annulus_radii_grid = []
        for inner_idx in range(num_radii_per_parameter):
            for outer_idx in range(inner_idx + 1, num_radii_per_parameter):
                inner_radius = radii_space[inner_idx]
                outer_radius = radii_space[outer_idx]

                if outer_radius - inner_radius <= outer_inner_radii_max_diff:
                    annulus_radii_grid.append((inner_radius, outer_radius))

        print("Grid searching...")
        gad_results = []
        P_man_counts = []
        for inner_radius, outer_radius in tqdm(annulus_radii_grid):
            print(f"Inner radius: {inner_radius:.3f}, outer radius: {outer_radius:.3f}")
            gad_result = self._compute_gad_mp(
                manifold_dimension=manifold_dimension,
                annulus_inner_radius=inner_radius,
                annulus_outer_radius=outer_radius,
                annoy_index_filepath=annoy_index_filepath,
                word_ints=word_ints,
                use_ripser_plus_plus=use_ripser_plus_plus,
                num_cpus=num_cpus,
            )
            P_man_counts.append(len(gad_result["P_man"]))
            gad_results.append(gad_result)

        # Find best result
        best_gad_result_idx = np.argmax(P_man_counts)

        return best_gad_result_idx, P_man_counts, gad_results, annulus_radii_grid
