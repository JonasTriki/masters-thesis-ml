from os import makedirs
from typing import Optional, Tuple, Union

import annoy
import numpy as np
import scann
from tqdm import tqdm
from typing_extensions import Literal

rng_seed = 399
np.random.seed(rng_seed)


class ApproxNN:
    """
    Approximate nearest neighbour class; using either ScaNN method [1] or Annoy index [2].

    References
    ----------
    .. [1] Guo, R., Sun, P., Lindgren, E., Geng, Q., Simcha, D., Chern, F., & Kumar, S. (2020).
       Accelerating Large-Scale Inference with Anisotropic Vector Quantization.
       In International Conference on Machine Learning.
    .. [2] Erik Bernhardsson. (2018). Annoy: Approximate Nearest Neighbors in C++/Python.
       Url: https://github.com/spotify/annoy.
    """

    def __init__(self, ann_alg: Literal["scann", "annoy"] = "scann") -> None:
        """
        Initializes the approximate nearest neighbour class.

        Parameters
        ----------
        ann_alg : str, "scann" or "annoy"
            Approximate nearest neighbour algorithm/method (defaults to "scann").
        """
        self._ann_alg = ann_alg
        self._ann_index: Optional[
            Union[scann.scann_ops_pybind.ScannSearcher, annoy.AnnoyIndex]
        ] = None

    def build(
        self,
        data: np.ndarray,
        distance_measure: Optional[str] = None,
        scann_num_leaves_scaling: float = 2.5,
        annoy_n_trees: int = 250,
        verbose: int = 1,
    ) -> None:
        """
        Builds the approximate nearest neighbour (ANN) index.

        Parameters
        ----------
        data : np.ndarray
            Data to build the ANN index on.
        distance_measure : str, optional
            Name of the distance measure (or metric). If ann_alg is set to "scann", then
            choose from ["dot_product", "squared_l2"]. Otherwise, choose one of the metrics
            from https://github.com/spotify/annoy. Defaults to "dot_product" if ann_alg is
            set to "scann" and "euclidean" otherwise.
        scann_num_leaves_scaling : float, optional
            Scaling to use when computing the number of leaves for building ScaNN (defaults
            to 2.5). Only has an effect if ann_alg is set to "scann".
        annoy_n_trees : int, optional
            Number of trees to use for building Annoy index (defaults to 250). Only has an
            effect if ann_alg is set to "annoy".
        verbose : int, optional
            Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose). Defaults to 1 (verbose).
        """
        n, d = data.shape
        if verbose == 1:
            print(f"Building ANN index using {self._ann_alg}...")
        if self._ann_alg == "scann":
            if distance_measure is None:
                distance_measure = "dot_product"

            # Compute number of leaves to use when building ScaNN
            scann_num_leaves_order_of_magnitude = int(np.log10(np.sqrt(n)))
            scann_num_leaves_num = 10 ** scann_num_leaves_order_of_magnitude
            scann_num_leaves_scaled = int(
                scann_num_leaves_scaling * scann_num_leaves_num
            )

            # Create and build index
            self._ann_index = (
                scann.scann_ops_pybind.builder(
                    db=data,
                    num_neighbors=1,
                    distance_measure=distance_measure,
                )
                .tree(
                    num_leaves=scann_num_leaves_scaled,
                    num_leaves_to_search=int(scann_num_leaves_scaled / 10),
                    training_sample_size=250000,  # TODO: How to select this number?
                )
                .score_ah(
                    dimensions_per_block=2, anisotropic_quantization_threshold=0.2
                )
                .reorder(
                    reordering_num_neighbors=250
                )  # TODO: How to select this number?
                .build()
            )
        elif self._ann_alg == "annoy":
            if distance_measure is None:
                distance_measure = "euclidean"

            # Add data to index and build it
            self._ann_index = annoy.AnnoyIndex(f=d, metric=distance_measure)
            self._ann_index.set_seed(rng_seed)
            if verbose == 1:
                print("Adding items to index...")
            for i in tqdm(range(n)):
                self._ann_index.add_item(i, data[i])
            if verbose == 1:
                print("Building index...")
            self._ann_index.build(n_trees=annoy_n_trees, n_jobs=-1)
        if verbose == 1:
            print("Done!")

    def save(self, output_path: str) -> None:
        """
        Saves the approximate nearest neighbour instance to disk.

        Parameters
        ----------
        output_path : str
            Output path (directory if ann_alg is "scann", filepath otherwise).
        """
        if self._ann_alg == "scann":
            makedirs(output_path, exist_ok=True)
            self._ann_index.serialize(output_path)
        elif self._ann_alg == "annoy":
            self._ann_index.save(output_path)

    def load(
        self,
        ann_path: str,
        annoy_data_dimensionality: Optional[int] = None,
        annoy_mertic: Optional[str] = None,
        annoy_prefault: bool = False,
    ) -> None:
        """
        Loads an approximate nearest neighbour (ANN) instance from disk.

        Parameters
        ----------
        ann_path : str
            Path of saved ANN instance (directory if ann_alg is "scann", filepath otherwise).
        annoy_data_dimensionality : int, optional
            Dimensionality of data (required if ann_alg is set to "annoy").
        annoy_mertic : str, optional
            Distance metric (required if ann_alg is set to "annoy").
        annoy_prefault : bool, optional
            Whether or not to enable the `prefault` option when loading Annoy index
            (defaults to False).
        """
        if self._ann_alg == "scann":
            self._ann_index = scann.scann_ops_pybind.load_searcher(ann_path)
        elif self._ann_alg == "annoy":
            self._ann_index = annoy.AnnoyIndex(
                f=annoy_data_dimensionality, metric=annoy_mertic
            )
            self._ann_index.load(fn=ann_path, prefault=annoy_prefault)

    def search(
        self,
        query_vector: np.ndarray,
        k_neighbours: int,
        excluded_neighbour_indices: list = [],
        scann_pre_reorder_num_neighbors: Optional[int] = None,
        scann_leaves_to_search: Optional[int] = None,
        return_distances: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Searches for the nearest neighbour of given query vector using approximate nearest
        neighbour instance.

        Parameters
        ----------
        query_vector : np.ndarray
            Vector to query.
        k_neighbours : int
            Number of neighbours to find.
        excluded_neighbour_indices : list, optional
            List of neighbour indices to exclude (defaults to []).
        scann_pre_reorder_num_neighbors : int, optional
            `pre_reorder_num_neighbors` argument sent to ScaNNs search method (defaults to None).
        scann_pre_reorder_num_neighbors : int, optional
            `scann_leaves_to_search` argument sent to ScaNNs search method (defaults to None).
        return_distances : bool, optional
            Whether or not to return distances, in addition to neighbour indices (defaults to False).

        Returns
        -------
        neighbours : np.ndarray
            Nearest neighbouring indices.
        distances : np.ndarray, optional
            Distances to nearest neighbouring data points.
            (Only returned if return_distances is set to True).
        """
        num_excluded_indices = len(excluded_neighbour_indices)
        k_neighbours_search = k_neighbours + num_excluded_indices
        if self._ann_alg == "scann":
            neighbours, distances = self._ann_index.search(
                q=query_vector,
                final_num_neighbors=k_neighbours_search,
                pre_reorder_num_neighbors=scann_pre_reorder_num_neighbors,
                leaves_to_search=scann_leaves_to_search,
            )
        elif self._ann_alg == "annoy":
            annoy_result = self._ann_index.get_nns_by_vector(
                v=query_vector,
                n=k_neighbours_search,
                include_distances=return_distances,
            )
            if return_distances:
                neighbours, distances = annoy_result
            else:
                neighbours = annoy_result

        if num_excluded_indices > 0:
            accepted_indices_filter = np.array(
                [idx not in excluded_neighbour_indices for idx in neighbours]
            )
            neighbours = neighbours[accepted_indices_filter][:k_neighbours]
            if return_distances:
                distances = distances[accepted_indices_filter][:k_neighbours]

        if return_distances:
            return neighbours, distances
        else:
            return neighbours

    def get_distance(self, i: int, j: int) -> float:
        """
        Gets distance between items i and j.

        Parameters
        ----------
        i : int
            Index of first item.
        j : int
            Index of second item.

        Returns
        -------
        i_j_dist : float
            Distance between items i and j.
        """
        if self._ann_alg == "annoy":
            return self._ann_index.get_distance(i, j)
        else:
            raise ValueError(
                "get_distance() method is only available if ANN algorithm is set to 'annoy'."
            )
