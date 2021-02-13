import numpy as np
import ripser_plusplus_python as rpp_py


def run_ripser_plus_plus(point_cloud: np.ndarray, max_dim: int) -> dict:
    """
    Runs Ripser++ on the given point cloud (computes hersistent homology up
    to dimension `max_dim`).

    Parameters
    ----------
    point_cloud : np.ndarray
        Point cloud to run Ripser++ on.
    max_dim : int
        Maximal persistent homology dimensionality.

    Returns
    -------
    barcodes : dict
        Barcodes dictionary, where the key is the homolohy dimension and the values
        are the birth/death diagrams.
    """
    return rpp_py.run(f"--dim {max_dim} --format point-cloud", point_cloud)


if __name__ == "__main__":

    # Testing
    num_pnts = 10
    pnt_dimension = 100
    rand_point_cloud = np.random.random((num_pnts, pnt_dimension))
    barcodes_dict = run_ripser_plus_plus(point_cloud=rand_point_cloud, max_dim=2)
    barcodes = list(barcodes_dict.values())
    print(len(barcodes), barcodes)
