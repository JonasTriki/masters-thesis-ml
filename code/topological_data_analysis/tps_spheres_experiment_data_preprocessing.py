import argparse
import sys
from os import makedirs
from os.path import isfile, join

import numpy as np
from tqdm import tqdm

rng_seed = 399
np.random(rng_seed)
sys.path.append("..")

from topological_data_analysis.tda_utils import generate_points_in_spheres  # noqa: E402
from topological_data_analysis.topological_polysemy import tps_point_cloud  # noqa: E402
from utils import normalize_array  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parses arguments sent to the python script.

    Returns
    -------
    parsed_args : argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tps_neighbourhood_size",
        type=int,
        default="",
        help="TPS neighbourhood size",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory where processed files will be saved to",
    )
    return parser.parse_args()


def prepare_spheres_data(output_dir: str) -> list:
    """
    Prepares spheres data.

    Parameters
    ----------
    output_dir : str
        Output directory where processed files will be saved to.

    Returns
    -------
    sphere_data_filepaths : list
        List of sphere data filepaths.
    """
    # Generate sphere data
    sphere_point_shift = 2
    space_dimensionality = 300
    sphere_dimensionalities = [2, 3, 4, 5, 10, 20, 50, 300]
    point_in_each_sphere_gen = 1000000
    sphere_sample_step = 100
    num_points_in_spheres = int(point_in_each_sphere_gen / sphere_sample_step) * 2 + 1
    points_frac_closest_to_intersection = 0.25
    points_max_dim_closest_to_intersection = 5
    sampled_sphere_points_data_filepaths = []
    for sphere_dimensionality in sphere_dimensionalities:
        print(f"Sphere dimensionality: {sphere_dimensionality}")
        sampled_sphere_points_data_filepath = join(
            output_dir,
            f"sphere_points_data_{sphere_dimensionality}.npy",
        )
        sampled_sphere_points_data_filepaths.append(
            (sphere_dimensionality, sampled_sphere_points_data_filepath)
        )
        if isfile(sampled_sphere_points_data_filepath):
            continue
        print("Generating points...")
        sphere_points, sphere_point_labels = generate_points_in_spheres(
            num_points=point_in_each_sphere_gen,
            sphere_dimensionality=sphere_dimensionality,
            space_dimensionality=space_dimensionality,
            create_intersection_point=True,
            random_state=rng_seed,
        )
        sphere_point_shift_arr = np.repeat(sphere_point_shift, space_dimensionality)
        sphere_points += sphere_point_shift_arr
        shpere_points_intersection = sphere_point_shift_arr

        distances_to_intersection_point = np.zeros(sphere_points.shape[0])
        print("Computing distances...")
        for i, sphere_point in enumerate(tqdm(sphere_points)):
            distances_to_intersection_point[i] = np.linalg.norm(
                sphere_point - shpere_points_intersection
            )
        distances_to_intersection_point_sorted_indices = np.argsort(
            distances_to_intersection_point
        )

        # Get sampled sphere points
        num_closest_to_intersection = (
            num_points_in_spheres
            * points_frac_closest_to_intersection
            * min(sphere_dimensionality, points_max_dim_closest_to_intersection)
            / points_max_dim_closest_to_intersection
        )
        sampled_sphere_point_indices = []
        sampled_sphere_point_indices.extend(
            distances_to_intersection_point_sorted_indices[:num_closest_to_intersection]
        )
        sampled_sphere_point_indices.extend(
            distances_to_intersection_point_sorted_indices[
                num_closest_to_intersection::sphere_sample_step
            ]
        )
        sampled_sphere_point_indices = np.array(sampled_sphere_point_indices)

        sampled_sphere_points = sphere_points[sampled_sphere_point_indices]
        sampled_sphere_point_labels = sphere_point_labels[sampled_sphere_point_indices]
        sampled_sphere_point_distances = distances_to_intersection_point[
            sampled_sphere_point_indices
        ]
        sampled_sphere_points_data = np.column_stack(
            (
                sampled_sphere_points,
                sampled_sphere_point_labels,
                sampled_sphere_point_distances,
            )
        )

        # Save data
        np.save(sampled_sphere_points_data_filepath, sampled_sphere_points_data)

    return sampled_sphere_points_data_filepaths


def compute_tps_scores(
    sphere_data_filepaths: list, tps_neighbourhood_size: int, output_dir: str
) -> None:
    """
    Prepares spheres data.

    Parameters
    ----------
    sphere_data_filepaths : list
        List of sphere dimensionalities and data filepaths.
    tps_neighbourhood_size : int
        TPS neighbourhood size.
    output_dir : str
        Output directory where processed files will be saved to.
    """
    for sphere_dimensionality, filepath in sphere_data_filepaths:
        sampled_sphere_points_data = np.load(filepath)
        sphere_points = sampled_sphere_points_data[:, :-2]
        sphere_points_normalized = normalize_array(sphere_points)

        # Check if TPS scores are computed already
        tps_scores_filepath = join(
            output_dir,
            f"sphere_points_data_{sphere_dimensionality}_tps_{tps_neighbourhood_size}_scores.npy",
        )
        if isfile(tps_scores_filepath):
            continue

        # Compute TPS
        num_total_sphere_points = len(sampled_sphere_points_data)
        tps_scores_point_in_spheres = np.zeros(num_total_sphere_points)
        for point_index in tqdm(range(num_total_sphere_points)):
            tps_score = tps_point_cloud(
                point_index=point_index,
                neighbourhood_size=tps_neighbourhood_size,
                point_cloud_normalized=sphere_points_normalized,
                return_persistence_diagram=False,
            )
            tps_scores_point_in_spheres[point_index] = tps_score
        np.save(tps_scores_filepath, tps_scores_point_in_spheres)


def tps_spheres_experiment_data_preprocessing(
    tps_neighbourhood_size: int, output_dir: str
) -> None:
    """
    Preprocesses data for the TPS spheres experiment.

    Parameters
    ----------
    tps_neighbourhood_size : int
        TPS neighbourhood size.
    output_dir : str
        Output directory where processed files will be saved to.
    """
    output_dir = join(output_dir, "tps_spheres_experiment")
    makedirs(output_dir, exist_ok=True)
    print("Preparing spheres data...")
    sphere_data_filepaths = prepare_spheres_data(output_dir=output_dir)
    print("Computing TPS scores...")
    compute_tps_scores(
        tps_neighbourhood_size=tps_neighbourhood_size,
        sphere_data_filepaths=sphere_data_filepaths,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    args = parse_args()
    tps_spheres_experiment_data_preprocessing(
        tps_neighbourhood_size=args.tps_neighbourhood_size, output_dir=args.output_dir
    )
