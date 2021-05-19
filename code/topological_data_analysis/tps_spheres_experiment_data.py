import argparse
import sys
from os import makedirs
from os.path import isfile, join

import numpy as np
from tqdm import tqdm

rng_seed = 399
np.random.seed(rng_seed)

sys.path.append("..")

from topological_data_analysis.tda_utils import generate_points_in_spheres  # noqa: E402
from topological_data_analysis.topological_polysemy import (  # noqa: E402
    tps_multiple_point_cloud,
)


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


def prepare_spheres_data(noisy_spheres: bool, output_dir: str) -> list:
    """
    Prepares spheres data.

    Parameters
    ----------
    noisy_spheres : bool
        Whether or not to create noisy sphere data
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
    sphere_sample_num_intervals = 20
    sphere_sample_size = 1000
    sphere_points_data_filepaths = []
    sphere_noisy_str = "_noisy" if noisy_spheres else ""
    for sphere_dimensionality in sphere_dimensionalities:
        print(f"Sphere dimensionality: {sphere_dimensionality}")
        sphere_points_data_filepath = join(
            output_dir,
            f"sphere_points_data_{sphere_dimensionality}{sphere_noisy_str}.npy",
        )
        sampled_sphere_points_data_filepath = join(
            output_dir,
            f"sampled_sphere_points_data_{sphere_dimensionality}{sphere_noisy_str}.npy",
        )
        sphere_points_data_filepaths.append(
            (
                sphere_dimensionality,
                sphere_points_data_filepath,
                sampled_sphere_points_data_filepath,
            )
        )
        if isfile(sphere_points_data_filepath) and isfile(
            sampled_sphere_points_data_filepath
        ):
            continue
        print("Generating points...")
        sphere_points, sphere_point_labels = generate_points_in_spheres(
            num_points=point_in_each_sphere_gen,
            sphere_dimensionality=sphere_dimensionality,
            space_dimensionality=space_dimensionality,
            create_intersection_point=True,
            noisy_spheres=noisy_spheres,
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

        # Sample sphere points from intervals, sorted by distance to intersection point
        sampled_sphere_point_indices = [
            distances_to_intersection_point_sorted_indices[0]  # <-- Intersection point
        ]
        interval_width = (sphere_points.shape[0] - 1) // sphere_sample_num_intervals
        for i in range(sphere_sample_num_intervals):
            min_interval_idx = max(i * interval_width, 1)
            max_interval_idx = (i + 1) * interval_width
            interval_indices = distances_to_intersection_point_sorted_indices[
                np.arange(min_interval_idx, max_interval_idx)
            ]
            sampled_indices = np.random.choice(
                interval_indices, size=sphere_sample_size, replace=False
            )
            sampled_sphere_point_indices.extend(sampled_indices)
        sampled_sphere_point_indices = np.array(sampled_sphere_point_indices)

        sphere_points_data = np.column_stack(
            (
                sphere_points,
                sphere_point_labels,
                distances_to_intersection_point,
            )
        )
        sampled_sphere_points_data = np.column_stack(
            (
                sphere_points[sampled_sphere_point_indices],
                sphere_point_labels[sampled_sphere_point_indices],
                distances_to_intersection_point[sampled_sphere_point_indices],
                sampled_sphere_point_indices,
            )
        )

        # Save data
        print("Saving data...")
        np.save(sphere_points_data_filepath, sphere_points_data)
        np.save(sampled_sphere_points_data_filepath, sampled_sphere_points_data)

        # Free resources
        del sphere_points_data
        del sphere_points
        del sphere_point_labels
        del distances_to_intersection_point
        del sampled_sphere_point_indices
        del sampled_sphere_points_data

    return sphere_points_data_filepaths


def compute_tps_scores(
    sphere_data_filepaths: list, tps_neighbourhood_size: int, output_dir: str
) -> None:
    """
    Computes TPS scores of sphere data.

    Parameters
    ----------
    sphere_data_filepaths : list
        List of sphere dimensionalities and data filepaths.
    tps_neighbourhood_size : int
        TPS neighbourhood size.
    output_dir : str
        Output directory where processed files will be saved to.
    """
    for (
        sphere_dimensionality,
        sphere_points_filepath,
        sphere_point_indices_filepath,
    ) in sphere_data_filepaths:

        # Check if TPS scores are computed already
        tps_scores_filepath = join(
            output_dir,
            f"sphere_points_data_{sphere_dimensionality}_tps_{tps_neighbourhood_size}_scores.npy",
        )
        if isfile(tps_scores_filepath):
            continue

        print(f"Sphere dimensionality: {sphere_dimensionality}")
        print("Loading data...")
        sphere_points_data = np.load(sphere_points_filepath)
        sphere_points = sphere_points_data[:, :-2]
        sphere_points_normalized = sphere_points / np.linalg.norm(
            sphere_points, axis=1
        ).reshape(-1, 1)
        sampled_sphere_points_data = np.load(sphere_point_indices_filepath)
        sampled_sphere_point_indices = sampled_sphere_points_data[:, -1].astype(int)
        print("Done!")

        # Compute TPS scores
        print("Computing TPS...")
        tps_scores_point_in_spheres = tps_multiple_point_cloud(
            point_indices=sampled_sphere_point_indices,
            neighbourhood_size=tps_neighbourhood_size,
            point_cloud_normalized=sphere_points_normalized,
            return_persistence_diagram=False,
            n_jobs=-1,
            progressbar_enabled=True,
        )
        np.save(tps_scores_filepath, tps_scores_point_in_spheres)

        # Free resources
        del sphere_points_data
        del sampled_sphere_points_data
        del sampled_sphere_point_indices
        del sphere_points
        del sphere_points_normalized
        del tps_scores_point_in_spheres


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
    for is_noisy in [False, True]:
        print(f"Noisy: {is_noisy}")
        noisy_str = "_noisy" if is_noisy else ""
        experiment_output_dir = join(output_dir, f"tps_spheres_experiment{noisy_str}")
        makedirs(experiment_output_dir, exist_ok=True)
        print("Preparing spheres data...")
        sphere_data_filepaths = prepare_spheres_data(noisy_spheres=is_noisy, output_dir=experiment_output_dir)
        print("Computing TPS scores...")
        compute_tps_scores(
            tps_neighbourhood_size=tps_neighbourhood_size,
            sphere_data_filepaths=sphere_data_filepaths,
            output_dir=experiment_output_dir,
        )


if __name__ == "__main__":
    args = parse_args()
    tps_spheres_experiment_data_preprocessing(
        tps_neighbourhood_size=args.tps_neighbourhood_size, output_dir=args.output_dir
    )
