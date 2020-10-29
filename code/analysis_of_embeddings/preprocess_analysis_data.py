import argparse
import sys
import zipfile
from os import makedirs
from os.path import isdir, isfile, join

import pandas as pd

sys.path.append("..")

from text_preprocessing_utils import preprocess_text
from utils import download_from_url


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
        "--raw_data_dir",
        type=str,
        default="",
        help="Directory where raw data will be downloaded/extracted to",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory to save processed data",
    )
    return parser.parse_args()


def preprocess_country_capitals(raw_data_dir: str, output_dir: str) -> None:
    """
    Downloads and prepares a .csv file containing all countries and
    its capitals of the world.

    Data is downloaded from SimpleMaps.com (https://simplemaps.com/data/world-cities)
    and is licenced under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

    Parameters
    ----------
    raw_data_dir : str
        Raw data directory
    output_dir : str
        Directory to save output data.
    """
    # Constants
    data_url = "https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.71.zip"
    raw_data_zip_filepath = join(raw_data_dir, "world-cities.zip")
    raw_data_extracted_zip_dir = join(raw_data_dir, "world-cities")
    raw_data_world_cities_csv = join(raw_data_extracted_zip_dir, "worldcities.csv")
    output_filepath = join(output_dir, "country_capitals.csv")

    # Download raw zip
    if not isfile(raw_data_zip_filepath):
        print(f"Downloading raw world-cities data...")
        download_from_url(url=data_url, destination_filepath=raw_data_zip_filepath)
        print("Done!")

    # Extract raw data if not present
    if not isdir(raw_data_extracted_zip_dir):
        print("Extracting raw data...")
        with zipfile.ZipFile(raw_data_zip_filepath, "r") as zip_file:
            zip_file.extractall(raw_data_extracted_zip_dir)
        print("Done!")

    # Extract capital cities only
    world_cities_df = pd.read_csv(raw_data_world_cities_csv)
    world_cities_df_capitals = world_cities_df[world_cities_df["capital"] == "primary"]

    # We only want the columns: country, city, lat, lng
    world_cities_df_capitals = world_cities_df_capitals[["country", "city", "lat", "lng"]]

    # Apply preprocessing to country/capital names
    preprocess_name = lambda name: " ".join(preprocess_text(name)).replace(" ", "_")
    world_cities_df_capitals["country"] = world_cities_df_capitals["country"].apply(
        preprocess_name
    )
    world_cities_df_capitals["city"] = world_cities_df_capitals["city"].apply(
        preprocess_name
    )

    # Save to file
    world_cities_df_capitals.to_csv(output_filepath, index=False)


def preprocess_analysis_data(raw_data_dir: str, output_dir: str) -> None:
    """
    Preprocesses data for analysing word embeddings

    Parameters
    ----------
    raw_data_dir : str
        Raw data directory
    output_dir : str
        Directory to save analysis data.
    """
    # Ensure raw data/output directories exist
    makedirs(raw_data_dir, exist_ok=True)
    makedirs(output_dir, exist_ok=True)

    print("-- Country/capitals --")
    preprocess_country_capitals(raw_data_dir, output_dir)
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    preprocess_analysis_data(raw_data_dir=args.raw_data_dir, output_dir=args.output_dir)
