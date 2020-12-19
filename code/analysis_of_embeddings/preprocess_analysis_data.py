import argparse
import sys
import zipfile
from os import makedirs
from os.path import isdir, isfile, join

import numpy as np
import pandas as pd
from analysis_utils import preprocess_name

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
        "--custom_data_dir",
        type=str,
        help="Data directory containing custom data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory to save processed data",
    )
    parser.add_argument(
        "--geonames_username",
        type=str,
        default="jtr008",
        help="GeoNames username (create account here: https://www.geonames.org/login)",
    )
    parser.add_argument(
        "--words_filepath",
        type=str,
        help="Filepath of words text file (vocabulary) from word2vec training output",
    )
    return parser.parse_args()


def preprocess_country_info(
    raw_data_dir: str, output_dir: str, geonames_username: str
) -> None:
    """
    Downloads and prepares a .csv file containing all countries and
    its capitals of the world.

    Data is fetched from geonames.org (https://www.geonames.org/)
    and is licenced under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

    Parameters
    ----------
    raw_data_dir : str
        Raw data directory
    output_dir : str
        Directory to save output data.
    geonames_username : str
        GeoNames username
    """
    # Constants
    all_countries_combined_data_url = (
        "https://download.geonames.org/export/dump/allCountries.zip"
    )
    all_countries_raw_data_zip_filepath = join(raw_data_dir, "allCountries.zip")
    all_countries_raw_data_txt_filepath = join(raw_data_dir, "allCountries.txt")
    country_info_csv_data_url = (
        f"https://secure.geonames.org/countryInfoCSV?username={geonames_username}"
    )
    country_info_raw_data_csv_filepath = join(raw_data_dir, "country-info.csv")
    output_filepath = join(output_dir, "country-info.csv")

    # Download raw data
    if not isfile(all_countries_raw_data_zip_filepath):
        print("Downloading raw country data...")
        download_from_url(
            all_countries_combined_data_url, all_countries_raw_data_zip_filepath
        )
        print("Done!")

    if not isfile(all_countries_raw_data_txt_filepath):
        print("Extracting raw data...")
        with zipfile.ZipFile(all_countries_raw_data_zip_filepath, "r") as zip_file:
            zip_file.extractall(raw_data_dir)
        print("Done!")

    if not isfile(country_info_raw_data_csv_filepath):
        print("Downloading country info data...")
        download_from_url(country_info_csv_data_url, country_info_raw_data_csv_filepath)
        print("Done!")

    if not isfile(output_filepath):
        # Load raw data into Pandas DataFrames and join them
        all_countries_info_df = pd.read_csv(
            all_countries_raw_data_txt_filepath,
            sep="\t",
            na_filter=False,
            header=None,
            names=[
                "geonameId",
                "name",
                "asciiname",
                "alternatenames",
                "latitude",
                "longitude",
                "feature class",
                "feature code",
                "country code",
                "cc2",
                "admin1 code",
                "admin2 code",
                "admin3 code",
                "admin4 code",
                "population",
                "elevation",
                "dem",
                "timezone",
                "modification date",
            ],
            usecols=["geonameId", "latitude", "longitude"],
            index_col="geonameId",
        )
        country_info_df = pd.read_csv(
            country_info_raw_data_csv_filepath,
            sep="\t",
            na_filter=False,
            usecols=["name", "capital", "continent", "geonameId"],
        )
        country_info_df = country_info_df.join(
            all_countries_info_df, on="geonameId", how="left"
        )

        # Remove unused GeoNameId column
        country_info_df.drop("geonameId", inplace=True, axis=1)

        # Replace continent codes with names
        continent_code_to_name = {
            "AF": "Africa",
            "AS": "Asia",
            "EU": "Europe",
            "NA": "North America",
            "OC": "Oceania",
            "SA": "South America",
            "AN": "Antarctica",
        }
        country_info_df["continent"] = country_info_df["continent"].apply(
            lambda code: continent_code_to_name[code]
        )

        # Apply preprocessing to country name and capital
        country_info_df["name"] = country_info_df["name"].apply(preprocess_name)
        country_info_df["capital"] = country_info_df["capital"].apply(preprocess_name)

        # Save to file
        country_info_df.to_csv(output_filepath, index=False)


def preprocess_custom_data(custom_data_dir: str, output_dir: str) -> None:
    """
    Preprocesses custom data files

    Parameters
    ----------
    custom_data_dir : str
        Data directory containing custom data files.
    output_dir : str
        Directory to save output data.
    """
    custom_data_filenames = [
        "word2vec_analysis_category_of_words.xlsx"  # Analysing category of words
    ]
    custom_data_filepaths = [join(custom_data_dir, fn) for fn in custom_data_filenames]

    # Parsing excel files
    for custom_data_filepath in custom_data_filepaths:
        excel_file = pd.ExcelFile(custom_data_filepath)
        excel_sheet_dfs = [
            (sheet_name, excel_file.parse(sheet_name))
            for sheet_name in excel_file.sheet_names
        ]

        # Iterate over all sheets in excel file
        for sheet_name, custom_data_df in excel_sheet_dfs:

            # Apply preprocessing to names
            custom_data_df["Name"] = custom_data_df["Name"].apply(preprocess_name)

            # Save to file
            custom_data_df.to_csv(join(output_dir, f"{sheet_name}.csv"), index=False)


def preprocess_word_cluster_groups(
    raw_data_dir: str, output_dir: str, words_filepath: str
) -> None:
    """
    Preprocesses word cluster groups

    Parameters
    ----------
    raw_data_dir : str
        Raw data directory
    output_dir : str
        Directory to save output data.
    words_filepath: str
        Filepath of words text file (vocabulary) from word2vec training output
    """
    # Load words from vocabulary
    with open(words_filepath, "r") as words_file:
        words = np.array(words_file.read().split("\n"))
    word_to_int = {word: i for i, word in enumerate(words)}  # Word integer lookup table

    # -- Numbers --
    numbers_list = []
    numbers_list.extend(list(range(100)))
    numbers_list.extend([100, 1000, 1000000, 1000000000, 1000000000000])
    numbers_textual_reprs = []
    for number in numbers_list:
        for num in preprocess_text(str(number)):
            if num != "and" and num not in numbers_textual_reprs:
                numbers_textual_reprs.append(num)
    number_words_in_vocab = [
        num_word for num_word in numbers_textual_reprs if num_word in word_to_int
    ]
    with open(join(output_dir, "numbers.txt"), "w") as words_output_file:
        for i, word in enumerate(number_words_in_vocab):
            if i > 0:
                words_output_file.write("\n")
            words_output_file.write(f"{word}")

    # -- Names --
    names_data_url = "https://www.ssa.gov/oact/babynames/names.zip"
    names_raw_zip_filepath = join(raw_data_dir, "names.zip")
    names_raw_zip_dir = join(raw_data_dir, "names")
    names_year = 2019
    names_raw_filepath = join(names_raw_zip_dir, f"yob{names_year}.txt")
    names_output_filepath = join(output_dir, "names.csv")

    # Download raw data
    if not isfile(names_raw_zip_filepath):
        print("Downloading names data...")
        download_from_url(names_data_url, names_raw_zip_filepath)
        print("Done!")

    if not isdir(names_raw_zip_dir):
        print("Extracting raw data...")
        with zipfile.ZipFile(names_raw_zip_filepath, "r") as zip_file:
            zip_file.extractall(names_raw_zip_dir)
        print("Done!")

    word_in_vocab = lambda word: word in word_to_int
    if not isfile(names_output_filepath):
        names_raw_df = pd.read_csv(
            names_raw_filepath,
            delimiter=",",
            header=None,
            names=["name", "gender", "count"],
        )
        names_raw_df["name"] = names_raw_df["name"].str.lower()
        names_raw_df = names_raw_df[names_raw_df["name"].apply(word_in_vocab)]
        names_raw_df.to_csv(names_output_filepath, index=False)


def preprocess_analysis_data(
    raw_data_dir: str,
    custom_data_dir: str,
    output_dir: str,
    geonames_username: str,
    words_filepath: str,
) -> None:
    """
    Preprocesses data for analysing word embeddings

    Parameters
    ----------
    raw_data_dir : str
        Raw data directory
    custom_data_dir : str
        Data directory containing custom data files
    output_dir : str
        Directory to save analysis data.
    geonames_username : str
        GeoNames username (create account here: https://www.geonames.org/login)
    words_filepath : str
        Filepath of words text file (vocabulary) from word2vec training output
    """
    # Ensure raw data/output directories exist
    makedirs(raw_data_dir, exist_ok=True)
    makedirs(output_dir, exist_ok=True)

    print("-- Country info --")
    preprocess_country_info(raw_data_dir, output_dir, geonames_username)
    print("Done!")

    print("-- Custom data -- ")
    preprocess_custom_data(
        custom_data_dir,
        output_dir,
    )
    print("Done!")

    print("-- Word cluster groups --")
    preprocess_word_cluster_groups(
        raw_data_dir,
        output_dir,
        words_filepath,
    )
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    preprocess_analysis_data(
        raw_data_dir=args.raw_data_dir,
        custom_data_dir=args.custom_data_dir,
        output_dir=args.output_dir,
        geonames_username=args.geonames_username,
        words_filepath=args.words_filepath,
    )
