import argparse
import sys
from os import makedirs
from os.path import isfile, join
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from persim import PersistenceImager
from skdim import id as est_ids
from skdim._commonfuncs import GlobalEstimator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append("..")

from approx_nn import ApproxNN  # noqa: E402
from topological_data_analysis.geometric_anomaly_detection import (  # noqa: E402
    compute_gad,
)
from topological_data_analysis.topological_polysemy import tps_multiple  # noqa: E402
from word_embeddings.word2vec import load_model_training_output  # noqa: E402

rng_seed = 399
np.random.seed(rng_seed)


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
        "--model_dir",
        type=str,
        default="",
        help="Directory of the model to load",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="word2vec",
        help="Name of the trained word2vec model",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="enwiki",
        help="Name of the dataset the model is trained on",
    )
    parser.add_argument(
        "--id_estimation_num_neighbours",
        nargs="+",
        default=["25", "50", "100", "150", "200"],
        help="Number of neighbours to use when estimating intrinsic dimension for each word",
    )
    parser.add_argument(
        "--semeval_2010_14_word_senses_filepath",
        type=str,
        default="",
        help="Filepath of SemEval-2010 task 14 word senses joblib dict",
    )
    parser.add_argument(
        "--tps_neighbourhood_sizes",
        nargs="+",
        default=["10", "40", "50", "60", "100"],
        help="List of TPS neighbourhood sizes",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="",
        help="Directory where raw data will be saved to",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory",
    )
    return parser.parse_args()


def create_word_meaning_model_data_features(
    target_words: list,
    word_to_int: dict,
    tps_scores: dict,
    tps_pds: dict,
    tps_neighbourhood_sizes: list,
    words_estimated_ids: dict,
    words_to_meanings: dict,
    gad_categories: dict,
    gad_features_dict: dict,
    gad_features_pd_vecs_dict: dict,
) -> pd.DataFrame:
    """
    Creates a Pandas DataFrame with columns containing data features for the supervised
    word meaning estimation task.

    Parameters
    ----------
    target_words : list
        List of words to use when creating features.
    word_to_int : dict
        Dictionary mapping from a word to its integer representation.
    tps_scores : dict
        Dictionary containing TPS scores, where the keys are neighbourhood sizes (n) and
        values are TPS_n scores.
    tps_pds : dict
        Dictionary containing TPS persistence diagrams, where the keys are neighbourhood
        sizes (n) and values are TPS_n scores.
    tps_neighbourhood_sizes : list
        List of TPS neighbourhood sizes to use.
    words_estimated_ids : dict
        Dictionary containing estimated intrinsic dimension (ID) for all words using different
        estimators.
    words_to_meanings : dict
        Dictionary mapping a word to its number of meanings; serves as the labels (y) for
        the supervised task.
    gad_categories : dict
        Dictionary containing GAD categories and its unique indices.
    gad_features_dict : dict
        Dictionary containing features from GAD.
    gad_features_pd_vecs_dict : dict
        Dictionary containing vectorized persistence diagrams from GADs features.

    Returns
    -------
    data_features_df : pd.DataFrame
        DataFrame with data for the word meaning estimation task.
    """
    data_features = {
        "word": [],
        "y": [],
    }
    for n_size in tps_neighbourhood_sizes:
        data_features[f"X_tps_{n_size}"] = []
        data_features[f"X_tps_{n_size}_pd_max"] = []
        data_features[f"X_tps_{n_size}_pd_avg"] = []
        data_features[f"X_tps_{n_size}_pd_std"] = []
    for id_estimator_name in words_estimated_ids.keys():
        data_features[f"X_estimated_id_{id_estimator_name}"] = []
    for gad_config in gad_features_dict.keys():
        for gad_category in gad_categories.keys():
            data_features[f"X_{gad_config}_{gad_category}"] = []
        gad_features_pd_vecs = gad_features_pd_vecs_dict[gad_config]
        for pd_feature_idx in range(gad_features_pd_vecs.shape[1]):
            data_features[f"X_{gad_config}_pd[{pd_feature_idx}]"] = []

    for target_word in tqdm(target_words):
        word_int = word_to_int[target_word]

        # -- Fill in DataFrame --
        data_features["word"].append(target_word)  # Word
        data_features["y"].append(words_to_meanings[target_word])  # Number of meanings

        # ID estimates
        for id_estimator_name, id_estimates in words_estimated_ids.items():
            data_features[f"X_estimated_id_{id_estimator_name}"].append(
                id_estimates[word_int]
            )  # Estimated ID

        # TPS features
        for n_size in tps_neighbourhood_sizes:
            data_features[f"X_tps_{n_size}"].append(tps_scores[n_size][word_int])
            tps_pd_zero_dim_deaths = tps_pds[n_size][word_int][:, 1]
            data_features[f"X_tps_{n_size}_pd_max"].append(tps_pd_zero_dim_deaths.max())
            data_features[f"X_tps_{n_size}_pd_avg"].append(
                tps_pd_zero_dim_deaths.mean()
            )
            data_features[f"X_tps_{n_size}_pd_std"].append(tps_pd_zero_dim_deaths.std())

        # Features from GAD (P_man, P_int, P_bnd)
        for gad_config, gad_features in gad_features_dict.items():
            for gad_category, gad_feature_val in zip(
                gad_categories.keys(), gad_features[word_int]
            ):
                data_features[f"X_{gad_config}_{gad_category}"].append(gad_feature_val)
            gad_features_pd_vecs = gad_features_pd_vecs_dict[gad_config]
            for pd_feature_idx in range(gad_features_pd_vecs.shape[1]):
                pd_feature_val = gad_features_pd_vecs[word_int, pd_feature_idx]
                data_features[f"X_{gad_config}_pd[{pd_feature_idx}]"].append(
                    pd_feature_val
                )

    # Create df and return it
    data_features_df = pd.DataFrame(data_features)

    return data_features_df


def prepare_num_word_meanings_supervised_data(
    model_dir: str,
    model_name: str,
    dataset_name: str,
    id_estimation_num_neighbours: list,
    semeval_2010_14_word_senses_filepath: str,
    tps_neighbourhood_sizes: list,
    raw_data_dir: str,
    output_dir: str,
) -> None:
    """
    Prepares data for the supervised word meanings prediction task.

    Parameters
    ----------
    model_dir : str
        Directory of the model to load.
    model_name : str
        Name of the trained word2vec model.
    dataset_name : str
        Name of the dataset the model is trained on.
    id_estimation_num_neighbours : list
        Number of neighbours to use when estimating intrinsic dimension for each word
    semeval_2010_14_word_senses_filepath : str
        Filepath of SemEval-2010 task 14 word senses joblib dict.
    tps_neighbourhood_sizes : list
        List of TPS neighbourhood sizes.
    raw_data_dir : str
        Directory where raw data will be saved to.
    output_dir: str
        Output directory.
    """
    # Convert list arguments to int
    tps_neighbourhood_sizes = [int(n_size) for n_size in tps_neighbourhood_sizes]
    id_estimation_num_neighbours = [
        int(num_neighbours) for num_neighbours in id_estimation_num_neighbours
    ]

    # Prepare directory constants and create raw data dir for caching data files
    task_id = f"wme_{model_name}_{dataset_name}"  # wme = word meaning estimation
    task_raw_data_dir = join(raw_data_dir, task_id)
    task_raw_data_tps_dir = join(task_raw_data_dir, "tps")
    makedirs(task_raw_data_dir, exist_ok=True)

    # Load word embeddings from model
    print("Loading word embeddings...")
    w2v_training_output = load_model_training_output(
        model_training_output_dir=model_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        return_normalized_embeddings=True,
        return_scann_instance_filepath=True,
    )
    last_embedding_weights_normalized = w2v_training_output[
        "last_embedding_weights_normalized"
    ]
    last_embedding_weights_scann_instance_filepath = w2v_training_output[
        "last_embedding_weights_scann_instance_filepath"
    ]
    words = w2v_training_output["words"]
    word_to_int = w2v_training_output["word_to_int"]
    print("Done!")

    # Prepare SemEval-2010 task 14 data
    semeval_2010_14_word_senses = joblib.load(semeval_2010_14_word_senses_filepath)
    semeval_target_words = np.array(list(semeval_2010_14_word_senses["all"].keys()))
    semeval_target_words_in_vocab_filter = [
        i for i, word in enumerate(semeval_target_words) if word in word_to_int
    ]
    semeval_target_words_in_vocab = semeval_target_words[
        semeval_target_words_in_vocab_filter
    ]
    semeval_gs_clusters = np.array(list(semeval_2010_14_word_senses["all"].values()))
    semeval_gs_clusters_in_vocab = semeval_gs_clusters[
        semeval_target_words_in_vocab_filter
    ]
    semeval_2010_14_word_senses_in_vocab = {
        word: gs_meanings
        for word, gs_meanings in zip(
            semeval_target_words_in_vocab, semeval_gs_clusters_in_vocab
        )
    }

    # (1) -- Find words in Wordnet that are in the word2vec model's vocabulary --
    words_to_num_meanings_filepath = join(
        task_raw_data_dir, "words_to_num_meanings.joblib"
    )
    if not isfile(words_to_num_meanings_filepath):
        words_to_num_meanings = semeval_2010_14_word_senses_in_vocab.copy()
        print("Finding words in vocabulary with #Wordnet synsets > 0")
        for word in tqdm(words):
            if word in semeval_target_words_in_vocab:
                continue
            num_synsets = len(wn.synsets(word))
            if num_synsets > 0:
                words_to_num_meanings[word] = num_synsets
        joblib.dump(words_to_num_meanings, words_to_num_meanings_filepath)
    else:
        words_to_num_meanings = joblib.load(words_to_num_meanings_filepath)
        print("Loaded words_to_num_meanings!")
    data_words = np.array(list(words_to_num_meanings.keys()))
    data_words_no_semeval = [
        word for word in data_words if word not in semeval_target_words_in_vocab
    ]
    data_word_to_int = {word: i for i, word in enumerate(data_words)}

    # Filter out word embeddings using Wordnet words (data_words)
    data_words_to_full_vocab_ints = np.array([word_to_int[word] for word in data_words])

    # (2) -- Estimate the intrinsic dimension (ID) for each word vector --
    words_estimated_ids_dir = join(task_raw_data_dir, "estimated_ids")
    id_estimators: List[Tuple[str, GlobalEstimator, dict]] = [
        ("lpca", est_ids.lPCA, {}),
        ("knn", est_ids.KNN, {}),
        ("twonn", est_ids.TwoNN, {}),
        ("mle", est_ids.MLE, {}),
        ("tle", est_ids.TLE, {}),
    ]
    makedirs(words_estimated_ids_dir, exist_ok=True)
    for id_estimator_name, id_estimator_cls, id_estimator_params in id_estimators:
        for num_neighbours in id_estimation_num_neighbours:
            estimated_ids_filepath = join(
                words_estimated_ids_dir, f"{id_estimator_name}_{num_neighbours}.npy"
            )
            if isfile(estimated_ids_filepath):
                continue

            print(
                f"Estimating IDs using {id_estimator_cls.__name__} with {num_neighbours} neighbours..."
            )
            id_estimator = id_estimator_cls(**id_estimator_params)
            estimated_ids = id_estimator.fit_predict_pw(
                X=last_embedding_weights_normalized[data_words_to_full_vocab_ints],
                n_neighbors=num_neighbours,
                n_jobs=-1,
            )
            # estimated_ids = estimated_ids_full[data_words_to_full_vocab_ints]

            print("Done! Saving to file...")
            np.save(estimated_ids_filepath, estimated_ids)

    # (3) -- Compute TPS_n for train/test words --
    makedirs(task_raw_data_tps_dir, exist_ok=True)
    tps_scores_filepaths = [
        join(task_raw_data_tps_dir, f"tps_{tps_neighbourhood_size}_scores.npy")
        for tps_neighbourhood_size in tps_neighbourhood_sizes
    ]
    tps_pds_filepaths = [
        join(task_raw_data_tps_dir, f"tps_{tps_neighbourhood_size}_pds.npy")
        for tps_neighbourhood_size in tps_neighbourhood_sizes
    ]
    for tps_neighbourhood_size, tps_scores_filepath, tps_pds_filepath in zip(
        tps_neighbourhood_sizes, tps_scores_filepaths, tps_pds_filepaths
    ):
        if isfile(tps_scores_filepath) and isfile(tps_pds_filepath):
            continue
        print(
            f"Computing TPS scores using neighbourhood size {tps_neighbourhood_size}..."
        )

        # Load ScaNN instance
        scann_instance = ApproxNN(ann_alg="scann")
        scann_instance.load(ann_path=last_embedding_weights_scann_instance_filepath)

        # Compute TPS
        tps_scores_ns, tps_pds_ns = tps_multiple(
            target_words=data_words,
            word_to_int=word_to_int,
            neighbourhood_size=tps_neighbourhood_size,
            word_embeddings_normalized=last_embedding_weights_normalized,
            ann_instance=scann_instance,
            return_persistence_diagram=True,
            n_jobs=-1,
            progressbar_enabled=True,
        )

        # Save result
        print("Saving TPS result...")
        np.save(tps_scores_filepath, tps_scores_ns)
        np.save(tps_pds_filepath, tps_pds_ns)
        print("Done!")

        # Free resources
        del scann_instance

    # (4) -- Compute GAD --
    gad_dir = join(task_raw_data_dir, "gad")
    makedirs(gad_dir, exist_ok=True)
    gad_params = [
        (25, 250),
        (25, 500),
        (50, 250),
        (50, 550),
        (50, 750),
        (50, 1000),
        (100, 1000),
        (100, 1250),
        (100, 1500),
        (100, 1750),
        (100, 2000),
        (150, 1000),
        (150, 1250),
        (150, 1500),
        (150, 1750),
        (150, 2000),
        (150, 1000),
        (200, 1250),
        (200, 1500),
        (200, 1750),
        (200, 2000),
    ]
    gad_categories = {"P_man": 0, "P_int": 1, "P_bnd": 2}
    for inner_param, outer_param in gad_params:
        gad_id = f"gad_knn_{inner_param}_{outer_param}"

        gad_filepath = join(gad_dir, f"{gad_id}.joblib")
        if isfile(gad_filepath):
            continue
        print(f"-- {gad_id} -- ")

        # Load ScaNN instance
        approx_nn = ApproxNN(ann_alg="scann")
        approx_nn.load(ann_path=last_embedding_weights_scann_instance_filepath)

        # Compute features
        gad_result = compute_gad(
            data_points=last_embedding_weights_normalized,
            data_point_ints=data_words_to_full_vocab_ints,
            manifold_dimension=2,
            data_points_approx_nn=approx_nn,
            use_knn_annulus=True,
            knn_annulus_inner=inner_param,
            knn_annulus_outer=outer_param,
            return_annlus_persistence_diagrams=True,
            progressbar_enabled=True,
            n_jobs=-1,
        )
        print(
            "P_man:",
            len(gad_result["P_man"]),
            "P_int:",
            len(gad_result["P_int"]),
            "P_bnd:",
            len(gad_result["P_bnd"]),
        )
        joblib.dump(gad_result, gad_filepath, protocol=4)

        # Free resources
        del approx_nn

    # (5) -- Create features from GAD result to speed up combining of data --
    gad_features_dir = join(task_raw_data_dir, "gad_features")
    makedirs(gad_features_dir, exist_ok=True)
    for inner_param, outer_param in gad_params:
        gad_id = f"gad_knn_{inner_param}_{outer_param}"

        gad_features_filepath = join(gad_features_dir, f"{gad_id}.npy")
        if isfile(gad_features_filepath):
            continue
        print(f"Creating GAD features for {gad_id}...")

        # Load GAD result
        gad_result_filepath = join(gad_dir, f"{gad_id}.joblib")
        gad_result = joblib.load(gad_result_filepath)

        # Features from GAD (P_man, P_int, P_bnd)
        gad_features = np.zeros((len(data_words_to_full_vocab_ints), 3), dtype=int)
        for i, word_int in enumerate(tqdm(data_words_to_full_vocab_ints)):
            for gad_category, gad_category_idx in gad_categories.items():
                if word_int in gad_result[gad_category]:
                    gad_features[i, gad_category_idx] = 1

        # Save GAD features
        np.save(gad_features_filepath, gad_features)

    # (6) -- Vectorize persistence diagrams from GAD features --
    gad_features_pd_vectorized_dir = join(
        task_raw_data_dir, "gad_features_pd_vectorized"
    )
    gad_features_pd_vectorized_size = 5
    gad_features_pd_vectorized_size_flat = gad_features_pd_vectorized_size ** 2
    makedirs(gad_features_pd_vectorized_dir, exist_ok=True)
    for inner_param, outer_param in gad_params:
        gad_id = f"gad_knn_{inner_param}_{outer_param}"
        gad_features_pd_vecs_filepath = join(
            gad_features_pd_vectorized_dir, f"{gad_id}.npy"
        )
        if isfile(gad_features_pd_vecs_filepath):
            continue
        print(f"Vectorizing GAD features for {gad_id}...")

        # Load GAD features
        gad_result_filepath = join(gad_dir, f"{gad_id}.joblib")
        gad_result = joblib.load(gad_result_filepath)

        # Use PersistenceImage to vectorize persistence diagrams
        gad_features_pd_vecs = np.zeros(
            (len(data_words_to_full_vocab_ints), gad_features_pd_vectorized_size_flat)
        )
        for i, point_index in enumerate(tqdm(data_words_to_full_vocab_ints)):

            # Get persistence diagram and create a range such that we get a square image from PersistenceImager
            gad_features_pd = gad_result["annulus_pds"][point_index]
            if len(gad_features_pd) == 0:
                gad_features_pd_vecs[i] = np.zeros(
                    gad_features_pd_vectorized_size_flat, dtype=int
                )
                continue

            births, deaths = gad_features_pd.T
            persistence = deaths - births
            square_min = min(births.min(), persistence.min())
            square_max = max(births.max(), persistence.max())
            square_range = (square_min, square_max)
            pixel_size = (square_max - square_min) / gad_features_pd_vectorized_size

            # Vectorize persistence diagram
            pimgr = PersistenceImager(
                birth_range=square_range, pers_range=square_range, pixel_size=pixel_size
            )
            pd_vec = pimgr.transform(gad_features_pd)
            gad_features_pd_vecs[i] = pd_vec.flatten()

        # Save persistence image vectors to file
        np.save(gad_features_pd_vecs_filepath, gad_features_pd_vecs)

    # (7) -- Combine data into data (features and labels) for WME task --
    word_meaning_train_data_filepath = join(output_dir, "word_meaning_train_data.csv")
    word_meaning_test_data_filepath = join(output_dir, "word_meaning_test_data.csv")
    word_meaning_semeval_test_data_filepath = join(
        output_dir, "word_meaning_semeval_test_data.csv"
    )
    if (
        not isfile(word_meaning_train_data_filepath)
        or not isfile(word_meaning_test_data_filepath)
        or not isfile(word_meaning_semeval_test_data_filepath)
    ):
        # -- Load data for creating features --
        # Load estimated IDs from file
        words_estimated_ids = {
            f"{id_estimator_name}_{num_neighbours}": np.load(
                join(
                    words_estimated_ids_dir, f"{id_estimator_name}_{num_neighbours}.npy"
                )
            )
            for num_neighbours in id_estimation_num_neighbours
            for id_estimator_name, _, _ in id_estimators
        }
        print("Loaded estimated IDs!")

        # Load GAD features
        gad_features_dict = {}
        gad_features_pd_vecs_dict = {}
        for inner_param, outer_param in gad_params:
            gad_id = f"gad_knn_{inner_param}_{outer_param}"

            # Load GAD features
            gad_features_filepath = join(gad_features_dir, f"{gad_id}.npy")
            gad_features_dict[gad_id] = np.load(gad_features_filepath)

            # Load vectorized PDs from GAD features
            gad_features_pd_vecs_filepath = join(
                gad_features_pd_vectorized_dir, f"{gad_id}.npy"
            )
            gad_features_pd_vecs_dict[gad_id] = np.load(gad_features_pd_vecs_filepath)
        print("Loaded GAD features!")

        # Load TPS features
        tps_scores = {}
        tps_pds = {}
        for tps_neighbourhood_size, tps_scores_filepath, tps_pds_filepath in zip(
            tps_neighbourhood_sizes, tps_scores_filepaths, tps_pds_filepaths
        ):
            tps_scores[tps_neighbourhood_size] = np.load(tps_scores_filepath)
            tps_pds[tps_neighbourhood_size] = np.load(
                tps_pds_filepath, allow_pickle=True
            )
        print("Loaded TPS features!")

        data_words_train, data_words_test = train_test_split(
            data_words_no_semeval, test_size=0.05, random_state=rng_seed
        )
        if not isfile(word_meaning_train_data_filepath):
            train_data_df = create_word_meaning_model_data_features(
                target_words=data_words_train,
                word_to_int=data_word_to_int,
                tps_scores=tps_scores,
                tps_pds=tps_pds,
                tps_neighbourhood_sizes=tps_neighbourhood_sizes,
                words_estimated_ids=words_estimated_ids,
                words_to_meanings=words_to_num_meanings,
                gad_categories=gad_categories,
                gad_features_dict=gad_features_dict,
                gad_features_pd_vecs_dict=gad_features_pd_vecs_dict,
            )
            train_data_df.to_csv(word_meaning_train_data_filepath, index=False)
        if not isfile(word_meaning_test_data_filepath):
            test_data_df = create_word_meaning_model_data_features(
                target_words=data_words_test,
                word_to_int=data_word_to_int,
                tps_scores=tps_scores,
                tps_pds=tps_pds,
                tps_neighbourhood_sizes=tps_neighbourhood_sizes,
                words_estimated_ids=words_estimated_ids,
                words_to_meanings=words_to_num_meanings,
                gad_categories=gad_categories,
                gad_features_dict=gad_features_dict,
                gad_features_pd_vecs_dict=gad_features_pd_vecs_dict,
            )
            test_data_df.to_csv(word_meaning_test_data_filepath, index=False)
        if not isfile(word_meaning_semeval_test_data_filepath):
            semeval_test_data_df = create_word_meaning_model_data_features(
                target_words=semeval_target_words_in_vocab,
                word_to_int=data_word_to_int,
                tps_scores=tps_scores,
                tps_pds=tps_pds,
                tps_neighbourhood_sizes=tps_neighbourhood_sizes,
                words_estimated_ids=words_estimated_ids,
                words_to_meanings=words_to_num_meanings,
                gad_categories=gad_categories,
                gad_features_dict=gad_features_dict,
                gad_features_pd_vecs_dict=gad_features_pd_vecs_dict,
            )
            semeval_test_data_df.to_csv(
                word_meaning_semeval_test_data_filepath, index=False
            )
    else:
        train_data_df = pd.read_csv(word_meaning_train_data_filepath)
        test_data_df = pd.read_csv(word_meaning_test_data_filepath)
        semeval_test_data_df = pd.read_csv(word_meaning_semeval_test_data_filepath)
    print("Train", train_data_df)
    print("Test", test_data_df)
    print("SemEval test", semeval_test_data_df)


if __name__ == "__main__":
    args = parse_args()
    prepare_num_word_meanings_supervised_data(
        model_dir=args.model_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        id_estimation_num_neighbours=args.id_estimation_num_neighbours,
        semeval_2010_14_word_senses_filepath=args.semeval_2010_14_word_senses_filepath,
        tps_neighbourhood_sizes=args.tps_neighbourhood_sizes,
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
    )
