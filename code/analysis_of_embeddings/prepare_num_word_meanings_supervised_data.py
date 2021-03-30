import argparse
import sys
from os import makedirs
from os.path import isfile, join
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from genericpath import isdir
from nltk.corpus import wordnet as wn
from skdim._commonfuncs import GlobalEstimator
from skdim.id import KNN, MLE, TwoNN, lPCA
from tqdm import tqdm

sys.path.append("..")

from approx_nn import ApproxNN  # noqa: E402
from topological_data_analysis.geometric_anomaly_detection import (  # noqa: E402
    compute_gad,
)
from topological_data_analysis.topological_polysemy import tps  # noqa: E402
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
        "--id_estimation_neighbours",
        type=int,
        default=100,
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
    gad_features_dict: dict,
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
    gad_features_dict : dict
        Dictionary containing features from GAD.

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
        gad_features = gad_features_dict[gad_config]
        for gad_category in gad_features.keys():
            if gad_category.startswith("P_"):
                data_features[f"X_{gad_config}_{gad_category}"] = []

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
            tps_pd_zero_dim_deaths = tps_pds[n_size][word_int, :, 1]
            data_features[f"X_tps_{n_size}_pd_max"].append(tps_pd_zero_dim_deaths.max())
            data_features[f"X_tps_{n_size}_pd_avg"].append(
                tps_pd_zero_dim_deaths.mean()
            )
            data_features[f"X_tps_{n_size}_pd_std"].append(tps_pd_zero_dim_deaths.std())

        # Features from GAD (P_man, P_int, P_bnd)
        for gad_config in gad_features_dict.keys():
            gad_features = gad_features_dict[gad_config]
            for gad_category in gad_features.keys():
                if gad_category.startswith("P_"):
                    word_in_gad_category = int(word_int in gad_features[gad_category])
                    data_features[f"X_{gad_config}_{gad_category}"].append(
                        word_in_gad_category
                    )

    # Create df and return it
    data_features_df = pd.DataFrame(data_features)

    return data_features_df


def prepare_num_word_meanings_supervised_data(
    model_dir: str,
    model_name: str,
    dataset_name: str,
    id_estimation_neighbours: int,
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
    id_estimation_neighbours : int
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
    # Convert TPS neighbourhood sizes to ints
    tps_neighbourhood_sizes = [int(n_size) for n_size in tps_neighbourhood_sizes]

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
        return_scann_instance=not isdir(task_raw_data_tps_dir),
    )
    last_embedding_weights_normalized = w2v_training_output[
        "last_embedding_weights_normalized"
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
    data_words = list(words_to_num_meanings.keys())
    data_words_no_semeval = [
        word for word in data_words if word not in semeval_target_words_in_vocab
    ]
    data_word_to_int = {word: i for i, word in enumerate(data_words)}

    # Filter out word embeddings using Wordnet words (data_words)
    data_words_to_full_vocab_ints = np.array([word_to_int[word] for word in data_words])
    word_embeddings_wordnet_words = last_embedding_weights_normalized[
        data_words_to_full_vocab_ints
    ]

    # (2) -- Create ApproxNN index of data_words word embeddings --
    approx_nn_index_dir = join(task_raw_data_dir, "approx_nn")
    if not isdir(approx_nn_index_dir):
        print("Building ApproxNN index...")
        approx_nn = ApproxNN(ann_alg="scann")
        approx_nn.build(
            data=word_embeddings_wordnet_words,
            scann_num_leaves_scaling=10,
            scann_default_num_neighbours=1000,
            distance_measure="squared_l2",
        )
        print("Saving ApproxNN index...")
        approx_nn.save(output_path=approx_nn_index_dir)
        print("Done!")

    approx_nn_index_annoy_filepath = join(task_raw_data_dir, "annoy.ann")
    if not isfile(approx_nn_index_annoy_filepath):
        approx_nn = ApproxNN(ann_alg="annoy")
        approx_nn.build(
            data=word_embeddings_wordnet_words,
            annoy_n_trees=500,
            distance_measure="euclidean",
        )
        print("Saving ApproxNN index...")
        approx_nn.save(output_path=approx_nn_index_annoy_filepath)
        print("Done!")

    # (3) -- Estimate the intrinsic dimension (ID) for each word vector --
    words_estimated_ids_dir = join(task_raw_data_dir, "estimated_ids")
    id_estimators: List[Tuple[str, GlobalEstimator, dict]] = [
        ("lpca", lPCA, {}),
        ("knn", KNN, {}),
        ("twonn", TwoNN, {}),
        ("mle", MLE, {}),
    ]
    makedirs(words_estimated_ids_dir, exist_ok=True)
    for id_estimator_name, id_estimator_cls, id_estimator_params in id_estimators:
        estimated_ids_filepath = join(
            words_estimated_ids_dir, f"{id_estimator_name}.npy"
        )
        if isfile(estimated_ids_filepath):
            continue

        id_estimator = id_estimator_cls(**id_estimator_params)

        print(f"Estimating IDs using {id_estimator_cls.__name__}...")
        estimated_ids = id_estimator.fit_predict_pw(
            X=word_embeddings_wordnet_words,
            n_neighbors=id_estimation_neighbours,
            n_jobs=-1,
        )

        print("Done! Saving to file...")
        np.save(estimated_ids_filepath, estimated_ids)

    # Load estimated IDs from file
    words_estimated_ids = {
        id_estimator_name: np.load(
            join(words_estimated_ids_dir, f"{id_estimator_name}.npy")
        )
        for id_estimator_name, _, _ in id_estimators
    }
    print("Loaded words_estimated_ids!")

    # (4) -- Compute TPS_n for train/test words --
    if not isdir(task_raw_data_tps_dir):
        makedirs(task_raw_data_tps_dir, exist_ok=True)
        print("Computing TPS scores...")
        last_embedding_weights_scann_instance = w2v_training_output[
            "last_embedding_weights_scann_instance"
        ]
        tps_scores = {}
        tps_pds = {}
        for tps_neighbourhood_size in tps_neighbourhood_sizes:
            print(f"Neighbourhood size: {tps_neighbourhood_size}")
            tps_scores_filepath = join(
                task_raw_data_tps_dir, f"tps_{tps_neighbourhood_size}_scores.npy"
            )
            tps_pds_filepath = join(
                task_raw_data_tps_dir, f"tps_{tps_neighbourhood_size}_pds.npy"
            )
            if isfile(tps_scores_filepath) and isfile(tps_pds_filepath):
                continue

            tps_scores[tps_neighbourhood_size] = []
            tps_pds[tps_neighbourhood_size] = []
            for word in tqdm(data_words):
                tps_score, tps_pd = tps(
                    target_word=word,
                    word_to_int=word_to_int,
                    neighbourhood_size=tps_neighbourhood_size,
                    words_vocabulary=data_words,
                    word_embeddings_normalized=last_embedding_weights_normalized,
                    ann_instance=last_embedding_weights_scann_instance,
                    return_persistence_diagram=True,
                )

                # Create Nx2 array from zero dimensional homology
                tps_pd_zero_dim = np.array(
                    [
                        [
                            [birth, death]
                            for dim, (birth, death) in tps_pd
                            if dim == 0 and death != np.inf
                        ]
                    ]
                )

                tps_scores[tps_neighbourhood_size].append(tps_score)
                tps_pds[tps_neighbourhood_size].append(tps_pd_zero_dim)

            # Save result
            np.save(tps_scores_filepath, np.array(tps_scores[tps_neighbourhood_size]))
            np.save(tps_pds_filepath, np.array(tps_pds[tps_neighbourhood_size]))
        print("Done!")
    else:
        tps_scores = {
            n_size: np.load(join(task_raw_data_tps_dir, f"tps_{n_size}_scores.npy"))
            for n_size in tps_neighbourhood_sizes
        }
        tps_pds = {
            n_size: np.load(join(task_raw_data_tps_dir, f"tps_{n_size}_pds.npy"))
            for n_size in tps_neighbourhood_sizes
        }
        print("Loaded tps_scores and tps_pds!")

    # (5) -- Compute GAD features --
    gad_features_dir = join(task_raw_data_dir, "gad_features")
    makedirs(gad_features_dir, exist_ok=True)
    gad_features_params = {
        "radius": [(0.5, 1.0), (1.0, 1.5), (1.5, 2.5)],
        "knn": [(25, 250), (50, 250), (50, 550)],
    }
    for gad_type, gad_params in gad_features_params.items():
        use_knn_annulus = gad_type == "knn"
        if use_knn_annulus:
            approx_nn = ApproxNN(ann_alg="scann")
            approx_nn.load(ann_path=approx_nn_index_dir)
        else:
            approx_nn = ApproxNN(ann_alg="annoy")
            approx_nn.load(
                ann_path=approx_nn_index_annoy_filepath,
                annoy_data_dimensionality=word_embeddings_wordnet_words.shape[1],
                annoy_mertic="euclidean",
                annoy_prefault=False,
            )
        for inner_param, outer_param in gad_params:
            gad_features_id = f"gad_{gad_type}_{inner_param}_{outer_param}"
            print(f"-- {gad_features_id} -- ")

            gad_features_filepath = join(gad_features_dir, f"{gad_features_id}.joblib")
            if isfile(gad_features_filepath):
                continue

            # Compute features
            if use_knn_annulus:
                gad_params_kwargs = {
                    "knn_annulus_inner": inner_param,
                    "knn_annulus_outer": outer_param,
                }
            else:
                gad_params_kwargs = {
                    "annulus_inner_radius": inner_param,
                    "annulus_outer_radius": outer_param,
                }
            gad_result = compute_gad(
                data_points=word_embeddings_wordnet_words,
                manifold_dimension=2,
                data_points_approx_nn=approx_nn,
                use_knn_annulus=use_knn_annulus,
                return_annlus_persistence_diagrams=False,
                progressbar_enabled=True,
                n_jobs=-1,
                **gad_params_kwargs,
            )
            print(
                "P_man:",
                len(gad_result["P_man"]),
                "P_int:",
                len(gad_result["P_int"]),
                "P_bnd:",
                len(gad_result["P_bnd"]),
            )
            joblib.dump(gad_result, gad_features_filepath, protocol=4)
        del approx_nn

    gad_features_dict = {}
    for gad_type, gad_params in gad_features_params.items():
        use_knn_annulus = gad_type == "knn"
        for inner_param, outer_param in gad_params:
            gad_features_id = f"gad_{gad_type}_{inner_param}_{outer_param}"
            gad_features_filepath = join(gad_features_dir, f"{gad_features_id}.joblib")
            gad_features_dict[gad_features_id] = joblib.load(gad_features_filepath)

    # gad_grid_search_filepath = join(
    #     task_raw_data_dir, "gad_features_grid_search.joblib"
    # )
    # if not isfile(gad_grid_search_filepath):

    #     # Load ANN index
    #     approx_nn = ApproxNN(ann_alg="annoy")
    #     approx_nn.load(
    #         ann_path=approx_nn_index_annoy_filepath,
    #         annoy_data_dimensionality=300,
    #         annoy_mertic="euclidean",
    #         annoy_prefault=False,
    #     )

    #     gad_result = grid_search_gad_annulus_radii(
    #         data_points=word_embeddings_wordnet_words,
    #         manifold_dimension=2,
    #         search_size=20,
    #         use_knn_annulus=True,
    #         search_params_max_diff=250,
    #         min_annulus_parameter=1,
    #         max_annulus_parameter=500,
    #         data_points_approx_nn=approx_nn,
    #         return_annlus_persistence_diagrams=True,
    #         progressbar_enabled=True,
    #         n_jobs=-1,
    #     )
    #     joblib.dump(gad_result, gad_grid_search_filepath)

    # (6) -- Combine data into data (features and labels) for WME task --
    word_meaning_train_data_filepath = join(output_dir, "word_meaning_train_data.csv")
    word_meaning_test_data_filepath = join(output_dir, "word_meaning_test_data.csv")
    if not isfile(word_meaning_train_data_filepath) and not isfile(
        word_meaning_test_data_filepath
    ):
        train_data_df = create_word_meaning_model_data_features(
            target_words=data_words_no_semeval,
            word_to_int=data_word_to_int,
            tps_scores=tps_scores,
            tps_pds=tps_pds,
            tps_neighbourhood_sizes=tps_neighbourhood_sizes,
            words_estimated_ids=words_estimated_ids,
            words_to_meanings=words_to_num_meanings,
            gad_features_dict=gad_features_dict,
        )
        train_data_df.to_csv(word_meaning_train_data_filepath, index=False)
        test_data_df = create_word_meaning_model_data_features(
            target_words=semeval_target_words_in_vocab,
            word_to_int=data_word_to_int,
            tps_scores=tps_scores,
            tps_pds=tps_pds,
            tps_neighbourhood_sizes=tps_neighbourhood_sizes,
            words_estimated_ids=words_estimated_ids,
            words_to_meanings=words_to_num_meanings,
            gad_features_dict=gad_features_dict,
        )
        test_data_df.to_csv(word_meaning_test_data_filepath, index=False)
    else:
        train_data_df = pd.read_csv(word_meaning_train_data_filepath)
        test_data_df = pd.read_csv(word_meaning_test_data_filepath)
    print("Train", train_data_df)
    print("Test", test_data_df)


if __name__ == "__main__":
    args = parse_args()
    prepare_num_word_meanings_supervised_data(
        model_dir=args.model_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        id_estimation_neighbours=args.id_estimation_neighbours,
        semeval_2010_14_word_senses_filepath=args.semeval_2010_14_word_senses_filepath,
        tps_neighbourhood_sizes=args.tps_neighbourhood_sizes,
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
    )
