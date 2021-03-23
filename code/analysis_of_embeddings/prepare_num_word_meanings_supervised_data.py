import argparse
import sys
from os import makedirs
from os.path import isfile, join

import joblib
import numpy as np
import pandas as pd
from genericpath import isdir
from nltk.corpus import wordnet as wn
from skdim.id import lPCA
from tqdm import tqdm

sys.path.append("..")

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
    words_estimated_ids: np.array,
    words_to_meanings: dict,
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
    words_estimated_ids : np.ndarray
        Estimated intrinsic dimension (ID) for all words.
    words_to_meanings : dict
        Dictionary mapping a word to its number of meanings; serves as the labels (y) for
        the supervised task.

    Returns
    -------
    data_features_df : pd.DataFrame
        DataFrame with data for the word meaning estimation task.
    """
    data_features = {
        "word": [],
        "estimated_id": [],
        "y": [],
    }
    for n_size in tps_neighbourhood_sizes:
        data_features[f"tps_{n_size}"] = []
        data_features[f"tps_{n_size}_bottle"] = []

    for target_word in tqdm(target_words):
        word_int = word_to_int[target_word]

        # Add word and word integer
        data_features["word"].append(target_word)  # Word
        data_features["estimated_id"].append(
            words_estimated_ids[word_int]
        )  # Estimated ID
        data_features["y"].append(words_to_meanings[target_word])  # Number of meanings

        for n_size in tps_neighbourhood_sizes:
            data_features[f"tps_{n_size}"].append(tps_scores[n_size][word_int])
            data_features[f"tps_{n_size}_bottle"].append(
                tps_pds[n_size][word_int, :, 1].max()
            )

    # Create df and return it
    data_features_df = pd.DataFrame(data_features)

    return data_features_df


def prepare_num_word_meanings_supervised_data(
    model_dir: str,
    model_name: str,
    dataset_name: str,
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
    semeval_2010_14_word_senses_filepath : str
        Filepath of SemEval-2010 task 14 word senses joblib dict.
    tps_neighbourhood_sizes : list
        List of TPS neighbourhood sizes.
    raw_data_dir : str
        Directory where raw data will be saved to.
    output_dir: str
        Output directory.
    """
    # Constants
    lpca_n_neighbors = 100
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
    semeval_gs_clusters_dict = {
        word: gs_meanings
        for word, gs_meanings in zip(
            semeval_target_words_in_vocab, semeval_gs_clusters_in_vocab
        )
    }

    # (1) -- Find words in Wordnet that are in the word2vec model's vocabulary --
    words_to_num_synsets_filepath = join(
        task_raw_data_dir, "words_to_num_synsets.joblib"
    )
    if not isfile(words_to_num_synsets_filepath):
        words_to_num_synsets = {}
        print("Finding words in vocabulary with #Wordnet synsets > 0")
        for word in tqdm(words):
            num_synsets = len(wn.synsets(word))
            if num_synsets > 0:
                words_to_num_synsets[word] = num_synsets
        joblib.dump(words_to_num_synsets, words_to_num_synsets_filepath)
    else:
        words_to_num_synsets = joblib.load(words_to_num_synsets_filepath)
        print("Loaded words_to_num_synsets!")
    data_words = list(words_to_num_synsets.keys())
    data_word_to_int = {word: i for i, word in enumerate(data_words)}

    # (2) -- Estimate the intrinsic dimension (ID) for each word vector --
    words_estimated_ids_filepath = join(task_raw_data_dir, "words_estimated_ids.npy")
    if not isfile(words_estimated_ids_filepath):
        print("Estimating the intrinsic dimension (ID)...")
        data_words_to_full_vocab_ints = np.array(
            [word_to_int[word] for word in data_words]
        )
        words_estimated_ids = lPCA().fit_predict_pw(
            X=last_embedding_weights_normalized[data_words_to_full_vocab_ints],
            n_neighbors=lpca_n_neighbors,
        )
        print("Done!")
        np.save(words_estimated_ids_filepath, words_estimated_ids)
    else:
        words_estimated_ids = np.load(words_estimated_ids_filepath)
        print("Loaded words_estimated_ids!")

    # (3) -- Compute TPS_n for train/test words --
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

    # (4) -- Combine data into data (features and labels) for WME task --
    word_meaning_train_data_filepath = join(output_dir, "word_meaning_train_data.csv")
    word_meaning_test_data_filepath = join(output_dir, "word_meaning_test_data.csv")
    if not isfile(word_meaning_train_data_filepath) and not isfile(
        word_meaning_test_data_filepath
    ):
        train_data_df = create_word_meaning_model_data_features(
            target_words=data_words,
            word_to_int=data_word_to_int,
            tps_scores=tps_scores,
            tps_pds=tps_pds,
            words_estimated_ids=words_estimated_ids,
            words_to_meanings=words_to_num_synsets,
        )
        train_data_df.to_csv(word_meaning_train_data_filepath, index=False)
        test_data_df = create_word_meaning_model_data_features(
            target_words=semeval_target_words_in_vocab,
            word_to_int=data_word_to_int,
            tps_scores=tps_scores,
            tps_pds=tps_pds,
            words_estimated_ids=words_estimated_ids,
            words_to_meanings=semeval_gs_clusters_dict,
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
        semeval_2010_14_word_senses_filepath=args.semeval_2010_14_word_senses_filepath,
        tps_neighbourhood_sizes=args.tps_neighbourhood_sizes,
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
    )
