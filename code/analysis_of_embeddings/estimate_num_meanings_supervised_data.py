import sys
from os.path import isfile
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from skdim.id import lPCA
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

sys.path.append("..")

from topological_data_analysis.topological_polysemy import tps  # noqa: E402
from word_embeddings.word2vec import load_model_training_output  # noqa: E402

rng_seed = 399
np.random.seed(rng_seed)


# Constants
lpca_n_neighbors = 100
tps_neighbourhood_sizes = np.linspace(start=10, stop=100, num=10, dtype=int)
compute_words_to_num_synsets = False
compute_id = False
compute_tps = True
compute_word_meaning_features = True

# Load word embeddings.
# TODO: Implement argparse and load word embeddings from there
w2v_training_output = load_model_training_output(
    model_training_output_dir="../output/word2vec_training/word2vec_enwiki_jan_2021_word2phrase",
    model_name="word2vec",
    dataset_name="enwiki",
    return_normalized_embeddings=True,
    return_scann_instance=compute_tps,
)
last_embedding_weights_normalized = w2v_training_output[
    "last_embedding_weights_normalized"
]
words = w2v_training_output["words"]
word_to_int = w2v_training_output["word_to_int"]


def find_synset_words(word: str) -> Optional[int]:
    """
    TODO: Docs
    """
    num_synsets = len(wn.synsets(word))
    if num_synsets > 0:
        return word, num_synsets
    else:
        return None


# Find words in Wordnet that are in the word2vec model's vocabulary.
if compute_words_to_num_synsets and compute_word_meaning_features:
    words_to_num_synsets = {}
    print("Finding words in vocabulary with #Wordnet synsets > 0")
    for word in tqdm(words):
        num_synsets = len(wn.synsets(word))
        if num_synsets > 0:
            words_to_num_synsets[word] = num_synsets
    joblib.dump(words_to_num_synsets, "data/words_to_num_synsets.joblib")
else:
    words_to_num_synsets = joblib.load("data/words_to_num_synsets.joblib")
    print("Loaded words_to_num_synsets!")

data_words = list(words_to_num_synsets.keys())
data_word_ints = np.array([word_to_int[word] for word in data_words])

# Estimate the intrinsic dimension (ID) for each word vector
if compute_id and compute_word_meaning_features:
    print("Estimating the intrinsic dimension (ID)...")
    words_estimated_ids = lPCA().fit_predict_pw(
        X=last_embedding_weights_normalized[data_word_ints],
        n_neighbors=lpca_n_neighbors,
    )
    print("Done!")
    np.save("data/words_estimated_ids.npy", words_estimated_ids)
else:
    words_estimated_ids = np.load("data/words_estimated_ids.npy")
    print("Loaded words_estimated_ids!")

if compute_tps and compute_word_meaning_features:
    print("Computing TPS scores...")
    last_embedding_weights_scann_instance = w2v_training_output[
        "last_embedding_weights_scann_instance"
    ]
    for tps_neighbourhood_size in tps_neighbourhood_sizes:
        print(f"Neighbourhood size: {tps_neighbourhood_size}")
        tps_scores_filepath = (
            f"data/tps_{tps_neighbourhood_size}_scores_wordnet_enwiki.npy"
        )
        tps_pds_filepath = f"data/tps_{tps_neighbourhood_size}_pds_wordnet_enwiki.npy"
        if isfile(tps_scores_filepath) and isfile(tps_pds_filepath):
            continue

        tps_scores = []
        tps_pds = []
        for i, word in enumerate(tqdm(data_words)):
            word_i = word_to_int[word]
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

            tps_scores.append(tps_score)
            tps_pds.append(tps_pd_zero_dim)

        # Save result
        np.save(tps_scores_filepath, np.array(tps_scores))
        np.save(tps_pds_filepath, np.array(tps_pds))
    print("Done!")
else:
    tps_scores = {
        [n_size]: np.load(f"data/tps_{n_size}_scores_wordnet_enwiki.npy")
        for n_size in tps_neighbourhood_sizes
    }
    tps_pds = {
        [n_size]: np.load(f"data/tps_{n_size}_pds_wordnet_enwiki.npy")
        for n_size in tps_neighbourhood_sizes
    }
    print("Loaded tps_scores and tps_pds!")


def create_word_meaning_model_data_features(
    words: list,
    word_to_int: dict,
    tps_scores: dict,
    tps_pds: dict,
    words_estimated_ids: np.array,
    words_to_num_synsets: dict,
) -> pd.DataFrame:
    """
    TODO: Docs
    """
    data_features = {
        "word": [],
        "word_int": [],
        "estimated_id": [],
        "y": [],
    }
    for n_size in tps_neighbourhood_sizes:
        data_features[f"tps_{n_size}"] = []
        data_features[f"tps_{n_size}_bottle"] = []
    for i, word in enumerate(tqdm(words)):
        word_i = word_to_int[word]

        # Add word and word integer
        data_features["word"].append(word)  # Word
        data_features["word_int"].append(word_i)  # Word integer
        data_features["estimated_id"].append(words_estimated_ids[i])  # Estimated ID
        data_features["y"].append(words_to_num_synsets[word])

        for n_size in tps_neighbourhood_sizes:
            data_features[f"tps_{n_size}"].append(tps_scores[n_size][i])
            data_features[f"tps_{n_size}_bottle"].append(tps_pds[n_size][i, :, 1].max())

    # Create df and scale features
    data_features_df = pd.DataFrame(data_features)
    feature_cols = [
        "estimated_id",
        *[f"tps_{n_size}" for n_size in tps_neighbourhood_sizes],
        *[f"tps_{n_size}_bottle" for n_size in tps_neighbourhood_sizes],
    ]
    data_features_df[feature_cols] = minmax_scale(data_features_df[feature_cols].values)

    return data_features_df


if compute_word_meaning_features:
    data_X_df = create_word_meaning_model_data_features(
        words=data_words,
        word_to_int=word_to_int,
        tps_scores=tps_scores,
        tps_pds=tps_pds,
        words_estimated_ids=words_estimated_ids,
        words_to_num_synsets=words_to_num_synsets,
    )
    data_X_df.to_csv("data/word_meaning_features.csv", index=False)
else:
    data_X_df = pd.read_csv("data/word_meaning_features.csv")
print(data_X_df)
