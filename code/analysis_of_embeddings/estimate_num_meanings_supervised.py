import sys
from typing import Optional

import joblib
import numpy as np
from nltk.corpus import wordnet as wn
from pervect import PersistenceVectorizer
from skdim.id import lPCA
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

sys.path.append("..")

from approx_nn import ApproxNN  # noqa: E402
from topological_data_analysis.topological_polysemy import tps  # noqa: E402
from word_embeddings.word2vec import load_model_training_output  # noqa: E402

rng_seed = 399
np.random.seed(rng_seed)

# Load word embeddings.
# TODO: Implement argparse and load word embeddings from there
w2v_training_output = load_model_training_output(
    model_training_output_dir="../output/word2vec_training/word2vec_enwiki_jan_2021_word2phrase",
    model_name="word2vec",
    dataset_name="enwiki",
    return_normalized_embeddings=True,
    return_scann_instance=True,
)
last_embedding_weights_normalized = w2v_training_output[
    "last_embedding_weights_normalized"
]
last_embedding_weights_scann_instance = w2v_training_output[
    "last_embedding_weights_scann_instance"
]
words = w2v_training_output["words"]
word_to_int = w2v_training_output["word_to_int"]

# Constants
lpca_n_neighbors = 100
tps_neighbourhood_size = 50
compute_words_to_num_synsets = False
compute_id = False
compute_word_meaning_features = True

# print("Load word embeddings...")
# last_embedding_weights_normalized = np.array(last_embedding_weights_normalized)
# print("Done!")


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
if compute_words_to_num_synsets:
    words_to_num_synsets = {}
    print("Finding words in vocabulary with #Wordnet synsets > 0")
    for word in tqdm(words):
        num_synsets = len(wn.synsets(word))
        if num_synsets > 0:
            words_to_num_synsets[word] = num_synsets
    joblib.dump(words_to_num_synsets, "data/words_to_num_synsets.joblib")
else:
    words_to_num_synsets = joblib.load("data/words_to_num_synsets.joblib")

data_words = list(words_to_num_synsets.keys())
data_word_ints = np.array([word_to_int[word] for word in data_words])
data_y = np.array(list(words_to_num_synsets.values()))

# Estimate the intrinsic dimension (ID) for each word vector
if compute_id:
    print("Estimating the intrinsic dimension (ID)...")
    words_estimated_ids = lPCA().fit_predict_pw(
        X=last_embedding_weights_normalized[data_word_ints],
        n_neighbors=lpca_n_neighbors,
    )
    print("Done!")
    np.save("data/words_estimated_ids.npy", words_estimated_ids)
else:
    words_estimated_ids = np.load("data/words_estimated_ids.npy")

print(words_estimated_ids.shape)


def create_word_meaning_model_features(
    words: list,
    word_to_int: dict,
    word_embeddings_normalized: np.ndarray,
    ann_instance: ApproxNN,
    tps_neighbourhood_size: int,
    words_estimated_ids: np.array,
    include_word_vector: bool,
    include_tps_score: bool,
    include_pd_vector: bool,
    include_esimated_id: bool,
) -> np.ndarray:
    """
    TODO: Docs
    """
    # TODO: Move TPS code outside method and cache result to file.
    features = []
    for i, word in enumerate(tqdm(words)):
        word_i = word_to_int[word]
        feature_vec = []

        # Add word vector to feature vector
        if include_word_vector:
            feature_vec.extend(word_embeddings_normalized[word_i])

        # Compute TPS scores and (vectorized) persistence diagrams
        if include_tps_score or include_pd_vector:
            tps_result = tps(
                target_word=word,
                word_to_int=word_to_int,
                neighbourhood_size=tps_neighbourhood_size,
                words_vocabulary=words,
                word_embeddings_normalized=word_embeddings_normalized,
                ann_instance=ann_instance,
                return_persistence_diagram=include_pd_vector,
            )
            if include_pd_vector:
                tps_score, tps_pd = tps_result
            else:
                tps_score = tps_result

            # Add TPS score to feature vector
            if include_tps_score:
                feature_vec.append(tps_score)

            if include_pd_vector:

                # Vectorize zero-degree persistence diagram
                tps_pd = np.array(
                    [
                        [
                            [birth, death]
                            for _, (birth, death) in tps_pd
                            if death != np.inf
                        ]
                    ]
                )
                tps_pd_vects = PersistenceVectorizer(
                    random_state=rng_seed
                ).fit_transform(tps_pd)[0]

                # Add vectorized TPS persistence diagram to feature vector
                feature_vec.extend(tps_pd_vects)

        # Add estimated ID to feature vector
        if include_esimated_id:
            feature_vec.append(words_estimated_ids[i])

        # Add new feature vector to features
        features.append(feature_vec)

    # Convert to numpy and min-max scale data to [0-1] range.
    features = np.array(features)
    features_scaled = MinMaxScaler().fit_transform(features)

    return features_scaled


if compute_word_meaning_features:
    word_meaning_features = create_word_meaning_model_features(
        words=data_words,
        word_to_int=word_to_int,
        word_embeddings_normalized=last_embedding_weights_normalized,
        ann_instance=last_embedding_weights_scann_instance,
        tps_neighbourhood_size=tps_neighbourhood_size,
        words_estimated_ids=words_estimated_ids,
        include_word_vector=True,
        include_tps_score=True,
        include_pd_vector=True,
        include_esimated_id=True,
    )
    np.save("data/word_meaning_features.npy", word_meaning_features)
else:
    word_meaning_features = np.load("data/word_meaning_features.npy")
print(word_meaning_features.shape)
