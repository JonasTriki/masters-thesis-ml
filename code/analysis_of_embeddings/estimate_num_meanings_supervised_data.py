import sys
from os.path import isfile
from typing import Optional

import joblib
import numpy as np
import tensorflow as tf
from nltk.corpus import wordnet as wn
from pervect import PersistenceVectorizer
from skdim.id import lPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

sys.path.append("..")

from topological_data_analysis.topological_polysemy import tps  # noqa: E402
from word_embeddings.word2vec import load_model_training_output  # noqa: E402

rng_seed = 399
np.random.seed(rng_seed)
tf.random.set_seed(rng_seed)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# Constants
lpca_n_neighbors = 100
tps_neighbourhood_sizes = [40, 50, 60]
compute_words_to_num_synsets = False
compute_id = False
compute_tps = False
compute_word_meaning_features = False

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
data_y = np.array(list(words_to_num_synsets.values()))

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
    tps_scores = np.load("data/tps_50_scores_wordnet_enwiki.npy")
    tps_pds = np.load("data/tps_50_pds_wordnet_enwiki.npy")
    print("Loaded tps_scores and tps_pds!")


def create_word_meaning_model_features(
    words: list,
    word_to_int: dict,
    word_embeddings_normalized: np.ndarray,
    tps_scores: np.ndarray,
    tps_pds: np.ndarray,
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

        # Add TPS score to feature vector
        if include_tps_score:
            feature_vec.append(tps_scores[i])

        if include_pd_vector:

            # Vectorize zero-degree persistence diagram
            tps_pd_vects = PersistenceVectorizer(random_state=rng_seed).fit_transform(
                tps_pds[i]
            )[0]

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
    data_X = create_word_meaning_model_features(
        words=data_words,
        word_to_int=word_to_int,
        word_embeddings_normalized=last_embedding_weights_normalized,
        tps_scores=tps_scores,
        tps_pds=tps_pds,
        words_estimated_ids=words_estimated_ids,
        include_word_vector=True,
        include_tps_score=True,
        include_pd_vector=True,
        include_esimated_id=True,
    )
    np.save("data/word_meaning_features.npy", data_X)
else:
    data_X = np.load("data/word_meaning_features.npy")
print(data_X.shape)
data_N, data_d = data_X.shape


def create_model(
    input_dim: int,
    num_hidden_layers: int,
    optimizer: tf.keras.optimizers.Optimizer,
    loss: str,
) -> Model:
    """
    TODO: Docs
    """
    # Input layer
    input_layer = Input(shape=(input_dim,))

    # Add hidden layers
    if num_hidden_layers > 0:
        input_layer = Dense(data_d // 2, activation=relu)(input_layer)
        for i in range(2, num_hidden_layers):
            divisor = 2 ** i
            input_layer = Dense(data_d // divisor, activation=relu)(input_layer)

    # Output layer
    output_layer = Dense(1, activation=relu)(input_layer)

    # Create model and compile
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])

    return model


X_train, X_val, y_train, y_val = train_test_split(
    data_X[:, :300], data_y, test_size=0.05, random_state=rng_seed
)

# Fit model
wm_model = create_model(
    input_dim=X_train.shape[1],
    num_hidden_layers=0,
    optimizer=Adam(learning_rate=0.0001),
    loss=MSE,
)
wm_model.summary()
wm_model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    batch_size=256,
    epochs=500,
)
