import sys
from typing import Optional

import joblib
import numpy as np
from nltk.corpus import wordnet as wn
from tqdm import tqdm

sys.path.append("..")

from word_embeddings.word2vec import load_model_training_output  # noqa: E402

# Load word embeddings.
# TODO: Implement argparse and load word embeddings from there
w2v_training_output = load_model_training_output(
    model_training_output_dir="../output/word2vec_training/word2vec_enwiki_jan_2021_word2phrase",
    model_name="word2vec",
    dataset_name="enwiki",
    # return_normalized_embeddings=True,
)
last_embedding_weights = w2v_training_output["last_embedding_weights"]
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
compute_words_to_num_synsets = False
if compute_words_to_num_synsets:
    words_to_num_synsets = {}
    for word in tqdm(words):
        num_synsets = len(wn.synsets(word))
        if num_synsets > 0:
            words_to_num_synsets[word] = num_synsets
    joblib.dump(words_to_num_synsets, "data/words_to_num_synsets.joblib")
else:
    words_to_num_synsets = joblib.load("data/words_to_num_synsets.joblib")

data_words = list(words_to_num_synsets.keys())
data_y = np.array(list(words_to_num_synsets.values()))

# TODO: Compute:
# - Word vector?
# - Estimated intrinsic dimension (ID)
# - TPS scores
# - Persistence vector of PDs (from TPS or other relevant features)
