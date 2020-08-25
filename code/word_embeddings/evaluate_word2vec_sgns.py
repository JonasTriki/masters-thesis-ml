import argparse

from tensorflow.keras.models import load_model

from utils import (
    get_target_embedding_weights,
    get_words_from_word_dict,
    read_vocabulary_from_file,
    similar_words,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_filepath",
        type=str,
        # TODO: Remove default
        default="checkpoints/model-sgns.20-0.05373597890138626.h5",
        help="Path of saved model checkpoint",
    )
    parser.add_argument(
        "--vocab_filepath",
        type=str,
        default="data/alice_in_wonderland_vocab.pickle",  # TODO: Remove default
        help="Vocabulary filepath containing the word vocabulary we want to use",
    )
    return parser.parse_args()


def evaluate_word2vec_sgns(model_filepath: str, vocab_filepath: str) -> None:
    """
    Evaluates a Word2vec skipgram negative sampling model
    """
    print(f"-- Evaluating model from {model_filepath}... --")

    # Load model
    print("Loading model...")
    model = load_model(model_filepath)
    print("Done!")
    model.summary()

    # Get target embedding weights
    target_embedding_weights = get_target_embedding_weights(model)

    # Load vocabulary
    _, word_dict, _ = read_vocabulary_from_file(vocab_filepath)
    words = get_words_from_word_dict(word_dict)
    print(word_dict)
    print(target_embedding_weights.shape, words.shape)

    # Test similarities
    print(similar_words("sixteen", target_embedding_weights, words))


if __name__ == "__main__":
    args = parse_args()
    evaluate_word2vec_sgns(args.model_filepath, args.vocab_filepath)
