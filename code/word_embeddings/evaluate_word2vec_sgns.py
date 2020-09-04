import argparse
from pprint import pprint

from eval_utils import get_word_vec, similar_words_vec
from word2vec import Word2vec


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
        "--model_checkpoint_filepath",
        type=str,
        default="",
        help="Path of saved model checkpoint",
    )
    return parser.parse_args()


def evaluate_word2vec_sgns(model_filepath: str) -> None:
    """
    Evaluates a trained Word2vec skip-gram negative sampling model

    Parameters
    ----------
    model_filepath : str
        Filepath of the model to evaluate
    """
    print(f"-- Evaluating model from {model_filepath}... --")

    # Load model
    print("Loading model...")
    word2vec = Word2vec()
    word2vec.load_model(model_filepath)
    print("Done!")

    # Get target embedding weights
    embedding_weights = word2vec.embedding_weights

    # Test similarities
    a_vec = get_word_vec("king", word2vec.tokenizer.word_to_int, embedding_weights)
    b_vec = get_word_vec("adult", word2vec.tokenizer.word_to_int, embedding_weights)
    # c_vec = get_word_vec("berlin", word2vec.tokenizer.word_to_int, embedding_weights)
    d_vec = a_vec - b_vec  # + c_vec

    pprint(
        similar_words_vec(d_vec, embedding_weights, word2vec.tokenizer.words, top_n=20)
    )

    # print(
    #    similar_words(
    #        "king",
    #        embedding_weights,
    #        word2vec.tokenizer.word_to_int,
    #        word2vec.tokenizer.words,
    #    )
    # )


if __name__ == "__main__":
    args = parse_args()
    evaluate_word2vec_sgns(args.model_checkpoint_filepath)
