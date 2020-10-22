import argparse
import pickle
import sys
from datetime import datetime
from os import makedirs
from os.path import join

import numpy as np
from eval_utils import evaluate_model_word_analogies

sys.path.append("..")
from utils import get_model_checkpoint_filepaths


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
        help="Directory of the model to evaluate",
    )
    parser.add_argument(
        "--model_name", type=str, default="word2vec", help="Name of the trained model"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="enwiki",
        help="Name of the dataset the model is trained on",
    )
    parser.add_argument(
        "--sswr_dataset_filepath",
        type=str,
        default="",
        help="Filepath of the SSWR test dataset",
    )
    parser.add_argument(
        "--msr_dataset_filepath",
        type=str,
        default="",
        help="Filepath of the MSR test dataset",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=30000,
        help="Vocabulary size to use when evaluating on the test datasets",
    )
    parser.add_argument(
        "--top_n_prediction",
        type=int,
        default=1,
        help="Which top-N prediction we would like to do",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory to save evaluation results",
    )
    return parser.parse_args()


def save_analogies_accuracies_to_file(
    analogies_dataset_name: str, output_dir: str, analogies_accuracies: dict
) -> None:
    """
    TODO: Docs
    """
    analogies_output_filepath = join(output_dir, f"{analogies_dataset_name}.pkl")
    with open(analogies_output_filepath, "wb") as file:
        pickle.dump(analogies_accuracies, file)


def evaluate_word2vec(
    model_dir: str,
    model_name: str,
    dataset_name: str,
    sswr_dataset_filepath: str,
    msr_dataset_filepath: str,
    vocab_size: int,
    top_n_prediction: int,
    output_dir: str,
) -> None:
    """
    TODO: Docs
    """
    # Get filepaths of the model output
    checkpoint_filepaths_dict = get_model_checkpoint_filepaths(
        output_dir=model_dir,
        model_name=model_name,
        dataset_name=dataset_name,
    )

    # Get target embedding weights of last model
    last_embedding_weights_filepath = checkpoint_filepaths_dict[
        "intermediate_embedding_weight_filepaths"
    ][-1]

    # Load words and create word to int lookup dict
    with open(checkpoint_filepaths_dict["train_words_filepath"], "r") as file:
        words = np.array(file.read().split("\n"))
    word_to_int = {word: i for i, word in enumerate(words)}

    # Append date/time to output directory.
    output_dir = join(output_dir, datetime.now().strftime("%d-%b-%Y_%H-%M-%S"))
    makedirs(output_dir, exist_ok=True)

    # SSWR
    print("--- Evaluating SSWR ---")
    sswr_accuracies = evaluate_model_word_analogies(
        analogies_filepath=sswr_dataset_filepath,
        word_embeddings_filepath=last_embedding_weights_filepath,
        word_to_int=word_to_int,
        words=words,
        vocab_size=vocab_size,
        top_n=top_n_prediction,
    )
    save_analogies_accuracies_to_file("sswr", output_dir, sswr_accuracies)
    print(sswr_accuracies)

    # MSR
    print("--- Evaluating MSR ---")
    msr_accuracies = evaluate_model_word_analogies(
        analogies_filepath=msr_dataset_filepath,
        word_embeddings_filepath=last_embedding_weights_filepath,
        word_to_int=word_to_int,
        words=words,
        vocab_size=vocab_size,
        top_n=top_n_prediction,
    )
    save_analogies_accuracies_to_file("msr", output_dir, msr_accuracies)
    print(msr_accuracies)


if __name__ == "__main__":
    args = parse_args()

    # Perform evaluation
    evaluate_word2vec(
        model_dir=args.model_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        sswr_dataset_filepath=args.sswr_dataset_filepath,
        msr_dataset_filepath=args.msr_dataset_filepath,
        vocab_size=args.vocab_size,
        top_n_prediction=args.top_n_prediction,
        output_dir=args.output_dir,
    )
