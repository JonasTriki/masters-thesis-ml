import argparse
import sys
from os import makedirs
from os.path import join
from typing import Optional

import annoy
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import wordnet as wn
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import euclidean_distances
from topological_polysemy import tps, tps_point_cloud
from tqdm import tqdm

sys.path.append("..")

from word_embeddings.word2vec import load_model_training_output


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
        "--semeval_word_senses_filepath",
        type=str,
        default="",
        help="Filepath of the SemEval-2010 task 14 word senses",
    )
    parser.add_argument(
        "--word2vec_semeval_model_dir",
        type=str,
        default="",
        help="Directory of the SemEval-2010 task 14 word2vec model",
    )
    parser.add_argument(
        "--word2vec_enwiki_model_dir",
        type=str,
        default="",
        help="Directory of the enwiki word2vec model",
    )
    parser.add_argument(
        "--word2vec_google_news_model_weights_filepath",
        type=str,
        default="",
        help="Filepath of the Google News 3M word2vec model weights",
    )
    parser.add_argument(
        "--word2vec_google_news_model_words_filepath",
        type=str,
        default="",
        help="Filepath of the Google News 3M word2vec model words",
    )
    parser.add_argument(
        "--tps_neighbourhood_sizes",
        nargs="+",
        help="Neighbourhood sizes to use when computing TPS (e.g. 50, 60)",
    )
    parser.add_argument(
        "--num_top_k_words_frequencies",
        type=int,
        help="Number of top words to use when computing TPS scores vs. word frequencies",
    )
    parser.add_argument(
        "--cyclo_octane_data_filepath",
        type=str,
        default="",
        help="Filepath of the cyclo-octane dataset",
    )
    parser.add_argument(
        "--henneberg_data_filepath",
        type=str,
        default="",
        help="Filepath of the Henneberg dataset",
    )
    parser.add_argument(
        "--custom_point_cloud_neighbourhood_size",
        type=int,
        help="Neighbourhood size to use when computing TPS for custom point clouds",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory to save results",
    )
    return parser.parse_args()


def tps_word_embeddings_correlation_plot(
    tps_scores: np.ndarray,
    y_values: np.ndarray,
    y_values_name: str,
    y_label: str,
    tps_vs_y_correlation: float,
    output_dir: str,
    word_embeddings_name: str,
    neighbourhood_size: int,
) -> None:
    """
    Saves a correlation plot between TPS scores and some y values.

    Parameters
    ----------
    tps_scores : np.ndarray
        TPS scores.
    y_values : np.ndarray
        Y-values to plot against TPS scores.
    y_values_name : np.ndarray
        Name of the y-values.
    y_label : str
        Y-axis label.
    tps_vs_y_correlation : float
        Correlation between TPS scores and y values.
    output_dir : str
        Output directory.
    word_embeddings_name : str
        Name of word embeddings (appended to output filepath).
    neighbourhood_size : int
        Neighbourhood size used to compute TPS scores (appended to output filepath).
    """
    # Ensure output directory exists
    output_dir_plots = join(output_dir, word_embeddings_name)
    makedirs(output_dir_plots, exist_ok=True)

    # Plot TPS scores to GS
    fig, ax = plt.subplots(figsize=(10, 5))
    scatter_h = ax.scatter(x=tps_scores, y=y_values)
    if len(tps_scores) > 1000:
        scatter_h.set_rasterized(True)
    ax.set_xlabel(f"TPS_{neighbourhood_size}")
    ax.set_ylabel(y_label)
    ax.set_title(f"Correlation: {tps_vs_y_correlation:.5f}")
    plt.tight_layout()
    plt.savefig(
        join(
            output_dir_plots,
            f"tps_{neighbourhood_size}_vs_{y_values_name}.pdf",
        ),
        backend="pgf",
    )
    plt.close(fig)


def tps_word_embeddings(
    word_embeddings_name: str,
    neighbourhood_sizes: int,
    semeval_target_words: np.ndarray,
    semeval_target_words_gs_clusters: np.ndarray,
    word_embeddings_normalized: np.ndarray,
    word_to_int: dict,
    word_vocabulary: list,
    num_top_k_words_frequencies: int,
    output_dir: str,
    word_counts: Optional[list] = None,
    annoy_index: annoy.AnnoyIndex = None,
) -> dict:
    """
    Computes TPS for word embeddings and saves correlation plots.

    Parameters
    ----------

    Returns
    -------
    """
    # Ensure output directory exists
    makedirs(output_dir, exist_ok=True)

    # Only use the SemEval-2010 task 14 words in vocabulary
    semeval_target_words_in_vocab_filter = [
        i for i, word in enumerate(semeval_target_words) if word in word_to_int
    ]
    semeval_target_words_in_vocab = semeval_target_words[
        semeval_target_words_in_vocab_filter
    ]
    semeval_target_words_gs_clusters_in_vocab = semeval_target_words_gs_clusters[
        semeval_target_words_in_vocab_filter
    ]

    tps_vs_gs_key = "TPS_n vs. GS"
    tps_vs_synsets_key = "TPS_n vs. synsets"
    tps_vs_frequency_key = "TPS_n vs. frequency"
    result_dict: dict = {
        "n": neighbourhood_sizes,
        tps_vs_gs_key: [],
        tps_vs_synsets_key: [],
    }
    has_word_counts = word_counts is not None
    if has_word_counts:
        result_dict[tps_vs_frequency_key] = []

    for neighbourhood_size in neighbourhood_sizes:
        print(f"-- Neighbourhood size: {neighbourhood_size} --")

        # -- Compute TPS scores and correlation vs GS words --
        tps_scores_semeval = []
        print("Computing TPS scores for GS words")
        for semeval_target_word in tqdm(semeval_target_words_in_vocab):
            tps_score_semeval = tps(
                target_word=semeval_target_word,
                word_to_int=word_to_int,
                neighbourhood_size=neighbourhood_size,
                word_embeddings_normalized=word_embeddings_normalized,
                annoy_index=annoy_index,
            )
            tps_scores_semeval.append(tps_score_semeval)

        # Compute correlation vs GS word meanings
        tps_score_vs_gs_correlation, _ = pearsonr(
            x=tps_scores_semeval, y=semeval_target_words_gs_clusters_in_vocab
        )
        result_dict[tps_vs_gs_key].append(tps_score_vs_gs_correlation)

        # Save plot of TPS scores vs. GS
        tps_word_embeddings_correlation_plot(
            tps_scores=tps_scores_semeval,
            y_values=semeval_target_words_gs_clusters_in_vocab,
            y_values_name="gs",
            y_label="Clusters in GS",
            tps_vs_y_correlation=tps_score_vs_gs_correlation,
            output_dir=output_dir,
            word_embeddings_name=word_embeddings_name,
            neighbourhood_size=neighbourhood_size,
        )

        # Find words in vocabulary that have synsets in Wordnet
        tps_scores_wordnet_synsets = []
        wordnet_synsets_words_in_vocab_meanings = []
        print("Computing TPS scores for words in vocabulary with Wordnet synsets")
        for word in tqdm(word_vocabulary):
            num_synsets_word = len(wn.synsets(word))
            if num_synsets_word > 0:
                wordnet_synsets_words_in_vocab_meanings.append(num_synsets_word)
                tps_score_wordnet_synset = tps(
                    target_word=word,
                    word_to_int=word_to_int,
                    neighbourhood_size=neighbourhood_size,
                    word_embeddings_normalized=word_embeddings_normalized,
                    annoy_index=annoy_index,
                )
                tps_scores_wordnet_synsets.append(tps_score_wordnet_synset)

        # Compute correlation vs Wordnet synsets
        tps_score_vs_wordnet_synsets_correlation, _ = pearsonr(
            x=tps_scores_wordnet_synsets, y=wordnet_synsets_words_in_vocab_meanings
        )
        result_dict[tps_vs_synsets_key].append(tps_score_vs_wordnet_synsets_correlation)

        # Save plot of TPS scores vs. Wordnet synsets
        tps_word_embeddings_correlation_plot(
            tps_scores=tps_scores_wordnet_synsets,
            y_values=wordnet_synsets_words_in_vocab_meanings,
            y_values_name="synsets",
            y_label="Synsets in WordNet",
            tps_vs_y_correlation=tps_score_vs_wordnet_synsets_correlation,
            output_dir=output_dir,
            word_embeddings_name=word_embeddings_name,
            neighbourhood_size=neighbourhood_size,
        )

        if has_word_counts:
            print(
                f"Computing TPS scores for top {num_top_k_words_frequencies} words vs. word frequencies"
            )
            tps_score_word_frequencies = []
            for word in tqdm(word_vocabulary[:num_top_k_words_frequencies]):
                tps_score_word_frequency = tps(
                    target_word=word,
                    word_to_int=word_to_int,
                    neighbourhood_size=neighbourhood_size,
                    word_embeddings_normalized=word_embeddings_normalized,
                    annoy_index=annoy_index,
                )
                tps_score_word_frequencies.append(tps_score_word_frequency)

            # Compute correlation vs Wordnet synsets
            tps_score_vs_word_frequency_correlation, _ = pearsonr(
                x=tps_score_word_frequencies,
                y=word_counts[:num_top_k_words_frequencies],
            )
            result_dict[tps_vs_frequency_key].append(
                tps_score_vs_word_frequency_correlation
            )

            # Save plot of TPS scores vs. Wordnet synsets
            tps_word_embeddings_correlation_plot(
                tps_scores=tps_score_word_frequencies,
                y_values=word_counts[:num_top_k_words_frequencies],
                y_values_name="frequency",
                y_label="Word frequency",
                tps_vs_y_correlation=tps_score_vs_word_frequency_correlation,
                output_dir=output_dir,
                word_embeddings_name=word_embeddings_name,
                neighbourhood_size=neighbourhood_size,
            )


def topological_polysemy_pipeline(
    semeval_word_senses_filepath: str,
    word2vec_semeval_model_dir: str,
    word2vec_enwiki_model_dir: str,
    word2vec_google_news_model_weights_filepath: str,
    word2vec_google_news_model_words_filepath: str,
    tps_neighbourhood_sizes: str,
    num_top_k_words_frequencies: int,
    cyclo_octane_data_filepath: str,
    henneberg_data_filepath: str,
    custom_point_cloud_neighbourhood_size: int,
    output_dir: str,
) -> None:
    """
    Computes the topological polysemy of various word embeddings and data sets.
    Saves results in output dir with some additional plots.

    Parameters
    ----------
    semeval_word_senses_filepath : str
        Filepath of the SemEval-2010 task 14 word senses
    word2vec_semeval_model_dir : str
        Directory of the SemEval-2010 task 14 word2vec model.
    word2vec_enwiki_model_dir : str
        Directory of the enwiki word2vec model.
    word2vec_google_news_model_weights_filepath : str
        Filepath of the Google News 3M word2vec model weights.
    word2vec_google_news_model_words_filepath : str
        Filepath of the Google News 3M word2vec model words.
    tps_neighbourhood_sizes : str
        Neighbourhood sizes to use when computing TPS (e.g. 50, 60).
    num_top_k_words_frequencies : int
        Number of top words to use when computing TPS scores vs. word frequencies.
    cyclo_octane_data_filepath : str
        Filepath of the cyclo-octane dataset.
    henneberg_data_filepath : str
        Filepath of the Henneberg dataset.
    custom_point_cloud_neighbourhood_size : int
        Neighbourhood size to use when computing TPS for custom point clouds.
    output_dir : str
        Output directory to save results.
    """
    # Ensure output directory exists
    makedirs(output_dir, exist_ok=True)

    # Load SemEval-2010 task 14 word senses
    semeval_word_senses: dict = joblib.load(semeval_word_senses_filepath)
    semeval_target_words = np.array(list(semeval_word_senses["all"].keys()))
    semeval_target_word_gs_clusters = np.array(
        list(semeval_word_senses["all"].values())
    )

    # Parse strings into int
    tps_neighbourhood_sizes = [int(n_size) for n_size in tps_neighbourhood_sizes]

    # -- Compute TPS for word embeddings (SemEval and enwiki) --
    for dataset_name, model_dir in zip(
        ["semeval_2010_task_14", "enwiki"],
        [word2vec_semeval_model_dir, word2vec_enwiki_model_dir],
    ):
        # Load word embeddings
        print(f"Loading {dataset_name} word embeddings...")
        w2v_training_output = load_model_training_output(
            model_training_output_dir=model_dir,
            model_name="word2vec",
            dataset_name=dataset_name,
            return_normalized_embeddings=True,
            return_annoy_index=True,
            annoy_index_prefault=True,
        )
        last_embedding_weights_normalized = w2v_training_output[
            "last_embedding_weights_normalized"
        ]
        last_embedding_weights_annoy_index = w2v_training_output[
            "last_embedding_weights_annoy_index"
        ]
        words = w2v_training_output["words"]
        word_to_int = w2v_training_output["word_to_int"]
        word_counts = w2v_training_output["word_counts"]
        print("Done!")

        print("Computing TPS for word embeddings...")
        tps_word_embeddings(
            word_embeddings_name=dataset_name,
            neighbourhood_sizes=tps_neighbourhood_sizes,
            semeval_target_words=semeval_target_words,
            semeval_target_words_gs_clusters=semeval_target_word_gs_clusters,
            word_embeddings_normalized=last_embedding_weights_normalized,
            word_to_int=word_to_int,
            word_vocabulary=words,
            num_top_k_words_frequencies=num_top_k_words_frequencies,
            output_dir=output_dir,
            word_counts=word_counts,
            annoy_index=last_embedding_weights_annoy_index,
        )
        print("Done!")

    # -- Compute TPS for GoogleNews 3M word embeddings --
    # Load data
    print("Loading GoogleNews 3M data...")
    google_news_model_weights = np.load(
        word2vec_google_news_model_weights_filepath, mmap_mode="r"
    )
    google_news_model_weights_normalized = google_news_model_weights / np.linalg.norm(
        google_news_model_weights, axis=1
    ).reshape(-1, 1)
    with open(word2vec_google_news_model_words_filepath, "r") as words_file:
        google_news_model_words = np.array(words_file.read().split("\n"))
    print("Done!")
    print("Computing TPS for GoogleNews 3M word embeddings...")
    tps_word_embeddings(
        word_embeddings_name="google_news_3m",
        neighbourhood_sizes=tps_neighbourhood_sizes,
        semeval_target_words=semeval_target_words,
        semeval_target_words_gs_clusters=semeval_target_word_gs_clusters,
        word_embeddings_normalized=google_news_model_weights_normalized,
        word_to_int={
            i: google_news_model_words[i] for i in range(len(google_news_model_words))
        },
        word_vocabulary=google_news_model_words,
        num_top_k_words_frequencies=num_top_k_words_frequencies,
        output_dir=output_dir,
    )
    print("Done!")

    # -- Compute TPS for custom point clouds --
    for point_cloud_name, point_cloud_filepath in zip(
        ["cyclo_octane", "henneberg"],
        [cyclo_octane_data_filepath, henneberg_data_filepath],
    ):
        # Load and prepare data for TPS
        point_cloud = pd.read_csv(point_cloud_filepath, header=None).values
        point_cloud_normalized = point_cloud / np.linalg.norm(
            point_cloud, axis=1
        ).reshape(-1, 1)
        point_cloud_pairwise_dists = euclidean_distances(point_cloud)

        # Compute TPS scores
        num_points = len(point_cloud)
        tps_scores = np.zeros(num_points)
        print(f"Computing TPS scores for {point_cloud_name}...")
        for point_index in tqdm(range(num_points)):
            tps_score = tps_point_cloud(
                point_index=point_index,
                neighbourhood_size=custom_point_cloud_neighbourhood_size,
                point_cloud_normalized=point_cloud_normalized,
                point_cloud_pairwise_dists=point_cloud_pairwise_dists,
            )
            tps_scores[point_index] = tps_score

        # Save result
        point_cloud_output_dir = join(output_dir, point_cloud_name)
        makedirs(point_cloud_output_dir, exist_ok=True)
        np.save(
            join(
                point_cloud_output_dir,
                f"tps_scores_{custom_point_cloud_neighbourhood_size}.npy",
            ),
            tps_scores,
        )


if __name__ == "__main__":
    args = parse_args()
    topological_polysemy_pipeline(
        semeval_word_senses_filepath=args.semeval_word_senses_filepath,
        word2vec_semeval_model_dir=args.word2vec_semeval_model_dir,
        word2vec_enwiki_model_dir=args.word2vec_enwiki_model_dir,
        word2vec_google_news_model_weights_filepath=args.word2vec_google_news_model_weights_filepath,
        word2vec_google_news_model_words_filepath=args.word2vec_google_news_model_words_filepath,
        tps_neighbourhood_sizes=args.tps_neighbourhood_sizes,
        num_top_k_words_frequencies=args.num_top_k_words_frequencies,
        cyclo_octane_data_filepath=args.cyclo_octane_data_filepath,
        henneberg_data_filepath=args.henneberg_data_filepath,
        custom_point_cloud_neighbourhood_size=args.custom_point_cloud_neighbourhood_size,
        output_dir=args.output_dir,
    )
