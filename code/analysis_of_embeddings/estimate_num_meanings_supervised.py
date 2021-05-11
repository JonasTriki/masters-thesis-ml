import argparse
from os import makedirs
from os.path import isfile, join
from string import ascii_lowercase
from time import time

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import (
    confusion_matrix,
    make_scorer,
    mean_squared_error,
    recall_score,
)
from sklearn.preprocessing import minmax_scale

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
        "--train_data_filepath",
        type=str,
        default="",
        help="Filepath of word meaning training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory",
    )
    return parser.parse_args()


def create_classification_labels(labels: np.ndarray, max_label: int) -> np.ndarray:
    """
    Converts real-valued labels into classfication labels, where `max_label` denotes the
    maximum real-valued label. Any label greater than max_label will be categorized using the
    same label.

    Parameters
    ----------
    labels : np.ndarray
        Labels to convert.
    max_label : int
        Maximum label.

    Returns
    -------
    classification_labels : np.ndarray
        Classification labels.
    """
    classification_labels = np.zeros_like(labels)
    for i, label in enumerate(labels):
        if label > max_label:
            classification_labels[i] = max_label
        else:
            classification_labels[i] = label - 1
    return classification_labels


def evaluate_regression_model(
    model: object,
    test_sets: list,
    show_plot: bool = True,
    use_rasterization: bool = False,
) -> None:
    """
    Evaluates a trained regression model on test data sets.

    Parameters
    ----------
    model : object
        Trained model (must have `predict()` method).
    test_sets : list
        List of test data sets, where each entry is a tuple:
            X_eval : np.ndarray
                Prediction data.
            y_true : np.ndarray
                True labels for `X_eval`.
            test_set_name : str
                Name of test set.
            xlabel : str
                Label for x-axis of pred/true plot.
            ylabel : str
                Label for y-axis of pred/true plot.
    show_plot : bool, optional
        Whether or not to call `plt.show()` to show the plot (defaults to True)
    use_rasterization : bool, optional
        Whether or not to enable rasterization for scatter plots of many data points
        (defaults to False).
    """
    num_test_sets = len(test_sets)
    _, axes = plt.subplots(nrows=1, ncols=num_test_sets, figsize=(5 * num_test_sets, 5))
    test_set_chars = ascii_lowercase[:num_test_sets]
    for test_set, test_set_char, ax in zip(test_sets, test_set_chars, axes):
        X_eval, y_true, test_set_name, xlabel, ylabel = test_set
        pred_labels = model.predict(X_eval)
        pred_true_corr, _ = pearsonr(pred_labels, y_true)

        ax_scatter_handle = ax.scatter(pred_labels, y_true, s=15)
        if use_rasterization and len(pred_labels) > 1000:
            ax_scatter_handle.set_rasterized(True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"({test_set_char}) Correlation: {pred_true_corr:.3f} ({test_set_name})"
        )
    if show_plot:
        plt.show()


def evaluate_classification_model(
    model: object,
    test_sets: list,
    cm_ticklabels: list,
    show_plot: bool = True,
) -> None:
    """
    Evaluates a trained classification model on test data sets.

    Parameters
    ----------
    model : object
        Trained model (must have `predict()` method).
    test_sets : list
        List of test data sets, where each entry is a tuple:
            X_eval : np.ndarray
                Prediction data.
            y_true : np.ndarray
                True labels for `X_eval`.
            test_set_name : str
                Name of test set.
            xlabel : str
                Label for x-axis of pred/true plot.
            ylabel : str
                Label for y-axis of pred/true plot.
    cm_ticklabels : list
        List of ticklabels to use for confusion matrix plot.
    show_plot : bool, optional
        Whether or not to call `plt.show()` to show the plot (defaults to True)
    """
    num_test_sets = len(test_sets)
    _, axes = plt.subplots(nrows=1, ncols=num_test_sets, figsize=(7 * num_test_sets, 7))
    test_set_chars = ascii_lowercase[:num_test_sets]
    for ax, test_set, test_set_char in zip(axes, test_sets, test_set_chars):
        X_eval, y_true, test_set_name, xlabel, ylabel = test_set
        pred_labels_proba = model.predict_proba(X_eval)
        is_multi_class = pred_labels_proba.shape[1] > 2
        pred_labels = np.argmax(pred_labels_proba, axis=1)
        if is_multi_class:
            pred_eval_score = recall_score(y_true, pred_labels, average="weighted")
        else:
            pred_eval_score = recall_score(
                y_true,
                pred_labels,
                pos_label=1,
            )

        pred_cm = confusion_matrix(y_true, pred_labels)
        heatmap_handle = sns.heatmap(
            pred_cm,
            cmap="YlGnBu",
            annot=True,
            fmt="d",
            annot_kws={"size": 16},
            square=True,
            xticklabels=cm_ticklabels,
            ax=ax,
            cbar_kws={"shrink": 0.75},
        )
        ax.set_yticklabels(cm_ticklabels, va="center", rotation=90, position=(0, 0.28))
        heatmap_handle.figure.axes[-1].set_title("Number of words", pad=15)
        ax.set_title(
            f"({test_set_char}) Sensitivity: {pred_eval_score: .3f} ({test_set_name})"
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    if show_plot:
        plt.show()


def create_feature_importance_df(
    feature_names: list, feature_importances: list
) -> pd.DataFrame:
    """
    Creates feature importances DataFrame.

    Parameters
    ----------
    feature_names : list
        List of feature names.
    feature_importances : list
        Feature importanes.

    Returns
    -------
    feature_importances_df : pd.DataFrame
        Feature importances DataFrame.
    """
    sorted_feature_importance_indices = np.argsort(feature_importances)[::-1]
    sorted_features_arr = np.array(
        list(
            zip(
                feature_names[sorted_feature_importance_indices],
                feature_importances[sorted_feature_importance_indices],
            )
        )
    )
    feature_importances_df = pd.DataFrame(
        {"feature": sorted_features_arr[:, 0], "importance": sorted_features_arr[:, 1]}
    )
    return feature_importances_df


def visualize_feature_importances(feature_importances: pd.DataFrame) -> None:
    """
    Visualize feature importances from `create_feature_importance_df` in a Jupyter
    Notebook.

    Parameters
    ----------
    feature_importances : pd.DataFrame
        Feature importances DataFrame from `create_feature_importance_df`.
    """
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        display(feature_importances)


def estimate_num_meanings_supervised(train_data_filepath: str, output_dir: str) -> None:
    """
    Estimates number of word meanings using supervised models.

    Parameters
    ----------
    train_data_filepath : str
        Filepath of word meaning training data.
    output_dir : str
        Output directory.
    """
    # Create output directory
    output_dir = join(output_dir, "estimate_num_meanings_supervised")
    makedirs(output_dir, exist_ok=True)

    # Load data
    print("Preparing data...")
    word_meaning_train_data = pd.read_csv(train_data_filepath)
    word_meaning_data_cols = word_meaning_train_data.columns.values
    word_meaning_data_feature_cols = np.array(
        [col for col in word_meaning_data_cols if col.startswith("X_")]
    )

    # Split into X and y
    X_train = minmax_scale(
        word_meaning_train_data[word_meaning_data_feature_cols].values
    )
    print(f"Train data shape: {X_train.shape}")
    y_train = word_meaning_train_data["y"].values
    y_train_binary_classes = create_classification_labels(labels=y_train, max_label=1)

    # Prepare train params
    num_folds = 20
    model_classes = [
        LassoCV,
        LogisticRegressionCV,
    ]
    model_names = [
        "lasso_reg",
        "binary_logistic_reg",
    ]
    binary_sensitivity_scorer = make_scorer(recall_score, pos_label=1)
    models_params = [
        {
            "alphas": np.linspace(0.0000001, 0.01, num=10000),
            "max_iter": 1000000,
            "random_state": rng_seed,
            "cv": num_folds,
            "n_jobs": -1,
            "verbose": 0,
        },
        {
            "Cs": 1 / np.linspace(0.00001, 0.01, num=10000),
            "penalty": "l1",
            "solver": "saga",
            "max_iter": 1000000,
            "scoring": binary_sensitivity_scorer,
            "random_state": rng_seed,
            "cv": num_folds,
            "n_jobs": -1,
            "verbose": 0,
        },
    ]
    models_train_params = [
        {"model_type": "regression"},
        {"model_type": "binary_classification"},
    ]

    for model_cls, model_name, model_params, model_train_params in zip(
        model_classes, model_names, models_params, models_train_params
    ):
        model_filepath = join(output_dir, f"{model_name}.joblib")
        if isfile(model_filepath):
            continue
        model_instance = model_cls(**model_params)
        model_type = model_train_params["model_type"]

        print(f"Training {model_name}...")
        start_time = time()
        if model_type == "regression":
            model_instance.fit(X_train, y_train)
        elif model_type == "binary_classification":
            model_instance.fit(X_train, y_train_binary_classes)
        train_time = time() - start_time
        print(f"Done! Spent {train_time:.3f} seconds training.")

        print("Saving to file...")
        joblib.dump(model_instance, model_filepath, protocol=4)
        print("Done!")


if __name__ == "__main__":
    args = parse_args()
    estimate_num_meanings_supervised(
        train_data_filepath=args.train_data_filepath,
        output_dir=args.output_dir,
    )
