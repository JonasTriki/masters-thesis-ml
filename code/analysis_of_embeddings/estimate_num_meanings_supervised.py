import argparse
from os import makedirs
from os.path import isfile, join

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
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


def plot_pred_vs_true_labels(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    xlabel: str,
    ylabel: str,
    show_plot: bool = True,
) -> None:
    """
    Plots predicted labels (x-axis) vs true labels (y-axis).

    Parameters
    ----------
    pred_labels : np.ndarray
        Predicted labels.
    true_labels : np.ndarray
        True labels.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    show_plot : bool, optional
        Whether or not to call `plt.show()` to show the plot (defaults to True).
    """
    pred_true_corr, _ = pearsonr(pred_labels, true_labels)
    pred_true_mse = mean_squared_error(true_labels, pred_labels)

    plt.figure(figsize=(10, 7))
    plt.scatter(pred_labels, true_labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Pred/True: correlation: {pred_true_corr:.3f}, MSE: {pred_true_mse:.3f}")
    if show_plot:
        plt.show()


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
    y_train = word_meaning_train_data["y"].values
    max_y_multi = np.quantile(y_train, q=0.9)
    print(f"Max label for multi classifcation: {max_y_multi}")
    y_train_binary_classes = create_classification_labels(labels=y_train, max_label=1)
    y_train_multi_classes = create_classification_labels(
        labels=y_train, max_label=max_y_multi
    )
    num_y_train_multi_classes = len(np.unique(y_train_multi_classes))

    # Prepare train params
    num_folds = 20
    model_classes = [
        LassoCV,
        LogisticRegressionCV,
        LogisticRegressionCV,
        GridSearchCV,
        GridSearchCV,
        GridSearchCV,
    ]
    model_names = [
        "lasso_reg",
        "binary_logistic_reg",
        "multi_class_logistic_reg",
        "xgb_reg",
        "xgb_binary_classification",
        "xgb_multi_classification",
    ]
    models_params = [
        {
            "alphas": np.linspace(0.00001, 0.99999, 10000),
            "cv": num_folds,
            "max_iter": 100000,
            "n_jobs": -1,
            "random_state": rng_seed,
        },
        {
            "Cs": 1 / np.linspace(0.00000001, 0.1, 10000),
            "cv": num_folds,
            "max_iter": 1000000,
            "penalty": "l1",
            "solver": "saga",
            "verbose": 0,
            "n_jobs": -1,
            "random_state": rng_seed,
        },
        {
            "Cs": 1 / np.linspace(0.00000001, 0.1, 10000),
            "cv": num_folds,
            "max_iter": 1000000,
            "penalty": "l1",
            "solver": "saga",
            "verbose": 0,
            "n_jobs": -1,
            "random_state": rng_seed,
        },
        {
            "estimator": xgb.XGBRegressor(
                objective="reg:squarederror", random_state=rng_seed
            ),
            "param_grid": {
                "eta": np.arange(0.1, 0.26, 0.05),
                "alpha": np.linspace(0.00001, 0.99999, 10000),
            },
            "n_jobs": -1,
            "cv": num_folds,
        },
        {
            "estimator": xgb.XGBClassifier(
                objective="binary:logistic", random_state=rng_seed
            ),
            "param_grid": {
                "eta": np.arange(0.1, 0.26, 0.05),
                "alpha": np.linspace(0.00001, 0.99999, 10000),
            },
            "n_jobs": -1,
            "cv": num_folds,
        },
        {
            "estimator": xgb.XGBClassifier(
                objective="multi:softmax", random_state=rng_seed
            ),
            "param_grid": {
                "num_class": [num_y_train_multi_classes],
                "eta": np.arange(0.1, 0.26, 0.05),
                "alpha": np.linspace(0.00001, 0.99999, 10000),
            },
            "n_jobs": -1,
            "cv": num_folds,
        },
    ]
    models_train_params = [
        {"model_type": "regression"},
        {"model_type": "binary_classification"},
        {"model_type": "multi_classification"},
        {"model_type": "regression"},
        {"model_type": "binary_classification"},
        {"model_type": "multi_classification"},
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
        if model_type == "regression":
            model_instance.fit(X_train, y_train)
        elif model_type == "binary_classification":
            model_instance.fit(X_train, y_train_binary_classes)
        elif model_type == "multi_classification":
            model_instance.fit(X_train, y_train_multi_classes)
        print("Done!")

        print("Saving to file...")
        joblib.dump(model_instance, model_filepath, protocol=4)
        print("Done!")


if __name__ == "__main__":
    args = parse_args()
    estimate_num_meanings_supervised(
        train_data_filepath=args.train_data_filepath,
        output_dir=args.output_dir,
    )
