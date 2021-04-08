import argparse
from os import makedirs
from os.path import isfile, join

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from IPython.display import display
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_auc_score
from sklearn.preprocessing import minmax_scale
from skopt import BayesSearchCV
from skopt.space import Integer, Real

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
    model: object, test_sets: list, show_plot: bool = True
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
        Whether or not to call `plt.show()` to show the plot (defaults to True).
    """
    num_test_sets = len(test_sets)
    _, axes = plt.subplots(
        nrows=1, ncols=num_test_sets, figsize=(5.5 * num_test_sets, 5)
    )
    for test_set, ax in zip(test_sets, axes):
        X_eval, y_true, test_set_name, xlabel, ylabel = test_set
        pred_labels = model.predict(X_eval)
        pred_true_corr, _ = pearsonr(pred_labels, y_true)
        pred_true_mse = mean_squared_error(y_true, pred_labels)

        ax.scatter(pred_labels, y_true, s=15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"Correlation: {pred_true_corr:.3f}, MSE: {pred_true_mse:.3f} ({test_set_name})"
        )
    if show_plot:
        plt.show()


def evaluate_classification_model(
    model: object, test_sets: list, cm_ticklabels: list
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
    """
    for test_set in test_sets:
        X_eval, y_true, test_set_name, xlabel, ylabel = test_set
        pred_labels_proba = model.predict_proba(X_eval)
        pred_labels = np.argmax(pred_labels_proba, axis=1)
        pred_auc = roc_auc_score(
            y_true, pred_labels_proba, average="weighted", multi_class="ovr"
        )

        pred_cm = confusion_matrix(y_true, pred_labels)
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            pred_cm,
            cmap="YlGnBu",
            annot=True,
            fmt="d",
            annot_kws={"size": 16},
            square=True,
            xticklabels=cm_ticklabels,
            yticklabels=cm_ticklabels,
        )
        plt.title(f"AUC: {pred_auc: .3f} ({test_set_name})")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
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
        BayesSearchCV,
        BayesSearchCV,
        BayesSearchCV,
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
                objective="reg:squarederror",
                n_estimators=100,
                random_state=rng_seed,
                n_jobs=1,
            ),
            "search_spaces": {
                "eta": Real(0.0001, 0.1),
                "max_depth": Integer(3, 10),
                "gamma": Real(0.001, 0.5),
                "subsample": Real(0.5, 1),
                "colsample_bytree": Real(0.5, 1),
                "alpha": Real(0.00001, 0.1),
            },
            "cv": num_folds,
            "n_iter": 250,
            "random_state": rng_seed,
            "verbose": 3,
            "n_jobs": -1,
        },
        {
            "estimator": xgb.XGBClassifier(
                objective="binary:logistic",
                use_label_encoder=False,
                n_estimators=100,
                scale_pos_weight=1,
                random_state=rng_seed,
                n_jobs=1,
            ),
            "search_spaces": {
                "min_child_weight": Integer(1, 6),
                "eta": Real(0.0001, 0.1),
                "max_depth": Integer(3, 10),
                "gamma": Real(0.001, 0.5),
                "subsample": Real(0.5, 1),
                "colsample_bytree": Real(0.5, 1),
                "alpha": Real(0.00001, 0.1),
            },
            "cv": num_folds,
            "n_iter": 250,
            "random_state": rng_seed,
            "verbose": 3,
            "fit_params": {
                "eval_metric": "auc",
            },
            "n_jobs": -1,
        },
        {
            "estimator": xgb.XGBClassifier(
                objective="multi:softprob",
                use_label_encoder=False,
                n_estimators=100,
                scale_pos_weight=1,
                random_state=rng_seed,
                n_jobs=1,
            ),
            "search_spaces": {
                "min_child_weight": Integer(1, 6),
                "eta": Real(0.0001, 0.1),
                "max_depth": Integer(3, 10),
                "gamma": Real(0.001, 0.5),
                "subsample": Real(0.5, 1),
                "colsample_bytree": Real(0.5, 1),
                "alpha": Real(0.00001, 0.1),
            },
            "cv": num_folds,
            "n_iter": 250,
            "random_state": rng_seed,
            "verbose": 3,
            "fit_params": {
                "eval_metric": "auc",
                "num_class": num_y_train_multi_classes,
            },
            "n_jobs": -1,
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
