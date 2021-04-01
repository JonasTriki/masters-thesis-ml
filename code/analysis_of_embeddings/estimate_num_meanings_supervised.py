import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import minmax_scale

rng_seed = 399
np.random.seed(rng_seed)


def create_multi_class_labels(labels: np.ndarray, max_label: int) -> np.ndarray:
    multi_class_labels = np.zeros_like(labels)
    for i, label in enumerate(labels):
        if label > max_label:
            multi_class_labels[i] = max_label
        else:
            multi_class_labels[i] = label - 1
    return multi_class_labels


# Load data
print("Preparing data...")
word_meaning_train_data = pd.read_csv("data/word_meaning_train_data.csv")
word_meaning_data_cols = word_meaning_train_data.columns.values
word_meaning_data_feature_cols = np.array(
    [col for col in word_meaning_data_cols if col.startswith("X_")]
)

# Split into X and y
X_train = minmax_scale(word_meaning_train_data[word_meaning_data_feature_cols].values)
y_train = word_meaning_train_data["y"].values

# Create multi-class labels
max_y_multi = np.quantile(y_train, q=0.9)
print(f"Max label: {max_y_multi}")
y_train_multi_class = create_multi_class_labels(labels=y_train, max_label=max_y_multi)
print("Done!")

# Parameters
cv = 20
max_iter = 1000000
l1_alphas = np.linspace(0.00000001, 0.1, 10000)
print("Running LogisticRegressionCV...")
log_reg_cv = LogisticRegressionCV(
    Cs=1 / l1_alphas,
    cv=cv,
    max_iter=max_iter,
    penalty="l1",
    solver="saga",
    n_jobs=-1,
    random_state=rng_seed,
    verbose=0,
)
log_reg_cv.fit(X_train, y_train_multi_class)
print("Done!")

print("Saving to file...")
joblib.dump(log_reg_cv, "data/log_reg_cv.joblib", protocol=4)
print("Done!")

print(f"L1 ratios: {log_reg_cv.l1_ratio_}")

# TODO: Add argparse to script and generalize
