"""
DSI validity index [1], source (downloaded 23th of February, 2021):
https://github.com/ShuyueG/CVI_using_DSI/blob/main/cluster_DSI_example.py

Modified to fit our needs.

References
----------
.. [1] Guan, S., & Loew, M. (2020). An Internal Cluster Validity Index Using a
   Distance-based Separability Measure. 2020 IEEE 32nd International Conference
   on Tools with Artificial Intelligence (ICTAI).
"""

from typing import Callable, Union

import numpy as np
from fastdist import fastdist
from scipy.stats import ks_2samp


def icd(X: np.ndarray, pos_indices: np.ndarray, dist_metric: Union[str, Callable]) -> np.ndarray:
    """
    Computes Intra-Class distances (ICD) [1].

    Parameters
    ----------
    X : np.ndarray
        Data points. If metric is set to "pairwise", then X is an (n*n) pairdise distance
        matrix.
    pos_indices : np.ndarray
        Indices of class-positive data points.
    dist_metric : str or Callable, optional
        Distance metric callable function. If metric is set to "pairwise", then X is an
        (n*n) pairdise distance matrix.

    Returns
    -------
    icd_dists : np.ndarray
        Intra-Class distances.

    References
    ----------
    .. [1] Guan, S., & Loew, M. (2020). An Internal Cluster Validity Index Using a
       Distance-based Separability Measure. 2020 IEEE 32nd International Conference
       on Tools with Artificial Intelligence (ICTAI).
    """
    num = len(pos_indices)
    icd_dists = []
    for i in range(0, num - 1):
        for j in range(i + 1, num):
            if dist_metric == "precomputed":
                dist = X[pos_indices[i], pos_indices[j]]
            else:
                dist = dist_metric(X[pos_indices[i]], X[pos_indices[j]])
            icd_dists.append(dist)
    return np.array(icd_dists)


def bcd(
    X: np.ndarray,
    pos_indices: np.ndarray,
    neg_indices: np.ndarray,
    dist_metric: Union[str, Callable],
) -> np.ndarray:
    """
    Computes Between-Class distances (BCD) [1].

    Parameters
    ----------
    X : np.ndarray
        Data points. If metric is set to "pairwise", then X is an (n*n) pairdise distance
        matrix.
    pos_indices : np.ndarray
        Indices of class-positive data points.
    neg_indices : np.ndarray
        Indices of class-negative data points.
    dist_metric : str or Callable, optional
        Distance metric callable function. If metric is set to "pairwise", then X is an
        (n*n) pairdise distance matrix.

    Returns
    -------
    bcd_dists : np.ndarray
        Intra-Class distances.

    References
    ----------
    .. [1] Guan, S., & Loew, M. (2020). An Internal Cluster Validity Index Using a
       Distance-based Separability Measure. 2020 IEEE 32nd International Conference
       on Tools with Artificial Intelligence (ICTAI).
    """
    pos_mask_num = len(pos_indices)
    neg_mask_num = len(neg_indices)
    bcd_dists = []
    for i in range(pos_mask_num):
        for j in range(neg_mask_num):
            if dist_metric == "precomputed":
                dist = X[pos_indices[i], neg_indices[j]]
            else:
                dist = dist_metric(X[pos_indices[i]], X[neg_indices[j]])
            bcd_dists.append(dist)
    return np.array(bcd_dists)


def dsi(
    X: np.ndarray,
    labels: np.ndarray,
    dist_metric: Union[str, Callable] = fastdist.euclidean,
) -> float:
    """
    Computes the Distance-based Separability Index (DSI) [1] of the given labels on the given data X.

    Parameters
    ----------
    X : np.ndarray
        Data points. If metric is set to "pairwise", then X is an (n*n) pairdise distance
        matrix.
    labels : np.ndarray
        Label for each data point.
    dist_metric : str or Callable, optional
        Distance metric callable function (defaults to fastdist.euclidean).
        If metric is set to "pairwise", then X is an (n*n) pairdise distance matrix.

    Returns
    -------
    dsi_score : float
        DSI score, ranging from 0 to 1, where higher values indicate better clustering.

    References
    ----------
    .. [1] Guan, S., & Loew, M. (2020). An Internal Cluster Validity Index Using a
       Distance-based Separability Measure. 2020 IEEE 32nd International Conference
       on Tools with Artificial Intelligence (ICTAI).
    """
    classes = np.unique(labels)
    dsi_sum = 0
    for c in classes:
        pos_indices = np.where(labels == c)[0]
        neg_indices = np.where(labels != c)[0]
        intra_class_distances = icd(X, pos_indices, dist_metric)
        between_class_distances = bcd(X, pos_indices, neg_indices, dist_metric)

        ks_stat, _ = ks_2samp(intra_class_distances, between_class_distances)
        dsi_sum += ks_stat

    dsi_score = dsi_sum / classes.shape[0]
    return dsi_score
