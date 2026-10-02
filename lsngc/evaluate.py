"""Directed-network recovery metrics with the original demo conventions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata


def recovery_performance(
    Adjs: Sequence[ArrayLike], label: Sequence[ArrayLike]
) -> NDArray[np.float32]:
    """Return one AUROC per predicted/true adjacency pair, including diagonals.

    Implements the paper's network-recovery AUROC metric used in the Results
    comparisons and original demo. For ``n_pos`` positive and ``n_neg`` negative
    labels, the rank expression is

    ``AUROC = (sum(positive_ranks) - n_pos*(n_pos+1)/2)/(n_pos*n_neg)``.

    Scores use ascending average ranks, so a tied positive/negative pair
    contributes one half. This is equivalent to integration of the empirical
    ROC curve. As in the original ``recovery_performance``, matrices are
    flattened *with their diagonals*, the larger of two label values denotes
    the positive class, and the returned vector has dtype ``float32``.
    """
    if len(Adjs) != len(label):
        raise ValueError("Adjs and label must contain the same number of matrices")
    auc_all = np.zeros(len(Adjs), dtype=np.float32)
    for index, (prediction, truth) in enumerate(zip(Adjs, label)):
        prediction = np.asarray(prediction)
        truth = np.asarray(truth)
        if prediction.shape != truth.shape:
            raise ValueError("Prediction and label shapes must match")
        scores = prediction.ravel()
        labels = truth.ravel()
        if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(labels)):
            raise ValueError("Predictions and labels must contain only finite values")
        classes = np.unique(labels)
        if classes.size != 2:
            raise ValueError("AUROC requires exactly two label classes")
        positive = labels == classes[-1]
        n_positive = int(positive.sum())
        n_negative = positive.size - n_positive
        ranks = rankdata(scores, method="average")
        auc_all[index] = (
            ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0
        ) / (n_positive * n_negative)
    return auc_all
