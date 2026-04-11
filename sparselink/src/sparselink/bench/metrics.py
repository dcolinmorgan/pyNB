"""Evaluation metrics for network inference benchmarking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""

    auroc: float
    aupr: float
    precision: float
    recall: float
    fdr: float


def evaluate(
    true_network: npt.NDArray[np.floating],
    predicted: npt.NDArray[np.floating],
    threshold: float | None = None,
) -> MetricsResult:
    """Compute evaluation metrics comparing predicted to true network.

    Args:
        true_network: Ground-truth adjacency matrix (n x n).
        predicted: Predicted adjacency/weight matrix (n x n).
        threshold: If provided, binarize predicted at this value.
            If None, uses median of nonzero values.

    Returns:
        MetricsResult with AUROC, AUPR, precision, recall, FDR.
    """
    # Flatten, ignoring diagonal
    n = true_network.shape[0]
    mask = ~np.eye(n, dtype=bool)
    y_true = (true_network[mask] != 0).astype(int)
    y_scores = np.abs(predicted[mask])

    # AUROC and AUPR (continuous scores)
    try:
        auroc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        auroc = 0.5
    try:
        aupr = float(average_precision_score(y_true, y_scores))
    except ValueError:
        aupr = 0.0

    # Binarize for precision/recall/FDR
    if threshold is None:
        nonzero = y_scores[y_scores > 0]
        threshold = float(np.median(nonzero)) if len(nonzero) > 0 else 0.0

    y_pred = (y_scores >= threshold).astype(int)

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    fdr = 1.0 - prec if np.any(y_pred) else 0.0

    return MetricsResult(auroc=auroc, aupr=aupr, precision=prec, recall=rec, fdr=fdr)
