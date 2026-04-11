"""GRN-specific evaluation against gold standard networks.

Wraps the existing CompareModels functionality with a cleaner array-based API.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compare_to_gold_standard(
    predicted: np.ndarray,
    gold_standard: np.ndarray,
) -> dict[str, float]:
    """Compare a predicted GRN against a gold standard network.

    Parameters
    ----------
    predicted : np.ndarray
        Predicted adjacency matrix (n x n), can be weighted.
    gold_standard : np.ndarray
        Ground truth binary adjacency matrix (n x n).

    Returns
    -------
    dict
        Metrics: AUROC, AUPR, F1, MCC, precision, recall, specificity.
    """
    ref_binary = (gold_standard != 0).astype(int).ravel()
    pred_scores = np.abs(predicted).ravel()
    pred_binary = (predicted != 0).astype(int).ravel()

    # AUROC / AUPR (using continuous scores)
    try:
        auroc = roc_auc_score(ref_binary, pred_scores)
    except ValueError:
        auroc = 0.5
    try:
        aupr = average_precision_score(ref_binary, pred_scores)
    except ValueError:
        aupr = 0.0

    # Binary metrics
    tp = int(np.sum((ref_binary == 1) & (pred_binary == 1)))
    tn = int(np.sum((ref_binary == 0) & (pred_binary == 0)))
    fp = int(np.sum((ref_binary == 0) & (pred_binary == 1)))
    fn = int(np.sum((ref_binary == 1) & (pred_binary == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "AUROC": auroc,
        "AUPR": aupr,
        "F1": f1,
        "MCC": mcc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


def compare_multiple(
    predicted_list: list[np.ndarray],
    gold_standard: np.ndarray,
    method_names: list[str] | None = None,
) -> list[dict[str, float | str]]:
    """Compare multiple predicted networks against a gold standard.

    Parameters
    ----------
    predicted_list : list of np.ndarray
        List of predicted adjacency matrices.
    gold_standard : np.ndarray
        Ground truth binary adjacency matrix.
    method_names : list of str, optional
        Names for each method.

    Returns
    -------
    list of dict
        Each dict contains metrics plus a 'method' key.
    """
    results: list[dict[str, float | str]] = []
    for i, pred in enumerate(predicted_list):
        metrics: dict[str, float | str] = compare_to_gold_standard(pred, gold_standard)  # type: ignore[assignment]
        metrics["method"] = method_names[i] if method_names else f"method_{i}"
        results.append(metrics)
    return results
