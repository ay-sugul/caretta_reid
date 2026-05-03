"""Evaluation metrics for retrieval and classification tasks."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def top_k_accuracy(predictions: Sequence[Sequence[str]], targets: Sequence[str], k: int) -> float:
    """Computes the fraction of targets that appear in the top-k predictions."""

    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")
    if k <= 0:
        raise ValueError("k must be positive.")

    hits = 0
    for candidate_list, target in zip(predictions, targets, strict=True):
        hits += int(target in list(candidate_list)[:k])
    return hits / max(len(targets), 1)


def roc_auc_score_binary(scores: Sequence[float], targets: Sequence[int]) -> float:
    """Computes the binary ROC-AUC score without external dependencies."""

    if len(scores) != len(targets):
        raise ValueError("Scores and targets must have the same length.")

    score_array = np.asarray(scores, dtype=float)
    target_array = np.asarray(targets, dtype=int)
    positive_count = int(target_array.sum())
    negative_count = int(len(target_array) - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Both classes must be present for ROC-AUC.")

    order = np.argsort(score_array)
    ranked_targets = target_array[order]
    positive_ranks = np.where(ranked_targets == 1)[0] + 1
    auc = (positive_ranks.sum() - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )
    return float(auc)
