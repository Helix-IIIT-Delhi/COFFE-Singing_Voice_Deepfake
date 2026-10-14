"""Evaluation Metrics for Singing Voice Deepfake Source Attribution"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def compute_attribution_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute attribution accuracy metrics.

    Args:
        predictions: Predicted source IDs
        targets: Ground truth source IDs

    Returns:
        Dictionary with accuracy, precision, recall, F1
    """
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def compute_confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute confusion matrix for source attribution"""
    return confusion_matrix(targets, predictions)
