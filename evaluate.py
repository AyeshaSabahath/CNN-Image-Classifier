"""
Model evaluation on the held-out test set.

Computes accuracy, precision, recall, F1, confusion matrix, and classification report.
Saves confusion_matrix.png under images/.

Run:
    python evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from preprocess import load_dataset
from utils import (
    DATASET_CHOICE,
    IMAGES_DIR,
    MODEL_PATH,
    MODELS_DIR,
    ensure_directories,
    get_dataset_config,
    setup_logging,
)

logger = setup_logging("evaluate")


def load_trained_model(model_path: Path = MODEL_PATH) -> tf.keras.Model:
    """Load a saved Keras model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train first with: python train.py"
        )
    logger.info("Loading model from %s", model_path)
    # Reuse predict.load_model so compile metrics are restored consistently
    from predict import load_model as _load

    return _load(model_path)


def _predictions_from_probs(probs: np.ndarray, is_binary: bool) -> np.ndarray:
    """Convert raw model outputs to integer class labels."""
    probs = np.asarray(probs)
    if is_binary:
        # Shape (N, 1) or (N,)
        flat = probs.reshape(-1)
        return (flat >= 0.5).astype(int)
    return np.argmax(probs, axis=-1)


def collect_test_predictions(
    model: tf.keras.Model,
    dataset_choice: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Run inference on the test split and return (y_true, y_pred, y_prob, config).
    """
    config = get_dataset_config(dataset_choice)
    data = load_dataset(config["name"], use_augmentation=False)

    if config["name"] == "fashion_mnist":
        _, _, _, _, x_test, y_test, config, _ = data
        y_prob = model.predict(x_test, verbose=0)
        y_true = np.asarray(y_test).reshape(-1)
    else:
        _, _, _, _, test_gen, _, config, _ = data
        # Ensure deterministic order
        test_gen.reset()
        y_prob = model.predict(test_gen, verbose=0)
        y_true = test_gen.classes

    y_pred = _predictions_from_probs(y_prob, config["is_binary"])
    return y_true, y_pred, y_prob, config


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    is_binary: bool,
) -> dict:
    """Compute classification metrics as a dictionary."""
    average = "binary" if is_binary else "weighted"
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    metrics["classification_report"] = report
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_path: Path = None,
) -> Path:
    """Render and save a confusion-matrix heatmap."""
    if output_path is None:
        output_path = IMAGES_DIR / "confusion_matrix.png"

    ensure_directories()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.6)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", output_path)
    return output_path


def save_metrics_json(metrics: dict, path: Path = None) -> Path:
    """Persist scalar metrics (without the long report text duplication)."""
    if path is None:
        path = MODELS_DIR / "evaluation_metrics.json"
    payload = {k: v for k, v in metrics.items() if k != "classification_report"}
    payload["classification_report"] = metrics.get("classification_report", "")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved metrics → %s", path)
    return path


def evaluate(dataset_choice: Optional[str] = None, model_path: Path = MODEL_PATH) -> dict:
    """
    Full evaluation pipeline.

    Returns metrics dict including accuracy, precision, recall, F1, and report.
    """
    ensure_directories()
    choice = dataset_choice or DATASET_CHOICE
    model = load_trained_model(model_path)
    y_true, y_pred, _, config = collect_test_predictions(model, choice)

    metrics = compute_metrics(
        y_true,
        y_pred,
        config["class_names"],
        config["is_binary"],
    )
    plot_confusion_matrix(y_true, y_pred, config["class_names"])
    save_metrics_json(metrics)

    logger.info("Accuracy : %.4f", metrics["accuracy"])
    logger.info("Precision: %.4f", metrics["precision"])
    logger.info("Recall   : %.4f", metrics["recall"])
    logger.info("F1 Score : %.4f", metrics["f1_score"])
    logger.info("\n%s", metrics["classification_report"])

    # Optional tabular summary for notebooks / CLI
    summary = pd.DataFrame(
        [
            {
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1_score"],
            }
        ]
    )
    logger.info("\n%s", summary.to_string(index=False))
    return metrics


def main() -> None:
    try:
        evaluate()
    except Exception:
        logger.exception("Evaluation failed.")
        raise


if __name__ == "__main__":
    main()
