"""
Single-image prediction module.

Usage:
    python predict.py path/to/image.jpg
    python predict.py path/to/image.jpg --dataset fashion_mnist
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from utils import (
    DATASET_CHOICE,
    MODEL_PATH,
    MODELS_DIR,
    format_confidence,
    get_class_label,
    get_dataset_config,
    get_probability_dict,
    load_and_preprocess_image,
    make_gradcam_heatmap,
    overlay_gradcam,
    preprocess_array_image,
    setup_logging,
)

logger = setup_logging("predict")


def load_model(model_path: Path = MODEL_PATH) -> tf.keras.Model:
    """Load the trained CNN from disk and ensure it is compiled for inference."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run: python train.py"
        )
    model = tf.keras.models.load_model(str(model_path))
    # HDF5 reloads sometimes drop compiled metrics under Keras 3 — restore safely.
    config = resolve_config()
    loss = "binary_crossentropy" if config["is_binary"] else "sparse_categorical_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def load_metadata() -> dict:
    """Load training metadata if available; fall back to DATASET_CHOICE config."""
    meta_path = MODELS_DIR / "training_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return get_dataset_config(DATASET_CHOICE)


def resolve_config(dataset_choice: Optional[str] = None) -> dict:
    """Prefer metadata from the last training run when dataset is not forced."""
    if dataset_choice:
        return get_dataset_config(dataset_choice)
    meta = load_metadata()
    if "name" in meta or "dataset" in meta:
        name = meta.get("name") or meta.get("dataset")
        cfg = get_dataset_config(name)
        # Prefer persisted class names / sizes if present
        for key in ("class_names", "img_size", "channels", "num_classes", "is_binary"):
            if key in meta:
                cfg[key] = meta[key]
                if key == "img_size":
                    cfg[key] = tuple(meta[key])
        return cfg
    return get_dataset_config(DATASET_CHOICE)


def predict_image(
    image_path: str | Path,
    model: Optional[tf.keras.Model] = None,
    dataset_choice: Optional[str] = None,
    model_path: Path = MODEL_PATH,
) -> dict:
    """
    Predict the class of an image file.

    Returns:
        dict with keys: predicted_class, confidence, confidence_pct,
        probabilities, inference_time_ms, raw_prediction
    """
    config = resolve_config(dataset_choice)
    if model is None:
        model = load_model(model_path)

    img_size = tuple(config["img_size"])
    channels = int(config["channels"])

    batch = load_and_preprocess_image(image_path, img_size, channels)

    start = time.perf_counter()
    raw = model.predict(batch, verbose=0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    label, confidence = get_class_label(raw[0], config)
    probabilities = get_probability_dict(raw[0], config)

    result = {
        "predicted_class": label,
        "confidence": confidence,
        "confidence_pct": format_confidence(confidence),
        "probabilities": probabilities,
        "inference_time_ms": round(elapsed_ms, 2),
        "raw_prediction": np.asarray(raw[0]).tolist(),
        "image_path": str(image_path),
        "dataset": config.get("name", config.get("dataset", "unknown")),
    }
    logger.info(
        "Prediction: %s (%s) in %.2f ms",
        result["predicted_class"],
        result["confidence_pct"],
        result["inference_time_ms"],
    )
    return result


def predict_array(
    image: np.ndarray,
    model: Optional[tf.keras.Model] = None,
    dataset_choice: Optional[str] = None,
    model_path: Path = MODEL_PATH,
) -> dict:
    """
    Predict from an in-memory image array (RGB/grayscale from Streamlit or webcam).
    """
    config = resolve_config(dataset_choice)
    if model is None:
        model = load_model(model_path)

    img_size = tuple(config["img_size"])
    channels = int(config["channels"])
    batch = preprocess_array_image(image, img_size, channels)

    start = time.perf_counter()
    raw = model.predict(batch, verbose=0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    label, confidence = get_class_label(raw[0], config)
    probabilities = get_probability_dict(raw[0], config)

    return {
        "predicted_class": label,
        "confidence": confidence,
        "confidence_pct": format_confidence(confidence),
        "probabilities": probabilities,
        "inference_time_ms": round(elapsed_ms, 2),
        "raw_prediction": np.asarray(raw[0]).tolist(),
        "preprocessed_batch": batch,
        "dataset": config.get("name", "unknown"),
        "config": config,
    }


def predict_with_gradcam(
    image: np.ndarray,
    model: Optional[tf.keras.Model] = None,
    dataset_choice: Optional[str] = None,
    model_path: Path = MODEL_PATH,
) -> Tuple[dict, np.ndarray]:
    """
    Run prediction and return (result_dict, gradcam_overlay_rgb).
    """
    import cv2

    result = predict_array(image, model, dataset_choice, model_path)
    config = result["config"]
    batch = result["preprocessed_batch"]

    if model is None:
        model = load_model(model_path)

    heatmap = make_gradcam_heatmap(model, batch)

    # Build a displayable original at model resolution
    display = batch[0].copy()
    if display.shape[-1] == 1:
        display = np.repeat(display, 3, axis=-1)
    display_uint8 = (np.clip(display, 0, 1) * 255).astype(np.uint8)
    overlay = overlay_gradcam(display_uint8, heatmap)
    result.pop("preprocessed_batch", None)
    result.pop("config", None)
    return result, overlay


def capture_webcam_frame(camera_index: int = 0) -> np.ndarray:
    """
    Capture a single frame from the default webcam using OpenCV.

    Returns:
        RGB image array.
    """
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open webcam. Check that a camera is connected and not in use."
        )
    try:
        # Warm up a few frames for auto-exposure
        frame = None
        for _ in range(5):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from webcam.")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict image class with the trained CNN.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="fashion_mnist or cats_dogs (defaults to last training run)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH),
        help="Path to the .h5 model file",
    )
    args = parser.parse_args()

    try:
        result = predict_image(args.image_path, dataset_choice=args.dataset, model_path=Path(args.model))
        print("\n========== Prediction Result ==========")
        print(f"Image           : {result['image_path']}")
        print(f"Predicted Class : {result['predicted_class']}")
        print(f"Confidence      : {result['confidence_pct']}")
        print(f"Inference Time  : {result['inference_time_ms']} ms")
        print("Probabilities   :")
        for name, prob in result["probabilities"].items():
            print(f"  - {name}: {prob * 100:.2f}%")
        print("=======================================\n")
    except Exception:
        logger.exception("Prediction failed.")
        raise


if __name__ == "__main__":
    main()
