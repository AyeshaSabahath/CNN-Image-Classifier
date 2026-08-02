"""
Utility helpers for the Image Classification CNN project.

Central configuration, logging, Grad-CAM visualization, and shared paths.
Change DATASET_CHOICE below to switch between Fashion-MNIST and Cats vs Dogs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Dataset selection — change this single variable to switch datasets
# Options: "fashion_mnist" | "cats_dogs"
# ---------------------------------------------------------------------------
DATASET_CHOICE = "fashion_mnist"

# Project root (directory containing this file)
PROJECT_ROOT = Path(__file__).resolve().parent

# Standard directories
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"
MODEL_PATH = MODELS_DIR / "cnn_model.h5"

# Training hyperparameters
EPOCHS = 15
BATCH_SIZE = 32
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Dataset-specific settings
FASHION_MNIST_CONFIG = {
    "name": "fashion_mnist",
    "img_size": (28, 28),
    "channels": 1,
    "num_classes": 10,
    "class_names": [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ],
    "is_binary": False,
}

CATS_DOGS_CONFIG = {
    "name": "cats_dogs",
    "img_size": (128, 128),
    "channels": 3,
    "num_classes": 2,
    "class_names": ["Cat", "Dog"],
    "is_binary": True,
}

# Fashion-MNIST label map for quick lookup
FASHION_LABEL_MAP = {i: name for i, name in enumerate(FASHION_MNIST_CONFIG["class_names"])}


def get_dataset_config(dataset_choice: Optional[str] = None) -> dict:
    """Return configuration dict for the selected dataset."""
    choice = (dataset_choice or DATASET_CHOICE).lower().strip()
    if choice in ("fashion_mnist", "fashion-mnist", "fashionmnist"):
        return FASHION_MNIST_CONFIG.copy()
    if choice in ("cats_dogs", "cats-dogs", "cat_dog", "cats_vs_dogs"):
        return CATS_DOGS_CONFIG.copy()
    raise ValueError(
        f"Unknown dataset '{choice}'. Use 'fashion_mnist' or 'cats_dogs'."
    )


def setup_logging(name: str = "image_classifier", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a named logger.

    Logs go to both console and a rotating-friendly file under images/.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_file = IMAGES_DIR / "app.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def ensure_directories() -> None:
    """Create all required project directories if they do not exist."""
    for path in (
        DATASET_DIR,
        TRAIN_DIR,
        VALIDATION_DIR,
        TEST_DIR,
        MODELS_DIR,
        IMAGES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1]."""
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    channels: int = 3,
) -> np.ndarray:
    """
    Resize an image to the target (height, width) and adjust channel count.

    Args:
        image: Input image (H, W) or (H, W, C).
        size: Target (height, width).
        channels: Desired channel count (1 or 3).
    """
    h, w = size
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

    if channels == 1:
        if resized.ndim == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        resized = np.expand_dims(resized, axis=-1)
    elif channels == 3:
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[-1] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
        elif resized.shape[-1] == 3:
            # Assume BGR from OpenCV; convert to RGB for Keras consistency
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    return resized


def load_and_preprocess_image(
    image_path: str | Path,
    img_size: Tuple[int, int],
    channels: int,
) -> np.ndarray:
    """
    Load an image from disk, resize, normalize, and add batch dimension.

    Returns:
        Array of shape (1, H, W, C) with values in [0, 1].
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")

    image = resize_image(image, img_size, channels)
    image = normalize_image(image)
    return np.expand_dims(image, axis=0)


def preprocess_array_image(
    image: np.ndarray,
    img_size: Tuple[int, int],
    channels: int,
) -> np.ndarray:
    """
    Preprocess an in-memory image (e.g. from Streamlit / webcam).

    Accepts RGB or grayscale arrays from PIL / OpenCV.
    Returns shape (1, H, W, C) normalized to [0, 1].
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        # grayscale — OpenCV path expects BGR-like; resize_image handles gray
        pass
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        # Streamlit / PIL give RGB; convert to BGR so resize_image's BGR→RGB is correct
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    arr = resize_image(arr, img_size, channels)
    arr = normalize_image(arr)
    return np.expand_dims(arr, axis=0)


def make_gradcam_heatmap(
    model,
    img_array: np.ndarray,
    last_conv_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a single preprocessed image batch.

    Uses a layer-by-layer forward pass so gradients remain connected for
    Sequential models saved/loaded as HDF5 under Keras 3.

    Args:
        model: Trained Keras model.
        img_array: Preprocessed image of shape (1, H, W, C).
        last_conv_layer_name: Optional Conv layer name; auto-detected if None.
        pred_index: Class index to explain; uses top prediction if None.

    Returns:
        2D heatmap normalized to [0, 1].
    """
    import tensorflow as tf
    from tensorflow.keras.layers import InputLayer

    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        x = img_tensor
        conv_outputs = None
        for layer in model.layers:
            if isinstance(layer, InputLayer):
                continue
            x = layer(x, training=False)
            if layer.name == last_conv_layer_name:
                conv_outputs = x
                tape.watch(conv_outputs)

        predictions = x
        if conv_outputs is None:
            raise ValueError(f"Layer '{last_conv_layer_name}' was not reached during forward pass.")

        if pred_index is None:
            if predictions.shape[-1] == 1:
                class_channel = predictions[:, 0]
            else:
                pred_index = int(tf.argmax(predictions[0]))
                class_channel = predictions[:, pred_index]
        else:
            if predictions.shape[-1] == 1:
                class_channel = predictions[:, 0]
            else:
                class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("Grad-CAM gradients are None — check model connectivity.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def find_last_conv_layer(model) -> str:
    """Find the name of the last Conv2D layer in the model."""
    for layer in reversed(model.layers):
        if "Conv2D" in layer.__class__.__name__:
            return layer.name
    raise ValueError("No Conv2D layer found in the model for Grad-CAM.")


def overlay_gradcam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap onto an RGB image.

    Args:
        original_image: RGB uint8 or float image.
        heatmap: 2D Grad-CAM map in [0, 1].
        alpha: Heatmap blend strength.

    Returns:
        RGB uint8 overlay image.
    """
    img = original_image.copy()
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 1:
        img = cv2.cvtColor(img.squeeze(-1), cv2.COLOR_GRAY2RGB)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    return overlay


def format_confidence(probability: float) -> str:
    """Format a probability as a percentage string."""
    return f"{probability * 100:.2f}%"


def get_class_label(prediction, config: dict) -> Tuple[str, float]:
    """
    Convert raw model output into (class_name, confidence).

    Handles both binary sigmoid and multi-class softmax outputs.
    """
    probs = np.asarray(prediction).reshape(-1)

    if config["is_binary"]:
        # Sigmoid output: single unit → Dog if >= 0.5 else Cat
        conf_dog = float(probs[0])
        if conf_dog >= 0.5:
            return config["class_names"][1], conf_dog
        return config["class_names"][0], 1.0 - conf_dog

    idx = int(np.argmax(probs))
    return config["class_names"][idx], float(probs[idx])


def get_probability_dict(prediction, config: dict) -> dict:
    """Return a mapping of class_name → probability for charting."""
    probs = np.asarray(prediction).reshape(-1)

    if config["is_binary"]:
        p_dog = float(probs[0])
        return {
            config["class_names"][0]: 1.0 - p_dog,
            config["class_names"][1]: p_dog,
        }

    return {name: float(probs[i]) for i, name in enumerate(config["class_names"])}
