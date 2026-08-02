"""
CNN model definition and training pipeline.

Run:
    python train.py

Trains for 15 epochs, saves the best model to models/cnn_model.h5,
and writes accuracy/loss curves to images/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from preprocess import fashion_augmented_generator, load_dataset
from utils import (
    BATCH_SIZE,
    DATASET_CHOICE,
    EPOCHS,
    IMAGES_DIR,
    MODEL_PATH,
    MODELS_DIR,
    ensure_directories,
    get_dataset_config,
    set_seeds,
    setup_logging,
)

logger = setup_logging("train")


def build_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    is_binary: bool,
) -> keras.Model:
    """
    Build the project CNN with the Sequential API.

    Architecture:
        Input → Conv2D(32) → ReLU → MaxPool
              → Conv2D(64) → ReLU → MaxPool
              → Conv2D(128) → ReLU → MaxPool
              → Flatten → Dense(256) → Dropout(0.5) → Dense(output)
    """
    model = keras.Sequential(name="image_classifier_cnn")
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(32, (3, 3), padding="same", name="conv2d_32"))
    model.add(layers.Activation("relu", name="relu_1"))
    model.add(layers.MaxPooling2D((2, 2), name="maxpool_1"))

    model.add(layers.Conv2D(64, (3, 3), padding="same", name="conv2d_64"))
    model.add(layers.Activation("relu", name="relu_2"))
    model.add(layers.MaxPooling2D((2, 2), name="maxpool_2"))

    model.add(layers.Conv2D(128, (3, 3), padding="same", name="conv2d_128"))
    model.add(layers.Activation("relu", name="relu_3"))
    model.add(layers.MaxPooling2D((2, 2), name="maxpool_3"))

    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(256, activation="relu", name="dense_256"))
    model.add(layers.Dropout(0.5, name="dropout_0_5"))

    if is_binary:
        model.add(layers.Dense(1, activation="sigmoid", name="output"))
        loss = "binary_crossentropy"
    else:
        model.add(layers.Dense(num_classes, activation="softmax", name="output"))
        loss = "sparse_categorical_crossentropy"

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def plot_history(history: keras.callbacks.History, output_dir: Path = IMAGES_DIR) -> None:
    """Save training accuracy and loss curves as PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hist = history.history

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(hist.get("accuracy", []), label="Train Accuracy", linewidth=2)
    if "val_accuracy" in hist:
        plt.plot(hist["val_accuracy"], label="Val Accuracy", linewidth=2)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = output_dir / "training_accuracy.png"
    plt.savefig(acc_path, dpi=150)
    plt.close()
    logger.info("Saved accuracy plot → %s", acc_path)

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(hist.get("loss", []), label="Train Loss", linewidth=2)
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="Val Loss", linewidth=2)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = output_dir / "training_loss.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()
    logger.info("Saved loss plot → %s", loss_path)


def save_training_metadata(
    config: dict,
    history: keras.callbacks.History,
    best_val_acc: float,
) -> None:
    """Persist lightweight metadata for the Streamlit dashboard."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": config["name"],
        "class_names": config["class_names"],
        "img_size": list(config["img_size"]),
        "channels": config["channels"],
        "num_classes": config["num_classes"],
        "is_binary": config["is_binary"],
        "best_val_accuracy": float(best_val_acc),
        "epochs_trained": len(history.history.get("loss", [])),
        "final_train_accuracy": float(history.history.get("accuracy", [0])[-1]),
        "final_val_accuracy": float(history.history.get("val_accuracy", [0])[-1]),
    }
    meta_path = MODELS_DIR / "training_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved training metadata → %s", meta_path)


def get_callbacks(model_path: Path = MODEL_PATH) -> list:
    """ModelCheckpoint + EarlyStopping + ReduceLROnPlateau."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        mode="max",
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )
    return [checkpoint, early_stop, reduce_lr]


def train_fashion_mnist(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    use_augmentation: bool = True,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Train CNN on Fashion-MNIST arrays."""
    x_train, y_train, x_val, y_val, x_test, y_test, config, _ = load_dataset(
        "fashion_mnist",
        use_augmentation=use_augmentation,
        batch_size=batch_size,
    )
    input_shape = (
        config["img_size"][0],
        config["img_size"][1],
        config["channels"],
    )
    model = build_cnn(input_shape, config["num_classes"], config["is_binary"])
    model.summary(print_fn=logger.info)

    callbacks = get_callbacks()

    if use_augmentation:
        train_flow = fashion_augmented_generator(x_train, y_train, batch_size)
        steps = int(np.ceil(len(x_train) / batch_size))
        history = model.fit(
            train_flow,
            steps_per_epoch=steps,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )

    # Ensure best weights are on disk (EarlyStopping may restore in-memory)
    model.save(str(MODEL_PATH))
    logger.info("Model saved → %s", MODEL_PATH)

    # Also stash test arrays path hint via metadata; evaluation loads data itself
    best_val = max(history.history.get("val_accuracy", [0.0]))
    plot_history(history)
    save_training_metadata(config, history, best_val)

    # Quick test peek
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    logger.info("Test accuracy (quick check): %.4f | loss: %.4f", test_acc, test_loss)
    return model, history


def train_cats_dogs(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    use_augmentation: bool = True,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Train CNN on Cats vs Dogs directory generators."""
    train_gen, _, val_gen, _, test_gen, _, config, _ = load_dataset(
        "cats_dogs",
        use_augmentation=use_augmentation,
        batch_size=batch_size,
    )
    input_shape = (
        config["img_size"][0],
        config["img_size"][1],
        config["channels"],
    )
    model = build_cnn(input_shape, config["num_classes"], config["is_binary"])
    model.summary(print_fn=logger.info)

    callbacks = get_callbacks()
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(str(MODEL_PATH))
    logger.info("Model saved → %s", MODEL_PATH)

    best_val = max(history.history.get("val_accuracy", [0.0]))
    plot_history(history)
    save_training_metadata(config, history, best_val)

    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    logger.info("Test accuracy (quick check): %.4f | loss: %.4f", test_acc, test_loss)
    return model, history


def train(
    dataset_choice: Optional[str] = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    use_augmentation: bool = True,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Entry-point trainer.

    Selects Fashion-MNIST or Cats vs Dogs based on DATASET_CHOICE / argument.
    """
    set_seeds()
    ensure_directories()
    choice = (dataset_choice or DATASET_CHOICE).lower()
    config = get_dataset_config(choice)
    logger.info("Starting training for dataset: %s", config["name"])
    logger.info("TensorFlow version: %s", tf.__version__)

    if config["name"] == "fashion_mnist":
        return train_fashion_mnist(epochs, batch_size, use_augmentation)
    return train_cats_dogs(epochs, batch_size, use_augmentation)


def main() -> None:
    """CLI entry point."""
    try:
        train()
        logger.info("Training completed successfully.")
    except Exception:
        logger.exception("Training failed.")
        raise


if __name__ == "__main__":
    main()
