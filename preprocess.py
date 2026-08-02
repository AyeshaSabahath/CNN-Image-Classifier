"""
Data loading, preprocessing, and augmentation for the CNN image classifier.

Supports Fashion-MNIST (via Keras) and Cats vs Dogs (local folders or auto-download).
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import (
    CATS_DOGS_CONFIG,
    DATASET_DIR,
    RANDOM_SEED,
    TEST_DIR,
    TEST_SPLIT,
    TRAIN_DIR,
    VALIDATION_DIR,
    VALIDATION_SPLIT,
    ensure_directories,
    get_dataset_config,
    normalize_image,
    set_seeds,
    setup_logging,
)

logger = setup_logging("preprocess")

# Microsoft Cats & Dogs zip (same images as the popular Kaggle cat-and-dog set)
CATS_DOGS_URL = (
    "https://download.microsoft.com/download/3/E/1/"
    "3E1C3F21-ECDB-4869-AC1B-ACFF3BD2AA5C/kagglecatsanddogs_5340.zip"
)


def create_augmentation_generator(
    rotation_range: int = 20,
    zoom_range: float = 0.15,
    horizontal_flip: bool = True,
    brightness_range: Optional[Tuple[float, float]] = (0.8, 1.2),
    shear_range: float = 0.15,
    fill_mode: str = "nearest",
) -> ImageDataGenerator:
    """
    Build an ImageDataGenerator with common augmentations.

    Normalization (rescale=1/255) is applied here so generators yield [0, 1] images.
    """
    return ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        brightness_range=brightness_range,
        shear_range=shear_range,
        fill_mode=fill_mode,
    )


def load_fashion_mnist(
    validation_split: float = VALIDATION_SPLIT,
    test_split: float = TEST_SPLIT,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Fashion-MNIST, normalize, reshape to (N, 28, 28, 1), and split.

    Keras already provides a held-out test set. We further carve validation
    from the original training set.
    """
    logger.info("Loading Fashion-MNIST dataset...")
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Combine then re-split so we control train / val / test ratios consistently
    x_all = np.concatenate([x_train_full, x_test], axis=0)
    y_all = np.concatenate([y_train_full, y_test], axis=0)

    x_temp, x_test, y_temp, y_test = train_test_split(
        x_all,
        y_all,
        test_size=test_split,
        random_state=seed,
        stratify=y_all,
    )
    relative_val = validation_split / (1.0 - test_split)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_val,
        random_state=seed,
        stratify=y_temp,
    )

    def _prepare(x: np.ndarray) -> np.ndarray:
        x = normalize_image(x)
        return np.expand_dims(x, axis=-1)

    x_train, x_val, x_test = _prepare(x_train), _prepare(x_val), _prepare(x_test)
    logger.info(
        "Fashion-MNIST splits — train: %s, val: %s, test: %s",
        x_train.shape,
        x_val.shape,
        x_test.shape,
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def _download_cats_dogs(dest: Path) -> Path:
    """Download and extract the Microsoft Cats & Dogs archive."""
    ensure_directories()
    zip_path = dest / "kagglecatsanddogs.zip"
    extract_dir = dest / "PetImages"

    if extract_dir.exists() and any(extract_dir.rglob("*.jpg")):
        logger.info("Cats vs Dogs images already extracted at %s", extract_dir)
        return extract_dir

    logger.info("Downloading Cats vs Dogs dataset (this may take a few minutes)...")
    archive = keras.utils.get_file(
        fname="kagglecatsanddogs_5340.zip",
        origin=CATS_DOGS_URL,
        cache_dir=str(dest),
        cache_subdir=".",
        extract=False,
    )
    archive_path = Path(archive)
    logger.info("Extracting archive...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(dest)

    # Microsoft zip extracts to PetImages/Cat and PetImages/Dog
    if not extract_dir.exists():
        # Fallback: search for Cat/Dog folders
        candidates = list(dest.rglob("Cat"))
        if candidates:
            extract_dir = candidates[0].parent
        else:
            raise FileNotFoundError("Could not locate PetImages after extraction.")

    # Remove corrupt / non-image files that commonly ship with this dataset
    _clean_pet_images(extract_dir)
    return extract_dir


def _clean_pet_images(root: Path) -> None:
    """Delete non-JPEG or truncated files that break Keras image loaders."""
    from PIL import Image

    removed = 0
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if not img_path.is_file():
                continue
            try:
                with Image.open(img_path) as im:
                    im.verify()
                # Re-open after verify (Pillow requirement)
                with Image.open(img_path) as im:
                    im.load()
            except Exception:
                img_path.unlink(missing_ok=True)
                removed += 1
    if removed:
        logger.warning("Removed %d corrupt image files from Cats vs Dogs dataset.", removed)


def _organize_cats_dogs_splits(
    source_root: Path,
    max_per_class: Optional[int] = 4000,
    seed: int = RANDOM_SEED,
) -> None:
    """
    Copy images into dataset/train|validation|test/{cat,dog} folders.

    Uses a stratified split. max_per_class keeps training time reasonable.
    """
    ensure_directories()
    rng = np.random.default_rng(seed)

    for split in (TRAIN_DIR, VALIDATION_DIR, TEST_DIR):
        for cls in ("cat", "dog"):
            (split / cls).mkdir(parents=True, exist_ok=True)

    for src_name, dst_name in (("Cat", "cat"), ("Dog", "dog")):
        src = source_root / src_name
        if not src.exists():
            src = source_root / dst_name
        if not src.exists():
            raise FileNotFoundError(f"Missing class folder: {src_name} under {source_root}")

        files = [
            p
            for p in src.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        rng.shuffle(files)
        if max_per_class is not None:
            files = files[:max_per_class]

        n = len(files)
        n_test = int(n * TEST_SPLIT)
        n_val = int(n * VALIDATION_SPLIT)
        test_files = files[:n_test]
        val_files = files[n_test : n_test + n_val]
        train_files = files[n_test + n_val :]

        for split_dir, split_files in (
            (TRAIN_DIR, train_files),
            (VALIDATION_DIR, val_files),
            (TEST_DIR, test_files),
        ):
            target = split_dir / dst_name
            for f in split_files:
                dest = target / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)

        logger.info(
            "Class '%s' — train: %d, val: %d, test: %d",
            dst_name,
            len(train_files),
            len(val_files),
            len(test_files),
        )


def prepare_cats_dogs_directories(force_download: bool = False) -> None:
    """
    Ensure dataset/train|validation|test contain cat/dog images.

    Downloads the Microsoft archive when local splits are empty.
    """
    train_cats = list((TRAIN_DIR / "cat").glob("*")) if (TRAIN_DIR / "cat").exists() else []
    train_dogs = list((TRAIN_DIR / "dog").glob("*")) if (TRAIN_DIR / "dog").exists() else []

    if train_cats and train_dogs and not force_download:
        logger.info("Cats vs Dogs folder structure already present.")
        return

    source = _download_cats_dogs(DATASET_DIR)
    _organize_cats_dogs_splits(source)


def get_cats_dogs_generators(
    img_size: Tuple[int, int] = CATS_DOGS_CONFIG["img_size"],
    batch_size: int = 32,
    use_augmentation: bool = True,
) -> Tuple[ImageDataGenerator, object, object, object]:
    """
    Create Keras directory iterators for Cats vs Dogs.

    Returns:
        (train_datagen, train_gen, val_gen, test_gen)
    """
    prepare_cats_dogs_directories()

    if use_augmentation:
        train_datagen = create_augmentation_generator()
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    target_size = (img_size[0], img_size[1])
    train_gen = train_datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=RANDOM_SEED,
        color_mode="rgb",
    )
    val_gen = val_test_datagen.flow_from_directory(
        str(VALIDATION_DIR),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
        color_mode="rgb",
    )
    test_gen = val_test_datagen.flow_from_directory(
        str(TEST_DIR),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
        color_mode="rgb",
    )
    logger.info("Class indices: %s", train_gen.class_indices)
    return train_datagen, train_gen, val_gen, test_gen


def load_dataset(
    dataset_choice: Optional[str] = None,
    use_augmentation: bool = True,
    batch_size: int = 32,
):
    """
    High-level loader that returns data ready for training.

    Fashion-MNIST returns NumPy arrays:
        (x_train, y_train, x_val, y_val, x_test, y_test, config, None)

    Cats vs Dogs returns generators:
        (train_gen, None, val_gen, None, test_gen, None, config, train_datagen)
        where y_* slots are unused (labels come from the generators).
    """
    set_seeds()
    ensure_directories()
    config = get_dataset_config(dataset_choice)

    if config["name"] == "fashion_mnist":
        x_train, y_train, x_val, y_val, x_test, y_test = load_fashion_mnist()
        return x_train, y_train, x_val, y_val, x_test, y_test, config, None

    _, train_gen, val_gen, test_gen = get_cats_dogs_generators(
        img_size=config["img_size"],
        batch_size=batch_size,
        use_augmentation=use_augmentation,
    )
    return train_gen, None, val_gen, None, test_gen, None, config, None


def fashion_augmented_generator(
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = 32,
) -> object:
    """Optional ImageDataGenerator flow for Fashion-MNIST arrays."""
    # Brightness augmentation is less meaningful for centered grayscale garments
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        shear_range=0.1,
        fill_mode="nearest",
    )
    return datagen.flow(x_train, y_train, batch_size=batch_size, seed=RANDOM_SEED)
