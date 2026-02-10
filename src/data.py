import os
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from preprocessing import preprocess_dataframe

logger = logging.getLogger("cyberbully")

DATASET_CONFIGS = {
    "ieee": {
        "file": "IEEE Data Port.csv",
        "text_col": "Tweet",
        "label_col": "Class",
        "encoding": "iso-8859-1",
    },
    "kaggle": {
        "file": "Kaggle.csv",
        "text_col": "tweet_text",
        "label_col": "cyberbullying_type",
        "encoding": "utf-8",
    },
    "tweeteval": {
        "file": "Tweeteval.csv",
        "text_col": "text",
        "label_col": "label",
        "encoding": "iso-8859-1",
    },
}


def load_dataset(dataset_name: str, data_dir: str) -> pd.DataFrame:
    """Load a dataset and normalize columns to 'text' and 'label' (string)."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset_name]
    filepath = os.path.join(data_dir, config["file"])
    logger.info("Loading dataset '%s' from %s", dataset_name, filepath)

    df = pd.read_csv(filepath, encoding=config["encoding"])
    df = df.rename(
        columns={config["text_col"]: "text", config["label_col"]: "label"}
    )
    df = df[["text", "label"]].copy()

    # Drop rows with missing text or label
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

    # Remove exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    logger.info(
        "Loaded %d samples with %d classes: %s",
        len(df),
        df["label"].nunique(),
        sorted(df["label"].unique().tolist()),
    )
    return df


def encode_labels(df: pd.DataFrame):
    """Encode string labels to integers. Returns df, label2id, id2label."""
    unique_labels = sorted(df["label"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    df = df.copy()
    df["label"] = df["label"].map(label2id).astype(int)

    logger.info("Label mapping: %s", label2id)
    return df, label2id, id2label


def split_data(
    df: pd.DataFrame,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
):
    """Stratified split into train, validation, and test DataFrames."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df["label"],
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def augment_training_data(train_df: pd.DataFrame) -> pd.DataFrame:
    """Apply RandomOverSampler to balance the training set."""
    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_resample(
        np.array(train_df["text"]).reshape(-1, 1),
        np.array(train_df["label"]).reshape(-1, 1),
    )
    augmented_df = pd.DataFrame(
        {"text": X_resampled.flatten(), "label": y_resampled.flatten()}
    )

    unique, counts = np.unique(augmented_df["label"], return_counts=True)
    logger.info(
        "After oversampling — class distribution: %s",
        dict(zip(unique.tolist(), counts.tolist())),
    )
    return augmented_df


def prepare_data(
    dataset_name: str,
    data_dir: str,
    seed: int,
    preprocess: bool = False,
    augment: bool = False,
):
    """Full data preparation pipeline: load, preprocess, encode, split, augment."""
    df = load_dataset(dataset_name, data_dir)

    if preprocess:
        logger.info("Applying text preprocessing...")
        df = preprocess_dataframe(df)
        logger.info("After preprocessing: %d samples remain", len(df))

    df, label2id, id2label = encode_labels(df)
    train_df, val_df, test_df = split_data(df, seed)

    if augment:
        logger.info("Applying training data augmentation (oversampling)...")
        train_df = augment_training_data(train_df)

    return train_df, val_df, test_df, label2id, id2label
