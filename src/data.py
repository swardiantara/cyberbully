import json
import logging
import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

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

    if dataset_name == "kaggle":
        df = df[df["label"] != "not_cyberbullying"]  # Drop 'not_cyberbullying' rows

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


def split_data(
    df: pd.DataFrame,
    seed: int = 2042,  # similar to the reference paper
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
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


def augment_training_data(train_df: pd.DataFrame) -> pd.DataFrame:
    """Apply RandomOverSampler to balance the training set."""
    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_resample(
        np.array(train_df["cleansed"]).reshape(-1, 1),
        np.array(train_df["label"]).reshape(-1, 1),
    )
    texts = X_resampled.flatten()
    augmented_df = pd.DataFrame(
        {"text": texts, "cleansed": texts, "label": y_resampled.flatten()}
    )

    unique, counts = np.unique(augmented_df["label"], return_counts=True)
    logger.info(
        "After oversampling -- class distribution: %s",
        dict(zip(unique.tolist(), counts.tolist())),
    )
    return augmented_df


# ---------------------------------------------------------------------------
# Split persistence helpers
# ---------------------------------------------------------------------------

def load_raw_test_texts(data_dir: str, dataset_name: str) -> list:
    """Return the original (pre-preprocessing) test texts from the saved split.

    Reads directly from the on-disk test.csv, which is written before any
    preprocessing is applied, so the texts are always in their raw form.
    Returns None if the split files do not exist yet.
    """
    split_dir = _split_dir(data_dir, dataset_name)
    test_path = os.path.join(split_dir, "test.csv")
    if not os.path.exists(test_path):
        return None
    return pd.read_csv(test_path)["text"].tolist()

def _split_dir(data_dir: str, dataset_name: str) -> str:
    return os.path.join(data_dir, dataset_name)


def _splits_exist(data_dir: str, dataset_name: str) -> bool:
    split_dir = _split_dir(data_dir, dataset_name)
    return all(
        os.path.exists(os.path.join(split_dir, fname))
        for fname in ["train.csv", "val.csv", "test.csv", "label_mapping.json"]
    )


def _save_splits(
    data_dir: str,
    dataset_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label2id: dict,
):
    split_dir = _split_dir(data_dir, dataset_name)
    os.makedirs(split_dir, exist_ok=True)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df.to_csv(os.path.join(split_dir, f"{name}.csv"), index=False)
    with open(os.path.join(split_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)
    logger.info("Saved raw splits to %s", split_dir)


def _load_splits(data_dir: str, dataset_name: str):
    split_dir = _split_dir(data_dir, dataset_name)
    train_df = pd.read_csv(os.path.join(split_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(split_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test.csv"))
    with open(os.path.join(split_dir, "label_mapping.json"), encoding="utf-8") as f:
        label2id = json.load(f)
    logger.info(
        "Loaded pre-split data from %s — train: %d, val: %d, test: %d",
        split_dir, len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df, label2id


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare_data(
    dataset_name: str,
    data_dir: str,
    preprocess: bool = False,
    augment: bool = False,
):
    """Full data preparation pipeline.

    On the first run, the raw dataset is loaded, split stratified by label,
    and saved to {data_dir}/{dataset_name}/{{train,val,test}}.csv along with
    label_mapping.json.  Subsequent runs reuse these fixed splits, ensuring
    every experiment operates on identical train/val/test sets regardless of
    preprocessing or augmentation settings.

    Preprocessing (optional) is applied to each split independently after
    loading, so the split boundaries are always determined on raw text.
    """
    if _splits_exist(data_dir, dataset_name):
        train_df, val_df, test_df, label2id = _load_splits(data_dir, dataset_name)
    else:
        logger.info("No pre-split data found. Splitting and saving for future runs...")
        df = load_dataset(dataset_name, data_dir)
        unique_labels = sorted(df["label"].unique().tolist())
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        train_df, val_df, test_df = split_data(df)
        _save_splits(data_dir, dataset_name, train_df, val_df, test_df, label2id)
    raw_train, raw_val, raw_test = train_df, val_df, test_df
    id2label = {idx: label for label, idx in label2id.items()}
    logger.info("Label mapping: %s", label2id)

    # Attach raw text before any preprocessing so it stays aligned with the
    # surviving rows after preprocess_dataframe filters/deduplicates.
    for df in (train_df, val_df, test_df):
        df["raw"] = df["text"]

    # Preprocessing is applied per-split to avoid data leakage and to preserve
    # the original distribution used for splitting.
    if preprocess:
        logger.info("Applying text preprocessing per split...")
        train_df = preprocess_dataframe(train_df)
        val_df = preprocess_dataframe(val_df)
        test_df = preprocess_dataframe(test_df)
        logger.info(
            "After preprocessing -- train: %d, val: %d, test: %d",
            len(train_df), len(val_df), len(test_df),
        )

    # check for sample size consistency after preprocessing (should be >= original split sizes)
    assert len(raw_train) == len(train_df), "Preprocessing removed too many training samples!"
    assert len(raw_val) == len(val_df), "Preprocessing removed too many validation samples!"
    assert len(raw_test) == len(test_df), "Preprocessing removed too many test samples!"

    # cleansed == raw when preprocessing is disabled; always present for
    # consistent downstream use (training, logging, attribution).
    for df in (train_df, val_df, test_df):
        df["cleansed"] = df["text"]

    # Encode string labels to integers using the saved mapping
    for df in (train_df, val_df, test_df):
        df["label"] = df["label"].map(label2id).astype(int)

    if augment:
        logger.info("Applying training data augmentation (oversampling)...")
        train_df = augment_training_data(train_df)

    return train_df, val_df, test_df, label2id, id2label
