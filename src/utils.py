import os
import random
import logging

import numpy as np
import torch


def set_seed(seed: int):
    """Set seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Return 'cuda' if GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_output_dir(
    base_dir: str,
    model_name: str,
    dataset_name: str,
    preprocess: bool,
    augment: bool,
    seed: int,
) -> str:
    """Build a systematic output directory path for a single experiment run."""
    # Sanitize model name (e.g., "bert-base-uncased" -> "bert-base-uncased")
    model_short = model_name.replace("/", "_")
    prep_flag = "prep1" if preprocess else "prep0"
    aug_flag = "aug1" if augment else "aug0"
    dir_name = os.path.join(model_short, dataset_name, f"{prep_flag}_{aug_flag}", f"seed_{seed}")
    output_dir = os.path.join("experiments", base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to both console and a file in the output directory."""
    logger = logging.getLogger("cyberbully")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = os.path.join(output_dir, "run.log")
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
