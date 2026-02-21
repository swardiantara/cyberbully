import logging
import math

from huggingface_hub import model_info
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
)
from torch.utils.data import DataLoader

from utils import set_seed

logger = logging.getLogger("cyberbully")

KNOWN_DATASETS = ["ieee", "kaggle", "tweeteval"]


def check_model_exists(model_path: str) -> bool:
    """Check if a model exists on HuggingFace Hub."""
    try:
        model_info(model_path)
        logger.info("Model '%s' found on HuggingFace Hub.", model_path)
        return True
    except Exception:
        logger.info("Model '%s' not found on HuggingFace Hub.", model_path)
        return False


def parse_custom_model_path(model_path: str):
    """Parse a custom model path into (dataset_name, base_model_name).

    Example: 'swardiantara/ieee-all-MiniLM-L6-v2' -> ('ieee', 'all-MiniLM-L6-v2')
    """
    repo_name = model_path.split("/")[-1]
    for ds in KNOWN_DATASETS:
        if repo_name.startswith(ds + "-"):
            base_model = repo_name[len(ds) + 1:]
            return ds, base_model
    raise ValueError(
        f"Cannot parse dataset from model path '{model_path}'. "
        f"Expected format: org/{{dataset}}-{{base_model}} "
        f"where dataset is one of {KNOWN_DATASETS}."
    )


def contrastive_finetune(
    base_model_name: str,
    train_df,
    push_model_path: str,
    seed: int,
):
    """Fine-tune a base SBERT model with OnlineContrastiveLoss and push to HF.

    Uses cosine distance with margin=0.5. Mines hard positive/negative pairs
    within each batch (no pre-constructed pairs needed).
    """
    set_seed(seed)
    logger.info(
        "Starting contrastive fine-tuning: base='%s', push_to='%s', "
        "train_samples=%d",
        base_model_name, push_model_path, len(train_df),
    )

    model = SentenceTransformer(base_model_name)

    # Create InputExamples: one per sample with text and integer label
    train_examples = [
        InputExample(texts=[row["text"]], label=int(row["label"]))
        for _, row in train_df.iterrows()
    ]

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=256,
    )

    train_loss = losses.OnlineContrastiveLoss(
        model=model,
        distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
        margin=0.5,
    )

    epochs = 3
    total_steps = len(train_dataloader) * epochs
    warmup_steps = math.ceil(total_steps * 0.1)

    logger.info(
        "Contrastive training config: epochs=%d, batch_size=256, "
        "lr=2e-5, warmup_steps=%d, total_steps=%d",
        epochs, warmup_steps, total_steps,
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": 2e-5},
        show_progress_bar=True,
    )

    logger.info("Contrastive fine-tuning complete. Pushing to '%s'...", push_model_path)
    model.push_to_hub(push_model_path)
    logger.info("Model pushed to HuggingFace Hub: '%s'", push_model_path)
