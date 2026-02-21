import glob
import logging
import os
import shutil

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

logger = logging.getLogger("cyberbully")


class CyberbullyDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that tokenizes raw texts on initialization."""

    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.labels = labels
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_attention_mask=True,
        )

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics_fn(eval_pred):
    """Compute metrics for HuggingFace Trainer during evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
    f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)

    return {
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "f1_micro": f1_micro,
    }


def get_training_args(
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> TrainingArguments:
    """Create HuggingFace TrainingArguments with sensible defaults."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=1e-8,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        save_strategy="epoch",
        metric_for_best_model="f1_weighted",
        logging_steps=50,
        seed=seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )


def train_model(
    model,
    tokenizer,
    train_dataset: CyberbullyDataset,
    val_dataset: CyberbullyDataset,
    training_args: TrainingArguments,
):
    """Train a model using HuggingFace Trainer and return the trainer."""
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    # Remove checkpoint directories saved during training — the best model is
    # already loaded into trainer.model, so these are no longer needed.
    checkpoint_dirs = glob.glob(
        os.path.join(training_args.output_dir, "checkpoint-*")
    )
    for ckpt in checkpoint_dirs:
        if os.path.isdir(ckpt):
            shutil.rmtree(ckpt)
            logger.info("Removed checkpoint: %s", ckpt)

    return trainer
