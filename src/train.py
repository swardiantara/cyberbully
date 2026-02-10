import logging

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

logger = logging.getLogger("cyberbully")


class CyberbullyDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapping tokenized encodings and integer labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def tokenize_dataset(texts, tokenizer, max_length: int) -> dict:
    """Tokenize a list of texts using the given tokenizer."""
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
    )
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
    }


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
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="no",
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

    return trainer
