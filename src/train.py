import glob
import logging
import os
import shutil

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

from loss import SupConLoss

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
    fp16: bool = True,
    grad_accum_steps: int = 1,
) -> TrainingArguments:
    """Create HuggingFace TrainingArguments with sensible defaults.

    grad_accum_steps accumulates gradients over N mini-batches before an
    optimizer step, simulating an effective batch size of
    batch_size * grad_accum_steps without additional GPU memory.
    """
    use_fp16 = fp16 and torch.cuda.is_available()
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        weight_decay=1e-8,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        save_strategy="epoch",
        metric_for_best_model="f1_weighted",
        logging_steps=50,
        seed=seed,
        fp16=use_fp16,
        report_to="none",
    )


class SupConTrainer(Trainer):
    """HuggingFace Trainer with a combined CE + SupCon auxiliary loss.

    Requires the model's forward() to return a SupConClassifierOutput that
    carries L2-normalized proj_features alongside the standard CE loss/logits.
    The total loss is: L_total = L_CE + supcon_weight * L_SupCon
    """

    def __init__(self, *args, supcon_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self._supcon_criterion = SupConLoss()
        self._supcon_weight = supcon_weight

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        labels = inputs.get("labels")
        outputs = model(**inputs)

        ce_loss = outputs.loss

        # proj_features: [bsz, proj_dim], already L2-normalized by SupConClassifier
        # SupConLoss expects [bsz, n_views, dim] — add the single-view dimension
        proj_features = outputs.proj_features.unsqueeze(1)  # [bsz, 1, proj_dim]
        supcon_loss = self._supcon_criterion(proj_features, labels)

        total_loss = ce_loss + self._supcon_weight * supcon_loss

        return (total_loss, outputs) if return_outputs else total_loss


def train_model(
    model,
    tokenizer,
    train_dataset: CyberbullyDataset,
    val_dataset: CyberbullyDataset,
    training_args: TrainingArguments,
    supcon_weight: float = 0.0,
):
    """Train a model using HuggingFace Trainer and return the trainer.

    When supcon_weight > 0, uses SupConTrainer which adds a SupCon auxiliary
    loss on the model's projection head output alongside the standard CE loss.
    """
    if supcon_weight > 0:
        trainer = SupConTrainer(
            supcon_weight=supcon_weight,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_fn,
            tokenizer=tokenizer,
        )
        logger.info("Using SupConTrainer with supcon_weight=%.3f", supcon_weight)
    else:
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
