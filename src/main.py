import argparse
import json
import os
import sys

import torch

from utils import set_seed, get_device, get_output_dir, setup_logging
from data import prepare_data
from model import load_model_and_tokenizer
from contrastive import check_model_exists, parse_custom_model_path, contrastive_finetune
from train import (
    CyberbullyDataset,
    get_training_args,
    train_model,
)
from evaluate import evaluate_model
from attribution import compute_attributions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cyberbullying Detection Pipeline — "
        "Fine-tune transformer models with evaluation and attribution."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model checkpoint name "
        "(e.g., bert-base-uncased, distilbert-base-uncased, gpt2, "
        "xlnet-base-cased, roberta-base)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ieee", "kaggle", "tweeteval"],
        help="Dataset identifier",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=False,
        help="Apply text preprocessing/cleaning",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=False,
        help="Apply RandomOverSampler to balance training data",
    )
    parser.add_argument(
        "--compute_attribution",
        action="store_true",
        default=False,
        help="Compute words attribution using Integrated Gradients",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max token sequence length (default: 128)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="grid-search",
        help="Base output directory (default: grid-search)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Datasets",
        help="Directory containing dataset CSV files (default: Datasets)",
    )
    parser.add_argument(
        "--sbert",
        action="store_true",
        default=False,
        help="Use SentenceTransformer-based pipeline (SBERTClassifier). "
        "For custom models (org/dataset-base), triggers contrastive "
        "fine-tuning if the model does not exist on HuggingFace.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing experiment results",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Setup ---
    set_seed(args.seed)
    device = get_device()
    output_dir = get_output_dir(
        args.output_dir, args.model, args.dataset,
        args.preprocess, args.augment, args.seed,
    )
    logger = setup_logging(output_dir)

    if str(args.model).startswith("all"):
        args.sbert = True
        logger.info(
            "Model '%s' detected as SBERT-based. Using SBERTClassifier pipeline.",
            args.model,
        )
    logger.info("=" * 60)
    logger.info("Cyberbullying Detection Pipeline")
    logger.info("=" * 60)
    logger.info("Configuration:")
    for k, v in vars(args).items():
        logger.info("  %-15s: %s", k, v)
    logger.info("  %-15s: %s", "device", device)
    logger.info("  %-15s: %s", "output_dir", output_dir)
    logger.info("=" * 60)

    # Check if the scenario has been executed successfully before
    if os.path.exists(os.path.join(output_dir, "metrics.json")) and not args.overwrite:
        logger.info(
            "This experiment has been completed before. Skipped!"
        )
        sys.exit(0)

    # --- Data preparation ---
    logger.info("Step 1: Preparing data...")
    train_df, val_df, test_df, label2id, id2label = prepare_data(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        preprocess=args.preprocess,
        augment=args.augment,
    )

    num_labels = len(label2id)
    logger.info("Number of classes: %d", num_labels)

    # Save run configuration with dataset statistics
    def class_distribution(df):
        counts = df["label"].value_counts().sort_index()
        return {id2label[int(k)]: int(v) for k, v in counts.items()}

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        config = vars(args).copy()
        config["device"] = device
        config["output_dir_resolved"] = output_dir
        config["dataset_statistics"] = {
            "num_classes": num_labels,
            "label_mapping": label2id,
            "train": {
                "num_samples": len(train_df),
                "class_distribution": class_distribution(train_df),
            },
            "val": {
                "num_samples": len(val_df),
                "class_distribution": class_distribution(val_df),
            },
            "test": {
                "num_samples": len(test_df),
                "class_distribution": class_distribution(test_df),
            },
        }
        json.dump(config, f, indent=2)

    # --- Contrastive fine-tuning (SBERT custom models only) ---
    if args.sbert and str(args.model).startswith("swardiantara"):
        if not check_model_exists(args.model):
            _, base_model = parse_custom_model_path(args.model)
            logger.info(
                "Step 1b: Contrastive fine-tuning '%s' -> '%s'",
                base_model, args.model,
            )
            contrastive_finetune(base_model, train_df, args.model, args.seed)

    # --- Model loading ---
    logger.info("Step 2: Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        args.model, num_labels, id2label, label2id, sbert=args.sbert,
    )

    # --- Dataset creation (tokenization happens inside the Dataset) ---
    logger.info("Step 3: Creating datasets...")
    train_dataset = CyberbullyDataset(
        train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, args.max_length,
    )
    val_dataset = CyberbullyDataset(
        val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, args.max_length,
    )
    test_dataset = CyberbullyDataset(
        test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, args.max_length,
    )

    logger.info(
        "Dataset sizes — train: %d, val: %d, test: %d",
        len(train_dataset), len(val_dataset), len(test_dataset),
    )

    # --- Training ---
    logger.info("Step 4: Training...")
    training_args = get_training_args(
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    trainer = train_model(model, tokenizer, train_dataset, val_dataset, training_args)

    # Use the best model from training
    model = trainer.model

    # --- Evaluation ---
    logger.info("Step 5: Evaluating on test set...")
    all_preds, all_labels, all_probs = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        id2label=id2label,
        device=device,
        output_dir=output_dir,
    )

    # --- Integrated Gradients Attribution ---
    if args.compute_attribution:
        logger.info("Step 6: Computing Integrated Gradients attributions...")
        compute_attributions(
            model=model,
            tokenizer=tokenizer,
            test_texts=test_df["text"].tolist(),
            test_labels=test_df["label"].tolist(),
            id2label=id2label,
            device=device,
            model_name=args.model,
            output_dir=output_dir,
            n_steps=50,
        )

    logger.info("=" * 60)
    logger.info("Pipeline complete. Results saved to: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
