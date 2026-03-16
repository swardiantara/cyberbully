import json
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch
from codecarbon import EmissionsTracker
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger("cyberbully")


def evaluate_model(
    model,
    test_dataset,
    id2label: dict,
    device: str,
    output_dir: str,
):
    """Evaluate the model on the test set and export metrics to JSON.

    Returns:
        all_preds: list of predicted label indices
        all_labels: list of true label indices
        all_probs: numpy array of softmax probabilities (n_samples, n_classes)
    """
    model.eval()
    model.to(device)

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    all_preds = []
    all_labels = []
    all_logits = []

    # --- Energy tracking ---
    tracker = EmissionsTracker(
        project_name="cyberbully_eval",
        log_level="warning",
        save_to_file=False,
    )

    tracker.start()
    start_time = time.time()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_logits.append(logits)

    inference_time = time.time() - start_time
    emissions = tracker.stop()

    # --- Compute probabilities ---
    all_logits = torch.cat(all_logits, dim=0)
    all_probs = torch.softmax(all_logits, dim=-1).numpy()

    # --- Metrics ---
    label_names = [id2label[i] for i in sorted(id2label.keys())]
    num_classes = len(label_names)
    label_indices = list(range(num_classes))

    # Per-class metrics via classification_report
    report = classification_report(
        all_labels,
        all_preds,
        labels=label_indices,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    # Micro average (computed explicitly)
    micro_precision = precision_score(
        all_labels, all_preds, average="micro", zero_division=0
    )
    micro_recall = recall_score(
        all_labels, all_preds, average="micro", zero_division=0
    )
    micro_f1 = f1_score(
        all_labels, all_preds, average="micro", zero_division=0
    )

    # Build metrics dict
    metrics = {
        "per_class": {},
        "weighted_avg": report.get("weighted avg", {}),
        "micro_avg": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1-score": micro_f1,
            "support": len(all_labels),
        },
        "macro_avg": report.get("macro avg", {}),
        "accuracy": accuracy_score(all_labels, all_preds),
        "inference_time_seconds": inference_time,
        "energy_kwh": emissions if emissions is not None else 0.0,
        "num_test_samples": len(all_labels),
    }

    for name in label_names:
        if name in report:
            metrics["per_class"][name] = report[name]

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds, labels=label_indices)
    cm_dict = {
        "labels": label_names,
        "matrix": cm.tolist(),
    }

    # --- Save to JSON ---
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Metrics saved to %s", metrics_path)

    cm_path = os.path.join(output_dir, "confusion_matrix.json")
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(cm_dict, f, indent=2, ensure_ascii=False)
    logger.info("Confusion matrix saved to %s", cm_path)

    # --- Confusion matrix PDF figure ---
    fig, ax = plt.subplots(figsize=(max(7, num_classes + 2), max(6, num_classes + 1)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15)
    plt.tight_layout()
    cm_pdf_path = os.path.join(output_dir, "confusion_matrix.pdf")
    fig.savefig(cm_pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix figure saved to %s", cm_pdf_path)

    # --- Print summary ---
    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Weighted F1: %.4f", metrics["weighted_avg"].get("f1-score", 0))
    logger.info("Micro F1: %.4f", micro_f1)
    logger.info("Inference time: %.2f seconds", inference_time)
    logger.info("Energy consumption: %.6f kWh", metrics["energy_kwh"])

    return all_preds, all_labels, all_probs


def plot_projection_tsne(
    model,
    dataset,
    id2label: dict,
    device: str,
    output_dir: str,
    texts: list = None,
    batch_size: int = 64,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
):
    """Extract projection-head embeddings, reduce to 2D with t-SNE, and save:

    - projection_tsne.pdf  — static figure (matplotlib/seaborn)
    - projection_tsne.html — interactive figure (Plotly); each point shows the
      original input text, true label, predicted label, and correctness on
      hover, enabling error analysis of misclassified samples.

    Only meaningful when the model is a SupConClassifier whose forward()
    returns proj_features.
    """
    model.eval()
    model.to(device)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    all_proj = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_proj.append(outputs.proj_features.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy().tolist()
            all_preds.extend(preds)

    proj_matrix = np.concatenate(all_proj, axis=0)   # (N, proj_dim)
    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)

    logger.info(
        "Running t-SNE on %d samples with proj_dim=%d, perplexity=%.1f...",
        len(labels_arr), proj_matrix.shape[1], perplexity,
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
    )
    embeddings_2d = tsne.fit_transform(proj_matrix)   # (N, 2)

    label_names = [id2label[i] for i in sorted(id2label.keys())]
    num_classes = len(label_names)

    # --- Static PDF (matplotlib) ---
    palette = sns.color_palette("tab10", n_colors=num_classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    for class_idx, class_name in enumerate(label_names):
        mask = labels_arr == class_idx
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=class_name,
            color=palette[class_idx],
            s=12,
            alpha=0.7,
            linewidths=0,
        )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.legend(title="Class", loc="best", fontsize=9)
    plt.tight_layout()
    pdf_path = os.path.join(output_dir, "projection_tsne.pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("t-SNE static plot saved to %s", pdf_path)

    # --- Interactive HTML (Plotly) ---
    true_label_names = [id2label[i] for i in labels_arr]
    pred_label_names = [id2label[i] for i in preds_arr]
    correct = ["correct" if p == l else "wrong" for p, l in zip(preds_arr, labels_arr)]

    # Truncate long texts for the hover tooltip
    MAX_HOVER_CHARS = 300
    if texts is not None:
        hover_texts = [
            t[:MAX_HOVER_CHARS] + "…" if len(t) > MAX_HOVER_CHARS else t
            for t in texts
        ]
    else:
        hover_texts = ["(text not provided)"] * len(labels_arr)

    hover_df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "true_label": true_label_names,
        "predicted": pred_label_names,
        "result": correct,
        "text": hover_texts,
    })

    plotly_fig = px.scatter(
        hover_df,
        x="x",
        y="y",
        color="true_label",
        symbol="result",
        symbol_map={"correct": "circle", "wrong": "x"},
        hover_data={"x": False, "y": False, "predicted": True, "result": True, "text": True},
        title="Projection Embeddings — t-SNE",
        labels={
            "x": "t-SNE Dimension 1",
            "y": "t-SNE Dimension 2",
            "true_label": "True Label",
        },
        color_discrete_sequence=px.colors.qualitative.T10,
    )
    plotly_fig.update_traces(marker=dict(size=6, opacity=0.8))
    plotly_fig.update_layout(
        legend_title_text="True Label",
        hoverlabel=dict(font_size=12, namelength=-1),
    )

    html_path = os.path.join(output_dir, "projection_tsne.html")
    plotly_fig.write_html(html_path)
    logger.info("t-SNE interactive plot saved to %s", html_path)
