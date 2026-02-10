import json
import logging
import os

import torch
import torch.nn.functional as F
from captum.attr import LayerIntegratedGradients

from model import ModelWrapper, get_embedding_layer

logger = logging.getLogger("cyberbully")


def compute_attributions(
    model,
    tokenizer,
    test_texts: list,
    test_labels: list,
    id2label: dict,
    device: str,
    model_name: str,
    output_dir: str,
    n_steps: int = 50,
):
    """Compute Integrated Gradients attributions for each test sample.

    Attributions are computed w.r.t. the TRUE label (ground truth).
    Raw token-level attribution scores are stored (no word-level aggregation).

    Args:
        model: trained HuggingFace model
        tokenizer: corresponding tokenizer
        test_texts: list of raw text strings
        test_labels: list of integer label indices (ground truth)
        id2label: mapping from integer to label name
        device: "cuda" or "cpu"
        model_name: model checkpoint name (for embedding layer lookup)
        output_dir: directory to save attributions.json
        n_steps: number of steps for Integrated Gradients (default 50)
    """
    model.eval()
    model.to(device)

    # Wrap model for Captum
    wrapper = ModelWrapper(model)
    wrapper.eval()
    wrapper.to(device)

    # Get the embedding layer for LayerIntegratedGradients
    embedding_layer = get_embedding_layer(model, model_name)
    lig = LayerIntegratedGradients(wrapper, embedding_layer)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0

    attributions_list = []
    total = len(test_texts)

    logger.info(
        "Computing Integrated Gradients for %d test samples (n_steps=%d)...",
        total,
        n_steps,
    )

    for idx in range(total):
        text = test_texts[idx]
        true_label = int(test_labels[idx])

        # Tokenize single sample
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Create baseline (all pad tokens, same shape)
        baseline_ids = torch.full_like(input_ids, pad_token_id)

        # Compute attributions w.r.t. the TRUE label
        attrs = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            target=true_label,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
            return_convergence_delta=False,
        )

        # Sum across embedding dimensions to get per-token attribution score
        # attrs shape: (1, seq_len, embedding_dim) -> (seq_len,)
        token_attributions = attrs.sum(dim=-1).squeeze(0).detach().cpu().tolist()

        # Get model prediction and confidence
        with torch.no_grad():
            logits = wrapper(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
            predicted_label = int(torch.argmax(logits, dim=-1).item())

        # Convert token IDs to token strings
        token_ids = input_ids.squeeze(0).cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        # Build confidence dict with label names
        confidence = {
            id2label[i]: round(probs[i], 6) for i in range(len(probs))
        }

        record = {
            "index": idx,
            "text": text,
            "tokens": tokens,
            "token_ids": token_ids,
            "attributions": [round(a, 6) for a in token_attributions],
            "true_label": true_label,
            "true_label_name": id2label[true_label],
            "predicted_label": predicted_label,
            "predicted_label_name": id2label[predicted_label],
            "confidence": confidence,
        }
        attributions_list.append(record)

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            logger.info(
                "Attribution progress: %d / %d (%.1f%%)",
                idx + 1,
                total,
                100.0 * (idx + 1) / total,
            )

    # Save to JSON
    attr_path = os.path.join(output_dir, "attributions.json")
    with open(attr_path, "w", encoding="utf-8") as f:
        json.dump(attributions_list, f, indent=2, ensure_ascii=False)

    logger.info("Attributions saved to %s (%d samples)", attr_path, len(attributions_list))
