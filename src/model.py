import logging

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("cyberbully")


def load_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    id2label: dict,
    label2id: dict,
):
    """Load a HuggingFace model and tokenizer with proper configuration."""
    logger.info("Loading model '%s' with %d labels", model_name, num_labels)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # GPT-2 has no pad token by default — use eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "left"
        logger.info(
            "Set pad_token to eos_token ('%s') for model '%s'",
            tokenizer.eos_token,
            model_name,
        )

    return model, tokenizer


class ModelWrapper(nn.Module):
    """Wrapper around a HuggingFace model for Captum compatibility.

    Captum requires a forward function that takes tensor inputs and returns
    logits directly, without the HuggingFace output object.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


def get_embedding_layer(model, model_name: str) -> nn.Module:
    """Return the word embedding layer for a given model architecture.

    This is the layer that Captum's LayerIntegratedGradients targets.
    Uses model.config.model_type to reliably detect the architecture,
    regardless of the model's HuggingFace hub name.
    """
    model_type = getattr(model.config, "model_type", "").lower()

    if model_type == "distilbert":
        return model.distilbert.embeddings.word_embeddings
    elif model_type in ("roberta", "bertweet"):
        return model.roberta.embeddings.word_embeddings
    elif model_type == "bert":
        return model.bert.embeddings.word_embeddings
    elif model_type == "xlnet":
        return model.transformer.word_embedding
    elif model_type == "gpt2":
        return model.transformer.wte
    else:
        raise ValueError(
            f"Unknown model architecture: model_type='{model_type}' "
            f"(from '{model_name}'). "
            "Cannot determine embedding layer. "
            "Supported model_types: bert, distilbert, roberta, bertweet, xlnet, gpt2."
        )
