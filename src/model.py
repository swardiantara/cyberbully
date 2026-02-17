import logging

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger("cyberbully")


class SBERTClassifier(nn.Module):
    """Classification model using a SentenceTransformer as the backbone.

    Extracts the underlying transformer from a SentenceTransformer model,
    applies mean pooling, and feeds the pooled embedding through a
    Linear-ReLU-Dropout-Linear classification head.
    """

    def __init__(self, model_name: str, num_labels: int, id2label: dict, label2id: dict):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer(model_name)
        self.transformer = st_model[0].auto_model
        embedding_dim = st_model.get_sentence_embedding_dimension()

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

        # Expose config for HF Trainer compatibility
        self.config = self.transformer.config
        self.config.num_labels = num_labels
        self.config.id2label = id2label
        self.config.label2id = label2id

        logger.info(
            "SBERTClassifier: embedding_dim=%d, hidden=256, num_labels=%d",
            embedding_dim, num_labels,
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        # Mean pooling over token embeddings, masked by attention_mask
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = (
            torch.sum(token_embeddings * mask_expanded, dim=1)
            / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        )

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def get_embedding_layer(self):
        """Return the word embedding layer for Captum attribution."""
        return self.transformer.embeddings.word_embeddings


def _resolve_sbert_path(model_name: str) -> str:
    """Resolve an SBERT model name to its full HuggingFace Hub path."""
    if "/" not in model_name:
        return f"sentence-transformers/{model_name}"
    return model_name


def load_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    id2label: dict,
    label2id: dict,
    sbert: bool = False,
):
    """Load a HuggingFace model and tokenizer with proper configuration."""
    logger.info("Loading model '%s' with %d labels (sbert=%s)", model_name, num_labels, sbert)

    if sbert:
        hf_path = _resolve_sbert_path(model_name)
        model = SBERTClassifier(hf_path, num_labels, id2label, label2id)
        tokenizer = AutoTokenizer.from_pretrained(hf_path)
    else:
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
    # SBERTClassifier provides its own method
    if hasattr(model, "get_embedding_layer"):
        return model.get_embedding_layer()

    model_type = getattr(model.config, "model_type", "").lower()

    if model_type == "distilbert":
        return model.distilbert.embeddings.word_embeddings
    elif model_type in ("roberta", "bertweet"):
        return model.roberta.embeddings.word_embeddings
    elif model_type == "bert":
        return model.bert.embeddings.word_embeddings
    elif model_type == "mpnet":
        return model.mpnet.embeddings.word_embeddings
    elif model_type == "xlnet":
        return model.transformer.word_embedding
    elif model_type == "gpt2":
        return model.transformer.wte
    else:
        raise ValueError(
            f"Unknown model architecture: model_type='{model_type}' "
            f"(from '{model_name}'). "
            "Cannot determine embedding layer. "
            "Supported model_types: bert, distilbert, roberta, bertweet, "
            "mpnet, xlnet, gpt2."
        )
