import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger("cyberbully")


# ---------------------------------------------------------------------------
# Custom output for SupCon models
# ---------------------------------------------------------------------------

@dataclass
class SupConClassifierOutput(SequenceClassifierOutput):
    """Extends SequenceClassifierOutput to carry L2-normalized projection features
    used by SupConLoss alongside the standard CE logits."""
    proj_features: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# SupCon model: transformer + projection head + classifier head
# ---------------------------------------------------------------------------

class SupConClassifier(nn.Module):
    """Transformer backbone with a projection head (SupCon) and a classification
    head (CE), both operating on the same mean-pooled representation.

    The projection head output is L2-normalized in forward(), ready for
    SupConLoss. The classification head produces raw logits for CrossEntropy.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        id2label: dict,
        label2id: dict,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )

        # Projection head: 2-layer MLP as in Khosla et al. (2020)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_dim),
        )

        # Expose config for HF Trainer compatibility
        self.config = self.transformer.config
        self.config.num_labels = num_labels
        self.config.id2label = id2label
        self.config.label2id = label2id

        logger.info(
            "SupConClassifier: hidden_size=%d, proj_dim=%d, num_labels=%d",
            hidden_size, proj_dim, num_labels,
        )

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return (
            torch.sum(last_hidden_state * mask_expanded, dim=1)
            / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)

        logits = self.classifier(pooled)
        proj_features = F.normalize(self.projector(pooled), dim=-1)

        ce_loss = None
        if labels is not None:
            ce_loss = nn.CrossEntropyLoss()(logits, labels)

        return SupConClassifierOutput(loss=ce_loss, logits=logits, proj_features=proj_features)

    def get_embedding_layer(self) -> nn.Module:
        """Return the word embedding layer for Captum attribution."""
        model_type = getattr(self.transformer.config, "model_type", "").lower()
        if model_type == "xlnet":
            return self.transformer.word_embedding
        elif model_type == "gpt2":
            return self.transformer.wte
        else:
            # Covers bert, roberta, distilbert, mobilebert, mpnet, etc.
            return self.transformer.embeddings.word_embeddings


# ---------------------------------------------------------------------------
# SBERT-based classifier
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

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
    supcon: bool = False,
    proj_dim: int = 128,
):
    """Load a HuggingFace model and tokenizer with proper configuration.

    When supcon=True, loads SupConClassifier (AutoModel + projection head +
    classifier head) to enable the combined CE + SupCon training objective.
    When sbert=True, loads SBERTClassifier (SentenceTransformer backbone).
    When neither flag is set, loads AutoModelForSequenceClassification.
    """
    logger.info(
        "Loading model '%s' with %d labels (sbert=%s, supcon=%s)",
        model_name, num_labels, sbert, supcon,
    )

    if sbert:
        hf_path = _resolve_sbert_path(model_name)
        model = SBERTClassifier(hf_path, num_labels, id2label, label2id)
        tokenizer = AutoTokenizer.from_pretrained(hf_path)
    elif supcon:
        model = SupConClassifier(model_name, num_labels, id2label, label2id, proj_dim=proj_dim)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
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


# ---------------------------------------------------------------------------
# Captum helpers
# ---------------------------------------------------------------------------

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
    # SBERTClassifier and SupConClassifier provide their own method
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
