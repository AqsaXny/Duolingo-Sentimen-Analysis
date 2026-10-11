"""
backend/inference.py
====================
Model loading and inference helpers for DuolingoLens.

Responsibilities:
  - Load the fine-tuned IndoBERT classification model ONCE at server
    startup via ``load_model()``.
  - Provide ``predict_single()`` for single-text inference.
  - Provide ``predict_batch()`` for bulk inference over a DataFrame.
  - All inference runs under ``torch.no_grad()`` on CPU.

How to run (standalone smoke-test — requires model files in ./model/):
    python inference.py
"""

import logging
import os
from typing import Any

import torch
from safetensors.torch import load_file as safetensors_load
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from preprocessing import preprocess, tokenize, MAX_TOKEN_LEN

# ── logging ────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────
# All paths are derived from the directory that contains *this* file so the
# application works regardless of the working directory it is launched from.
_HERE: str = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR: str = os.path.join(_HERE, "model")
_SAFETENSORS_PATH: str = os.path.join(_MODEL_DIR, "model.safetensors")

# ── label mapping — must match training order ──────────────────────────────
ID2LABEL: dict[int, str] = {0: "Negative", 1: "Positive"}
LABEL2ID: dict[str, int] = {"Negative": 0, "Positive": 1}

# ── module-level singletons (populated by load_model()) ───────────────────
_model: AutoModelForSequenceClassification | None = None
_tokenizer: AutoTokenizer | None = None
_model_loaded: bool = False
_load_error: str | None = None


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def load_model() -> bool:
    """Load tokenizer + model weights into module-level singletons.

    Must be called ONCE at Flask application startup (not per request).
    Returns ``True`` on success, ``False`` if the model directory or
    required files are missing (app will serve degraded but won't crash).

    Returns:
        ``True`` if the model was loaded successfully.
    """
    global _model, _tokenizer, _model_loaded, _load_error

    # Guard: check that all required artefacts exist before trying to load.
    required: list[str] = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing: list[str] = [
        f for f in required
        if not os.path.isfile(os.path.join(_MODEL_DIR, f))
    ]
    if missing:
        _load_error = (
            f"Missing model artefacts in {_MODEL_DIR}: {missing}. "
            "Place config.json, model.safetensors, tokenizer.json, "
            "and tokenizer_config.json in the /model folder."
        )
        logger.warning(_load_error)
        return False

    try:
        logger.info("Loading tokenizer from %s …", _MODEL_DIR)
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR, local_files_only=True)

        logger.info("Building model from config …")
        config = AutoConfig.from_pretrained(_MODEL_DIR, local_files_only=True)
        # Override label mappings in case config.json was saved without them.
        config.id2label = ID2LABEL
        config.label2id = LABEL2ID
        config.num_labels = 2

        logger.info("Instantiating model skeleton …")
        _model = AutoModelForSequenceClassification.from_config(config)

        logger.info("Loading safetensors weights …")
        state_dict = safetensors_load(_SAFETENSORS_PATH, device="cpu")
        _model.load_state_dict(state_dict, strict=True)
        _model.to(torch.device("cpu"))
        _model.eval()  # Disable dropout / batch-norm training behaviour.

        _model_loaded = True
        logger.info("Model loaded successfully (CPU, eval mode).")
        return True

    except Exception as exc:
        _load_error = str(exc)
        logger.exception("Failed to load model: %s", exc)
        return False


def is_loaded() -> bool:
    """Return whether the model was loaded successfully."""
    return _model_loaded


def get_load_error() -> str | None:
    """Return the error message from the last failed load attempt, or None."""
    return _load_error


def predict_single(text: str) -> dict[str, Any]:
    """Run single-text sentiment inference.

    Applies the full preprocessing pipeline, tokenises, and returns label +
    confidence + per-class probability scores.

    Args:
        text: Raw review string (may be messy / contain slang).

    Returns:
        Dict with keys:
            ``label``         — "Positive" or "Negative"
            ``confidence``    — float 0.0–1.0 (probability of predicted class)
            ``probabilities`` — {"Positive": float, "Negative": float}
            ``cleaned_text``  — string after preprocessing (useful for logging)

    Raises:
        RuntimeError: If the model has not been loaded via ``load_model()``.
        ValueError:   If *text* is empty after preprocessing.
    """
    if not _model_loaded or _model is None:
        raise RuntimeError(
            "Model is not loaded. Call inference.load_model() at app startup."
        )

    cleaned: str = preprocess(text)
    if not cleaned:
        raise ValueError("Text is empty after preprocessing.")

    # tokenize() internally calls preprocess() again — idempotent on clean input.
    inputs: dict[str, torch.Tensor] = tokenize(cleaned, max_length=MAX_TOKEN_LEN)
    inputs = {k: v.to(torch.device("cpu")) for k, v in inputs.items()}

    with torch.no_grad():
        logits: torch.Tensor = _model(**inputs).logits  # shape: (1, 2)
        probs: torch.Tensor = torch.softmax(logits, dim=-1).squeeze(0)  # shape: (2,)

    # Index 0 = Negative, index 1 = Positive (matches ID2LABEL).
    neg_prob: float = round(probs[0].item(), 6)
    pos_prob: float = round(probs[1].item(), 6)
    label_idx: int = int(torch.argmax(probs).item())
    label: str = ID2LABEL[label_idx]
    confidence: float = round(probs[label_idx].item(), 6)

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            "Positive": pos_prob,
            "Negative": neg_prob,
        },
        "cleaned_text": cleaned,
    }


def predict_batch(
    texts: list[str],
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Run inference over a list of texts with mini-batching.

    Processing in mini-batches keeps peak memory bounded even for large
    CSV uploads (up to the 5 000-row spec limit).

    Args:
        texts:      List of raw review strings.
        batch_size: Number of texts per forward pass (default: 32).

    Returns:
        List of dicts, each matching the ``predict_single()`` return format,
        plus a ``"raw_text"`` key containing the original unprocessed string.
        Texts that are empty after preprocessing receive
        ``label="Unknown", confidence=0.0``.
    """
    if not _model_loaded or _model is None:
        raise RuntimeError(
            "Model is not loaded. Call inference.load_model() at app startup."
        )

    if _tokenizer is None:
        # Should never happen if load_model() succeeded, but guard anyway.
        raise RuntimeError("Tokenizer not available.")

    results: list[dict[str, Any]] = []

    for batch_start in range(0, len(texts), batch_size):
        batch_raw: list[str] = texts[batch_start: batch_start + batch_size]

        # Preprocess all texts in the mini-batch first.
        cleaned_batch: list[str] = [preprocess(t) for t in batch_raw]

        # Build per-item results for empty-after-cleaning texts now so we
        # don't have to worry about them inside the model call.
        valid_indices: list[int] = []
        valid_cleaned: list[str] = []
        batch_results: list[dict[str, Any]] = [{}] * len(batch_raw)

        for i, (raw, clean) in enumerate(zip(batch_raw, cleaned_batch)):
            if not clean:
                batch_results[i] = {
                    "raw_text": raw,
                    "cleaned_text": clean,
                    "label": "Unknown",
                    "confidence": 0.0,
                    "probabilities": {"Positive": 0.0, "Negative": 0.0},
                }
            else:
                valid_indices.append(i)
                valid_cleaned.append(clean)

        if not valid_cleaned:
            results.extend(batch_results)
            continue

        # Tokenise the valid subset as a padded batch.
        encoding = _tokenizer(
            valid_cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKEN_LEN,
            padding=True,  # dynamic padding to longest in this mini-batch
        )
        encoding = {k: v.to(torch.device("cpu")) for k, v in encoding.items()}

        with torch.no_grad():
            logits: torch.Tensor = _model(**encoding).logits  # (N, 2)
            probs: torch.Tensor = torch.softmax(logits, dim=-1)  # (N, 2)

        # Map results back to original positions in batch_results.
        for rank, orig_idx in enumerate(valid_indices):
            p = probs[rank]
            neg_prob = round(p[0].item(), 6)
            pos_prob = round(p[1].item(), 6)
            label_idx = int(torch.argmax(p).item())
            label = ID2LABEL[label_idx]
            confidence = round(p[label_idx].item(), 6)

            batch_results[orig_idx] = {
                "raw_text": batch_raw[orig_idx],
                "cleaned_text": valid_cleaned[rank],
                "label": label,
                "confidence": confidence,
                "probabilities": {"Positive": pos_prob, "Negative": neg_prob},
            }

        results.extend(batch_results)

    return results


# ──────────────────────────────────────────────────────────────────────────
# Smoke-test (requires model artefacts in ./model/)
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not load_model():
        print(f"Model load failed: {get_load_error()}")
    else:
        _samples = [
            "Aplikasi ini bagus banget, belajar jadi menyenangkan!",
            "Gak bisa dibuka, error terus udah 3 hari parah banget",
            "Fitur barunya keren tapi sering lag waktu loading",
        ]
        for s in _samples:
            r = predict_single(s)
            print(f"  {r['label']} ({r['confidence']:.2%}) | {s[:50]}")
