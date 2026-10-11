"""
app.py — DuolingoLens Flask Application
========================================
Endpoints:
  GET  /                    → Serve dashboard SPA
  POST /api/predict         → Single-text sentiment prediction
  POST /api/predict-csv     → Batch CSV sentiment prediction
  GET  /api/health          → Model + server health check
"""

import os
import io
import json
import logging
from collections import Counter
from typing import Any

import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request
from safetensors.torch import load_file as safetensors_load
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from utils.preprocessor import clean_text

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("duolingo_lens")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")
SAFETENSORS_PATH = os.path.join(MODEL_DIR, "model.safetensors")
MAX_LEN = 128
DEVICE = torch.device("cpu")

# Label mapping (must match training label order)
ID2LABEL = {0: "Negative", 1: "Positive"}
LABEL2ID = {"Negative": 0, "Positive": 1}

# ---------------------------------------------------------------------------
# Lazy-loaded globals
# ---------------------------------------------------------------------------
_tokenizer = None
_model = None
_model_ready = False
_model_error: str | None = None


def _load_model() -> None:
    """Load tokenizer + model from model_artifacts/. Called once on startup."""
    global _tokenizer, _model, _model_ready, _model_error

    if not os.path.isdir(MODEL_DIR):
        _model_error = f"model_artifacts/ directory not found at {MODEL_DIR}"
        logger.warning(_model_error)
        return

    required = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    missing = [f for f in required if not os.path.isfile(os.path.join(MODEL_DIR, f))]
    if missing:
        _model_error = f"Missing model files: {missing}. Place them in model_artifacts/."
        logger.warning(_model_error)
        return

    try:
        logger.info("Loading tokenizer from %s …", MODEL_DIR)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

        logger.info("Loading model config …")
        config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
        config.id2label = ID2LABEL
        config.label2id = LABEL2ID
        config.num_labels = 2

        logger.info("Instantiating model skeleton …")
        _model = AutoModelForSequenceClassification.from_config(config)

        logger.info("Loading weights from model.safetensors …")
        state_dict = safetensors_load(SAFETENSORS_PATH, device="cpu")
        _model.load_state_dict(state_dict, strict=True)
        _model.to(DEVICE)
        _model.eval()

        _model_ready = True
        logger.info("Model loaded successfully (CPU, eval mode).")
    except Exception as exc:
        _model_error = str(exc)
        logger.exception("Failed to load model: %s", exc)


# Load at startup
_load_model()


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def _predict_single(raw_text: str) -> dict[str, Any]:
    """Run inference on a single text. Returns label + confidence scores."""
    cleaned = clean_text(raw_text)
    if not cleaned:
        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "label": "Unknown",
            "confidence": 0.0,
            "scores": {"Negative": 0.0, "Positive": 0.0},
        }

    inputs = _tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    # probs[0] = Negative, probs[1] = Positive
    label_idx = int(torch.argmax(torch.tensor(probs)).item())
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "label": ID2LABEL[label_idx],
        "confidence": round(probs[label_idx], 4),
        "scores": {
            "Negative": round(probs[0], 4),
            "Positive": round(probs[1], 4),
        },
    }


def _build_analytics(results: list[dict]) -> dict:
    """Derive aggregate analytics from a list of prediction results."""
    labels = [r["label"] for r in results]
    counter = Counter(labels)
    total = len(labels)
    positive = counter.get("Positive", 0)
    negative = counter.get("Negative", 0)

    # Confidence distribution buckets: Low (<0.6), Medium (0.6–0.85), High (>0.85)
    conf_buckets = {"Low": 0, "Medium": 0, "High": 0}
    for r in results:
        c = r["confidence"]
        if c < 0.60:
            conf_buckets["Low"] += 1
        elif c < 0.85:
            conf_buckets["Medium"] += 1
        else:
            conf_buckets["High"] += 1

    # Word frequency for word cloud (top-60 tokens from cleaned text)
    all_words: list[str] = []
    stopwords = {
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
        "ada", "tidak", "sudah", "juga", "bisa", "lebih", "akan", "saja",
        "pada", "dalam", "atau", "jadi", "karena", "kalau", "tapi", "lagi",
        "nya", "aja", "si", "ya", "sih", "nih", "deh", "lah", "kah",
    }
    for r in results:
        tokens = r.get("cleaned_text", "").split()
        all_words.extend([t for t in tokens if len(t) > 2 and t not in stopwords])

    word_freq = Counter(all_words).most_common(60)
    word_cloud_data = [{"text": w, "value": v} for w, v in word_freq]

    # Sentiment by "rating bucket" — only available if results contain a 'rating' field
    rating_sentiment: dict[str, dict] = {}
    for r in results:
        rating = r.get("rating")
        if rating is not None:
            key = str(rating)
            if key not in rating_sentiment:
                rating_sentiment[key] = {"Positive": 0, "Negative": 0}
            rating_sentiment[key][r["label"]] += 1

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_pct": round(positive / total * 100, 2) if total else 0,
        "negative_pct": round(negative / total * 100, 2) if total else 0,
        "confidence_distribution": conf_buckets,
        "word_cloud": word_cloud_data,
        "rating_sentiment": rating_sentiment,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok" if _model_ready else "degraded",
        "model_ready": _model_ready,
        "model_error": _model_error,
        "device": str(DEVICE),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Single-text prediction endpoint."""
    if not _model_ready:
        return jsonify({"error": f"Model not ready. {_model_error}"}), 503

    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "Field 'text' is required and must not be empty."}), 400

    result = _predict_single(text)
    analytics = _build_analytics([result])
    return jsonify({"result": result, "analytics": analytics})


@app.route("/api/predict-csv", methods=["POST"])
def predict_csv():
    """Batch CSV prediction endpoint."""
    if not _model_ready:
        return jsonify({"error": f"Model not ready. {_model_error}"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file part in request. Use field name 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only .csv files are accepted."}), 400

    try:
        content = file.read().decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(content))
    except Exception as exc:
        return jsonify({"error": f"Failed to parse CSV: {exc}"}), 400

    # Detect text column (flexible column name matching)
    text_col = None
    rating_col = None
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ("content", "review", "text", "ulasan", "komentar", "teks"):
            text_col = col
        if col_lower in ("score", "rating", "bintang", "stars"):
            rating_col = col

    if text_col is None:
        return jsonify({
            "error": (
                f"Cannot find a text column. Detected columns: {list(df.columns)}. "
                "Rename your review column to one of: content, review, text, ulasan, komentar, teks."
            )
        }), 400

    # Drop rows with missing text
    df = df.dropna(subset=[text_col])
    if len(df) == 0:
        return jsonify({"error": "CSV has no valid text rows after dropping nulls."}), 400

    # Cap at 1 000 rows to keep response time acceptable on CPU
    MAX_ROWS = 1000
    truncated = len(df) > MAX_ROWS
    df = df.head(MAX_ROWS)

    results = []
    for _, row in df.iterrows():
        pred = _predict_single(str(row[text_col]))
        if rating_col and pd.notna(row.get(rating_col)):
            try:
                pred["rating"] = int(row[rating_col])
            except (ValueError, TypeError):
                pred["rating"] = None
        results.append(pred)

    analytics = _build_analytics(results)
    return jsonify({
        "results": results,
        "analytics": analytics,
        "truncated": truncated,
        "rows_processed": len(results),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
