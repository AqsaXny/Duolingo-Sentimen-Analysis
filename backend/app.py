"""
backend/app.py
==============
DuolingoLens — Flask REST API + static front-end server.

Endpoints:
  POST /api/predict        — single-text sentiment prediction
  POST /api/predict-batch  — CSV batch prediction with trend + keywords
  GET  /api/health         — server / model liveness probe
  GET  /                   — serve frontend/index.html

How to run:
    cd backend
    pip install flask torch transformers safetensors pandas
    python app.py
    # → http://localhost:5000
"""

import io
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import pandas as pd
import torch
from flask import Flask, jsonify, render_template_string, request, send_from_directory

import inference
from preprocessing import preprocess

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("duolingo_lens")

# ── paths ──────────────────────────────────────────────────────────────────
# Resolve all paths relative to this file so the app is location-agnostic.
_HERE: str = os.path.dirname(os.path.abspath(__file__))
_FRONTEND_DIR: str = os.path.join(_HERE, "..", "frontend")

# ── Flask application ──────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=os.path.join(_FRONTEND_DIR),
    static_url_path="/static",
)

# Limit upload size to 32 MB — large enough for a 5 000-row CSV.
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

# ── constants ──────────────────────────────────────────────────────────────
# Maximum rows processed per CSV upload (spec requirement: 5 000).
CSV_MAX_ROWS: int = 5_000

# Column names accepted as the review text column (case-insensitive check).
REVIEW_COLUMN_CANDIDATES: list[str] = [
    "review", "ulasan", "content", "text", "komentar", "teks",
]

# Column names accepted as the date column.
DATE_COLUMN_CANDIDATES: list[str] = ["date", "tanggal", "at", "datetime"]

# Column names accepted as the numeric star-rating column.
RATING_COLUMN_CANDIDATES: list[str] = ["score", "rating", "bintang", "stars"]

# Indonesian stop-words excluded from keyword extraction.
_STOPWORDS: frozenset[str] = frozenset({
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
    "ada", "tidak", "sudah", "juga", "bisa", "lebih", "akan", "saja",
    "pada", "dalam", "atau", "jadi", "karena", "kalau", "tapi", "lagi",
    "nya", "aja", "si", "ya", "sih", "nih", "deh", "lah", "kah", "mau",
    "banget", "sangat", "sekali", "sering", "banyak", "sedikit",
})

# ── load model at startup ─────────────────────────────────────────────────
# Loading happens once here — never inside a request handler.
logger.info("Loading model at startup …")
inference.load_model()


# ──────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────

def _json_error(message: str, status: int) -> Any:
    """Return a JSON error response with the given HTTP *status* code.

    Args:
        message: Human-readable error description.
        status:  HTTP status code (400, 500, etc.).

    Returns:
        Flask Response object with ``Content-Type: application/json``.
    """
    return jsonify({"error": message}), status


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first column in *df* whose name matches any of *candidates*.

    Comparison is case-insensitive and strips leading/trailing whitespace.

    Args:
        df:         Input DataFrame.
        candidates: Ordered list of acceptable column names.

    Returns:
        The matching column name as it appears in *df*, or ``None``.
    """
    normalised: dict[str, str] = {c.strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalised:
            return normalised[candidate.lower()]
    return None


def _extract_keywords(
    texts: list[str],
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Extract the top-N most frequent tokens from *texts*.

    Tokens shorter than 3 characters and tokens in ``_STOPWORDS`` are
    excluded.  Returns a list of ``{word, weight}`` dicts where ``weight``
    is the raw frequency count (not normalised), sorted descending.

    Args:
        texts: List of cleaned review strings.
        top_n: Maximum number of keywords to return.

    Returns:
        List of ``{"word": str, "weight": int}`` dicts.
    """
    counter: Counter = Counter()
    for text in texts:
        tokens = text.split()
        for token in tokens:
            if len(token) >= 3 and token not in _STOPWORDS:
                counter[token] += 1
    return [{"word": w, "weight": v} for w, v in counter.most_common(top_n)]


def _build_trend(
    results: list[dict[str, Any]],
    date_series: pd.Series | None,
) -> list[dict[str, Any]]:
    """Aggregate prediction results into a time-series trend.

    If *date_series* is provided and parseable, results are grouped by
    calendar date (``YYYY-MM-DD``).  Otherwise each row is treated as an
    ordered observation and grouped into buckets of 10 rows to keep the
    trend chart readable.

    Args:
        results:     List of prediction result dicts (must include ``label``).
        date_series: Optional Series of raw date strings aligned with *results*.

    Returns:
        Sorted list of ``{"date": str, "positive": int, "negative": int}`` dicts.
    """
    buckets: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"positive": 0, "negative": 0}
    )

    if date_series is not None:
        # Try to parse dates; fall back to row-order bucketing on parse failure.
        parsed_dates: list[str | None] = []
        all_parsed = True
        for raw_date in date_series:
            try:
                dt = pd.to_datetime(raw_date, errors="raise")
                parsed_dates.append(dt.strftime("%Y-%m-%d"))
            except Exception:
                all_parsed = False
                parsed_dates.append(None)

        if all_parsed and None not in parsed_dates:
            for result, date_key in zip(results, parsed_dates):
                key = date_key  # type: ignore[assignment]
                if result["label"] == "Positive":
                    buckets[key]["positive"] += 1
                elif result["label"] == "Negative":
                    buckets[key]["negative"] += 1
            return sorted(
                [{"date": k, **v} for k, v in buckets.items()],
                key=lambda x: x["date"],
            )

    # Fallback: group into buckets of 10 rows each, labelled by row range.
    bucket_size = max(10, len(results) // 50)  # at most 50 trend points
    for idx, result in enumerate(results):
        bucket_start = (idx // bucket_size) * bucket_size + 1
        bucket_end = min(bucket_start + bucket_size - 1, len(results))
        key = f"Row {bucket_start}–{bucket_end}"
        if result["label"] == "Positive":
            buckets[key]["positive"] += 1
        elif result["label"] == "Negative":
            buckets[key]["negative"] += 1

    # Keep natural row order (insertion order in Python 3.7+ dicts).
    return [{"date": k, **v} for k, v in buckets.items()]


# ──────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────

@app.route("/")
def index() -> Any:
    """Serve the single-page front-end dashboard.

    Reads ``../frontend/index.html`` relative to this file so the Flask
    process can be started from any working directory.
    """
    index_path = os.path.join(_FRONTEND_DIR, "index.html")
    if not os.path.isfile(index_path):
        return _json_error(
            "frontend/index.html not found. "
            "Ensure the frontend/ folder is at the same level as backend/.",
            404,
        )
    with open(index_path, "r", encoding="utf-8") as fh:
        return fh.read(), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/health", methods=["GET"])
def health() -> Any:
    """Liveness probe — returns model load status.

    Returns:
        200 JSON: ``{"status": "ok", "model_loaded": true}``
        503 JSON: ``{"status": "degraded", "model_loaded": false, "error": "…"}``
    """
    loaded = inference.is_loaded()
    payload: dict[str, Any] = {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
    }
    if not loaded:
        payload["error"] = inference.get_load_error()
    return jsonify(payload), 200 if loaded else 503


@app.route("/api/predict", methods=["POST"])
def predict() -> Any:
    """Single-text sentiment prediction.

    Request body (JSON):
        ``{"text": "string"}``

    Returns:
        200 JSON: ``{"label", "confidence", "probabilities"}``
        400 JSON: ``{"error": "…"}``  — bad / empty input
        503 JSON: ``{"error": "…"}``  — model not loaded
        500 JSON: ``{"error": "…"}``  — inference failure
    """
    if not inference.is_loaded():
        return _json_error(
            f"Model is not loaded: {inference.get_load_error()}", 503
        )

    body = request.get_json(silent=True) or {}
    raw_text: str = body.get("text", "")

    if not isinstance(raw_text, str) or not raw_text.strip():
        return _json_error(
            "Field 'text' is required and must be a non-empty string.", 400
        )

    # Enforce a sensible character-length guard before tokenisation.
    # 512 tokens ≈ ~1 500–2 000 characters for IndoBERT's WordPiece vocab.
    if len(raw_text) > 5000:
        # Silently truncate to keep within reasonable memory bounds.
        raw_text = raw_text[:5000]
        logger.debug("Input text truncated to 5 000 characters.")

    try:
        result = inference.predict_single(raw_text)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        return _json_error(f"Inference failed: {exc}", 500)

    return jsonify({
        "label": result["label"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
    })


@app.route("/api/predict-batch", methods=["POST"])
def predict_batch() -> Any:
    """Batch CSV prediction with trend + keyword extraction.

    Request: ``multipart/form-data`` with field ``file`` (.csv).

    Required column (any name): review / ulasan / content / text
    Optional columns: date / tanggal / at  (for time-based trend)
                      score / rating        (for anomaly detection)

    Returns:
        200 JSON: ``{summary, results, trend, keywords_positive,
                      keywords_negative}``
        400 JSON: ``{"error": "…"}``
        503 JSON: ``{"error": "…"}``  — model not loaded
        500 JSON: ``{"error": "…"}``  — inference failure
    """
    if not inference.is_loaded():
        return _json_error(
            f"Model is not loaded: {inference.get_load_error()}", 503
        )

    if "file" not in request.files:
        return _json_error(
            "No file found. Send a multipart/form-data request with field 'file'.", 400
        )

    uploaded = request.files["file"]
    if not uploaded.filename or not uploaded.filename.lower().endswith(".csv"):
        return _json_error("Only .csv files are accepted.", 400)

    # Parse the CSV in-memory — no disk writes needed.
    try:
        raw_bytes = uploaded.read()
        df = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8", on_bad_lines="skip")
    except Exception as exc:
        return _json_error(f"Failed to parse CSV: {exc}", 400)

    if df.empty:
        return _json_error("The uploaded CSV is empty.", 400)

    # Detect required text column.
    text_col = _detect_column(df, REVIEW_COLUMN_CANDIDATES)
    if text_col is None:
        return _json_error(
            f"Cannot find a review text column. "
            f"Detected columns: {list(df.columns)}. "
            f"Rename your column to one of: {REVIEW_COLUMN_CANDIDATES}.",
            400,
        )

    # Detect optional columns.
    date_col = _detect_column(df, DATE_COLUMN_CANDIDATES)
    rating_col = _detect_column(df, RATING_COLUMN_CANDIDATES)

    # Drop rows with null text, then cap at CSV_MAX_ROWS.
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    truncated = len(df) > CSV_MAX_ROWS
    if truncated:
        logger.info("CSV truncated from %d to %d rows.", len(df), CSV_MAX_ROWS)
    df = df.head(CSV_MAX_ROWS)

    if df.empty:
        return _json_error("CSV has no valid text rows after dropping nulls.", 400)

    texts: list[str] = df[text_col].astype(str).tolist()
    date_series: pd.Series | None = df[date_col] if date_col else None
    rating_series: pd.Series | None = df[rating_col] if rating_col else None

    # Run batch inference.
    try:
        raw_results: list[dict[str, Any]] = inference.predict_batch(texts)
    except Exception as exc:
        logger.exception("Batch inference error: %s", exc)
        return _json_error(f"Batch inference failed: {exc}", 500)

    # Attach optional rating to each result (for anomaly detection on FE).
    for idx, result in enumerate(raw_results):
        if rating_series is not None:
            try:
                result["rating"] = int(rating_series.iloc[idx])
            except (ValueError, TypeError):
                result["rating"] = None
        else:
            result["rating"] = None

    # ── aggregate analytics ──────────────────────────────────────────────
    positive_count = sum(1 for r in raw_results if r["label"] == "Positive")
    negative_count = sum(1 for r in raw_results if r["label"] == "Negative")
    unknown_count  = sum(1 for r in raw_results if r["label"] == "Unknown")
    total = len(raw_results)

    # Keyword extraction per sentiment class.
    pos_texts = [r["cleaned_text"] for r in raw_results if r["label"] == "Positive"]
    neg_texts = [r["cleaned_text"] for r in raw_results if r["label"] == "Negative"]
    keywords_positive = _extract_keywords(pos_texts, top_n=30)
    keywords_negative = _extract_keywords(neg_texts, top_n=30)

    # Trend over time.
    trend = _build_trend(raw_results, date_series)

    # Shape results for the API response — strip internal cleaned_text from
    # the public payload to reduce payload size (raw_text stays for display).
    api_results = [
        {
            "review": r["raw_text"],
            "label": r["label"],
            "confidence": r["confidence"],
            "probabilities": r["probabilities"],
            "rating": r.get("rating"),
        }
        for r in raw_results
    ]

    response_payload: dict[str, Any] = {
        "summary": {
            "total": total,
            "positive": positive_count,
            "negative": negative_count,
            "unknown": unknown_count,
            "positive_pct": round(positive_count / total * 100, 2) if total else 0.0,
            "negative_pct": round(negative_count / total * 100, 2) if total else 0.0,
            "avg_confidence": round(
                sum(r["confidence"] for r in raw_results if r["label"] != "Unknown")
                / max(total - unknown_count, 1),
                4,
            ),
            "truncated": truncated,
        },
        "results": api_results,
        "trend": trend,
        "keywords_positive": keywords_positive,
        "keywords_negative": keywords_negative,
    }
    return jsonify(response_payload)


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # debug=False is intentional — keeps the model from loading twice
    # due to Flask's reloader spawning a second process in debug mode.
    app.run(host="0.0.0.0", port=5000, debug=False)
