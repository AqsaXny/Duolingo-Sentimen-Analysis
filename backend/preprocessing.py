"""
backend/preprocessing.py
========================
Text preprocessing pipeline for Indonesian Duolingo reviews.

Pipeline stages (always applied in this order):
  1. cleansing()     — strip noise: URLs, mentions, hashtags, emojis,
                       digits, excessive punctuation, non-alphanumeric chars.
  2. case_folding()  — lowercase the entire string.
  3. normalization() — replace Indonesian slang / abbreviations using
                       slang_dict.json loaded once at import time.
  4. tokenize()      — convert cleaned text to IndoBERT input tensors.

How to run (standalone smoke-test):
    python preprocessing.py
"""

import re
import json
import os
import unicodedata
import logging
from typing import Dict

import torch
from transformers import AutoTokenizer

# ── logging ────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────
_HERE: str = os.path.dirname(os.path.abspath(__file__))
_SLANG_PATH: str = os.path.join(_HERE, "slang_dict.json")
_MODEL_DIR: str = os.path.join(_HERE, "model")

# Maximum token length accepted by the model (spec cap: 512).
# IndoBERT base uses 512 but we honour the spec's stated limit.
MAX_TOKEN_LEN: int = 512

# ── load slang dictionary once at import time ──────────────────────────────
with open(_SLANG_PATH, "r", encoding="utf-8") as _f:
    _RAW_SLANG: Dict[str, str] = json.load(_f)

# Sort by key length descending so multi-word phrases are matched before
# their constituent single words (greedy longest-match strategy).
_SLANG_PAIRS: list[tuple[str, str]] = sorted(
    _RAW_SLANG.items(), key=lambda kv: len(kv[0]), reverse=True
)

# ── lazy tokenizer (loaded on first call to tokenize()) ───────────────────
_tokenizer: AutoTokenizer | None = None


def _get_tokenizer() -> AutoTokenizer:
    """Return the module-level tokenizer, loading it lazily on first call.

    Lazy loading keeps import time fast; the tokenizer is shared across
    all subsequent calls within the same process.
    """
    global _tokenizer
    if _tokenizer is None:
        logger.info("Loading tokenizer from %s …", _MODEL_DIR)
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR, local_files_only=True)
    return _tokenizer


# ──────────────────────────────────────────────────────────────────────────
# Public pipeline functions
# ──────────────────────────────────────────────────────────────────────────

def cleansing(text: str) -> str:
    """Remove textual noise from *text*.

    Removes, in order:
    - HTTP/HTTPS URLs and bare ``www.`` links
    - @mentions and #hashtags
    - Emoji and other non-BMP Unicode codepoints
    - ASCII punctuation runs longer than 1 character
    - Decimal digits
    - Any character that is not ASCII alphanumeric or a space

    Args:
        text: Raw review string (may be None or non-string).

    Returns:
        Noise-free string, or empty string if *text* is falsy.
    """
    # Guard: accept only strings; return empty for None / non-string input.
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. URLs (http / https / bare www.)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 2. @mentions and #hashtags (keep the token boundary intact)
    text = re.sub(r"[@#]\w+", " ", text)

    # 3. Emoji — strip every codepoint in Unicode "So" (Symbol, Other)
    #    and any surrogate / non-BMP character.
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in {"So", "Cn", "Cs"}
        and ord(ch) < 0x10000
    )

    # 4. Repeated punctuation (e.g. "!!!" → "!") — collapse to single char
    text = re.sub(r"([^\w\s])\1+", r"\1", text)

    # 5. Remove all decimal digits
    text = re.sub(r"\d+", " ", text)

    # 6. Remove any character that is not a Latin letter or whitespace
    #    (keeps Indonesian diacritics via the \w class under UNICODE mode)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # 7. Collapse runs of whitespace to a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def case_folding(text: str) -> str:
    """Lowercase *text*.

    Lowercasing is applied *after* cleansing so that URL/hashtag removal
    patterns remain case-insensitive (they already use re.IGNORECASE where
    needed), and *before* normalization so slang keys are matched correctly.

    Args:
        text: Cleansed string.

    Returns:
        Lowercase string.
    """
    return text.lower()


def normalization(text: str) -> str:
    """Replace Indonesian slang / abbreviations with their formal equivalents.

    Uses the module-level ``_SLANG_PAIRS`` list (sorted longest-first) so
    multi-word phrases like ``"ga bisa"`` are substituted before the
    single-word ``"ga"`` key can produce a partial replacement.

    Each replacement uses a word-boundary regex so ``"gak"`` does not
    accidentally match inside ``"kagak"`` before ``"kagak"`` is processed.

    Args:
        text: Lowercased string.

    Returns:
        String with all matched slang tokens replaced.
    """
    for slang, formal in _SLANG_PAIRS:
        # \b ensures we only match whole words / token boundaries.
        pattern = r"\b" + re.escape(slang) + r"\b"
        text = re.sub(pattern, formal, text)
    return text


def preprocess(text: str) -> str:
    """Apply the full preprocessing pipeline: cleanse → fold → normalise.

    Convenience wrapper used by inference.py to obtain a clean string
    before tokenisation (so callers can log / store the cleaned text).

    Args:
        text: Raw review string.

    Returns:
        Fully cleaned, lowercased, slang-normalised string.
    """
    return normalization(case_folding(cleansing(text)))


def tokenize(
    text: str,
    max_length: int = MAX_TOKEN_LEN,
) -> Dict[str, torch.Tensor]:
    """Tokenize *text* and return tensors ready for IndoBERT inference.

    Applies the full preprocessing pipeline internally so callers do not
    need to pre-clean the text (though it is safe to pass already-cleaned
    text — the pipeline is idempotent for clean input).

    Args:
        text:       Raw or pre-cleaned review string.
        max_length: Maximum subword token count (default: ``MAX_TOKEN_LEN``).
                    Inputs longer than this are *silently truncated* as
                    required by the spec.

    Returns:
        Dict with ``input_ids``, ``attention_mask``, and (if the tokenizer
        produces them) ``token_type_ids`` — all as batched 2-D tensors of
        shape ``(1, max_length)``.
    """
    cleaned: str = preprocess(text)

    # Return a minimal zero-tensor batch when the string is empty after
    # cleaning — the caller (inference.py) handles this case explicitly.
    tokenizer = _get_tokenizer()
    encoding = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    return dict(encoding)


# ──────────────────────────────────────────────────────────────────────────
# Smoke-test (run directly: python preprocessing.py)
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    _cases = [
        "Aplikasi ini gak bisa dibuka bgt parah!! 😡😡",
        "https://play.google.com @duolingo #belajar mantap bgt!!!",
        "udh 3 hari error terus, kapan difix min???",
        "bagus bngt, lg sering update fitur baru",
        "",
        None,  # type: ignore[arg-type]
    ]

    for raw in _cases:
        result = preprocess(raw)  # type: ignore[arg-type]
        print(f"IN : {repr(raw)}")
        print(f"OUT: {repr(result)}")
        print()
