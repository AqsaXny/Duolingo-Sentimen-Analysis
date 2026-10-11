"""
utils/preprocessor.py
Text preprocessing pipeline for Indonesian Duolingo reviews.
Steps:
  1. Lowercase
  2. Remove URLs, mentions, hashtags
  3. Remove non-alphanumeric characters (keep spaces)
  4. Normalize repeated characters (e.g. "bagussss" → "bagus")
  5. Slang normalization via slang_dict.json
  6. Optional: Sastrawi stemming (disabled by default for speed)
"""

import re
import json
import os

# ---------------------------------------------------------------------------
# Load slang dictionary once at import time
# ---------------------------------------------------------------------------
_SLANG_PATH = os.path.join(os.path.dirname(__file__), "slang_dict.json")
with open(_SLANG_PATH, "r", encoding="utf-8") as f:
    SLANG_DICT: dict[str, str] = json.load(f)

# Sort by length descending so multi-word phrases match before single words
_SLANG_SORTED = sorted(SLANG_DICT.items(), key=lambda x: len(x[0]), reverse=True)


def _remove_noise(text: str) -> str:
    """Remove URLs, mentions, hashtags, and special characters."""
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                        # mentions
    text = re.sub(r"#\w+", " ", text)                        # hashtags
    text = re.sub(r"[^\w\s]", " ", text)                     # punctuation
    text = re.sub(r"\d+", " ", text)                         # digits
    return text


def _normalize_repeated(text: str) -> str:
    """Collapse 3+ consecutive identical characters to 2 (heuristc)."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def _apply_slang(text: str) -> str:
    """Word-boundary slang replacement."""
    for slang, normal in _SLANG_SORTED:
        pattern = r"\b" + re.escape(slang) + r"\b"
        text = re.sub(pattern, normal, text, flags=re.IGNORECASE)
    return text


def clean_text(text: str, use_stemmer: bool = False) -> str:
    """
    Full preprocessing pipeline.
    Returns cleaned Indonesian text ready for tokenisation.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = _remove_noise(text)
    text = _normalize_repeated(text)
    text = _apply_slang(text)
    text = re.sub(r"\s+", " ", text).strip()

    if use_stemmer:
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            text = stemmer.stem(text)
        except ImportError:
            pass  # Sastrawi not installed — skip silently

    return text
