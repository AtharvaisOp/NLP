"""Two-path text preparation for Marathi and Marathi-English input.

The classifier path is intentionally conservative: MuRIL receives
``model_text`` with Unicode and noise normalization only. Stop-word removal,
lemmatization, and aggressive punctuation stripping are not performed there.
The analysis path is separately tokenized for KeyBERT, BERTopic, and EDA.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


PREPROCESSING_VERSION = "model-text-v1"


_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"]+")
_EMAIL_RE = re.compile(r"(?i)\b[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+\b")
_MENTION_RE = re.compile(r"(?<!\w)@[\w_]+")
_CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\ufeff\u00ad]")

# Keep Devanagari, Roman/code-mixed words, numbers, apostrophes, and hyphens.
# Punctuation is deliberately excluded from analysis tokens, not model_text.
_TOKEN_RE = re.compile(
    r"[\u0900-\u0963\u0966-\u097f]+(?:[-'’][\u0900-\u0963\u0966-\u097f]+)*"
    r"|[A-Za-z]+(?:[-'’][A-Za-z]+)*"
    r"|\d+(?:[.,]\d+)?"
)
_PLACEHOLDER_RE = re.compile(r"<(?:URL|EMAIL|USER)>")


@dataclass(frozen=True)
class PreparedText:
    """Outputs of the deterministic preprocessing boundary."""

    model_text: str
    analysis_text: str
    analysis_tokens: tuple[str, ...]


def normalize_model_text(text: str) -> str:
    """Normalize text without destroying sentiment or contextual signals.

    NFC normalization, control-character cleanup, URL/email/mention
    placeholders, and whitespace collapsing are safe shared operations.
    Roman code-mixed text, Devanagari, negation words, emojis, hashtags, and
    useful punctuation are retained. Repeated punctuation is not collapsed.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _CONTROL_RE.sub(" ", normalized)
    normalized = _EMAIL_RE.sub("<EMAIL>", normalized)
    normalized = _URL_RE.sub("<URL>", normalized)
    normalized = _MENTION_RE.sub("<USER>", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def build_analysis_tokens(model_text: str) -> tuple[str, ...]:
    """Create deeper-cleaning tokens for keyword/topic analysis only.

    This function does not remove stop words or lemmatize. Downstream
    KeyBERT/BERTopic configuration may choose domain-specific handling, but
    the shared preprocessor keeps those decisions explicit and separate from
    MuRIL classification.
    """

    without_placeholders = _PLACEHOLDER_RE.sub(" ", model_text)
    return tuple(token.casefold() for token in _TOKEN_RE.findall(without_placeholders))


def preprocess_text(text: str) -> PreparedText:
    """Return both canonical text paths from one raw input."""

    model_text = normalize_model_text(text)
    analysis_tokens = build_analysis_tokens(model_text)
    return PreparedText(
        model_text=model_text,
        analysis_text=" ".join(analysis_tokens),
        analysis_tokens=analysis_tokens,
    )
