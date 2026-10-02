"""Deterministic local extractive summarization."""

from __future__ import annotations

from collections import Counter
import re

from ml.preprocessing import preprocess_text

from ...config import Settings
from ..interfaces import ServiceMetadata, SummaryResult


_SENTENCE_RE = re.compile(r"(?<=[।.!?])\s+|[\r\n]+")


class ExtractiveSummaryService:
    """Selects source sentences; it never generates new content."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def metadata(self) -> ServiceMetadata:
        return ServiceMetadata(
            name="summarization",
            version="extractive-v1",
            device="cpu",
            state="ready",
            production_ready=True,
            backend="extractive",
            provider="extractive",
        )

    def summarize(self, model_text: str) -> SummaryResult:
        sentences = [item.strip() for item in _SENTENCE_RE.split(model_text) if item.strip()]
        if len(sentences) <= 1:
            return SummaryResult(text=None, provider="extractive")

        frequencies = Counter(
            token
            for sentence in sentences
            for token in preprocess_text(sentence).analysis_tokens
        )
        max_frequency = max(frequencies.values(), default=1)
        scored = []
        for index, sentence in enumerate(sentences):
            tokens = preprocess_text(sentence).analysis_tokens
            score = sum(frequencies[token] / max_frequency for token in tokens)
            # Position is only a deterministic tie-breaker and does not add content.
            scored.append((score, -index, index, sentence))
        selected = sorted(scored, reverse=True)[: self.settings.summary_max_sentences]
        selected_sentences = [item[3] for item in sorted(selected, key=lambda item: item[2])]
        return SummaryResult(text=" ".join(selected_sentences), provider="extractive")
