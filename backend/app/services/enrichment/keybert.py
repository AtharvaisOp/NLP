"""Local-only KeyBERT keyword extraction service.

The optional dependencies are imported only when this backend is selected.
The configured embedding model must already be present in the local cache;
requests never trigger a Hugging Face download.
"""

from __future__ import annotations

from threading import RLock
from typing import Any

from ml.preprocessing import build_analysis_tokens
from ...config import Settings
from ...exceptions import ServiceFailure
from ..interfaces import KeywordResult, ServiceMetadata


class KeyBERTServiceError(ServiceFailure):
    """Raised when local KeyBERT cannot be loaded or run."""


class KeyBERTKeywordService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = RLock()
        self._keybert: Any | None = None
        self._vectorizer: Any | None = None
        self._device: str | None = None
        self._error: str | None = None

    def metadata(self) -> ServiceMetadata:
        try:
            self._ensure_loaded()
        except KeyBERTServiceError:
            pass
        state = "ready" if self._keybert is not None else "unavailable"
        return ServiceMetadata(
            name="KeyBERT",
            version=self.settings.keyword_model_name,
            device=self._device or "not-loaded",
            state=state,
            production_ready=state == "ready",
            backend="keybert",
            provider="KeyBERT",
            embedding_model=self.settings.keyword_model_name,
        )

    def extract(self, analysis_text: str) -> list[KeywordResult]:
        if not analysis_text.strip():
            return []
        keybert = self._ensure_loaded()
        try:
            raw = keybert.extract_keywords(
                analysis_text,
                keyphrase_ngram_range=(
                    self.settings.keyword_ngram_min,
                    self.settings.keyword_ngram_max,
                ),
                stop_words=None,
                top_n=self.settings.keyword_top_n,
                use_mmr=self.settings.keyword_use_mmr,
                diversity=self.settings.keyword_diversity,
                vectorizer=self._vectorizer,
            )
        except Exception as exc:  # noqa: BLE001 - converted at service boundary
            raise KeyBERTServiceError("KeyBERT extraction failed") from exc
        return [
            KeywordResult(text=str(text), score=max(0.0, min(1.0, float(score))))
            for text, score in raw
            if str(text).strip()
        ]

    def _ensure_loaded(self) -> Any:
        if self._keybert is not None:
            return self._keybert
        if self._error is not None:
            raise KeyBERTServiceError("KeyBERT embedding model is unavailable locally")
        with self._lock:
            if self._keybert is not None:
                return self._keybert
            if self._error is not None:
                raise KeyBERTServiceError("KeyBERT embedding model is unavailable locally")
            try:
                import torch
                from keybert import KeyBERT
                from sentence_transformers import SentenceTransformer
                from sklearn.feature_extraction.text import CountVectorizer

                device = self._select_device(torch, self.settings.model_device)
                embedding_model = SentenceTransformer(
                    self.settings.keyword_model_name,
                    device=device,
                    local_files_only=True,
                )
                self._keybert = KeyBERT(model=embedding_model)
                # Python's default \w token pattern splits Marathi combining
                # marks into syllable fragments. Candidate n-grams must use
                # the shared Unicode-safe tokenizer while embeddings still
                # receive the complete conservative model_text.
                self._vectorizer = CountVectorizer(
                    tokenizer=build_analysis_tokens,
                    token_pattern=None,
                    lowercase=False,
                    ngram_range=(self.settings.keyword_ngram_min, self.settings.keyword_ngram_max),
                    stop_words=None,
                )
                self._device = device
                return self._keybert
            except KeyBERTServiceError:
                raise
            except Exception as exc:  # noqa: BLE001 - safe metadata and API boundary
                self._error = str(exc)
                raise KeyBERTServiceError(
                    "KeyBERT embedding model is unavailable locally"
                ) from exc

    @staticmethod
    def _select_device(torch_module: Any, configured: str) -> str:
        if configured == "cuda":
            if not torch_module.cuda.is_available():
                raise KeyBERTServiceError("CUDA was explicitly requested but is unavailable")
            return "cuda"
        if configured == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        return "cpu"
