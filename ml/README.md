# MahaPulse ML pipeline

The training pipeline prepares the official L3Cube MahaSent-MD Marathi sentiment dataset,
fine-tunes `google/muril-base-cased` for three classes, evaluates a selected
checkpoint, and records a versioned artifact. The FastAPI sentiment service
consumes these artifacts. Optional KeyBERT, BERTopic,
and extractive-summary services without changing the sentiment artifact or
API contract.

The default MuRIL checkpoint is `google/muril-base-cased`, verified against the
MuRIL model evaluated in the L3Cube MahaSent-MD research paper and retained as
the centralized `TrainingConfig.model_name` default.

## Dataset provenance

The default acquisition path is the official [L3Cube MarathiNLP repository](https://github.com/l3cube-pune/MarathiNLP), specifically `L3Cube-MahaSent-MD/MahaSent_All`. The verified all-domain files are `MahaSent_All_Train.csv`, `MahaSent_All_Val.csv`, and `MahaSent_All_Test.csv`, with columns `Unnamed: 0,text,label` and raw labels `-1`, `0`, and `1` for negative, neutral, and positive. The official split is preserved.

The upstream README identifies CC BY-NC-SA 4.0 and research-only/non-commercial share-alike attribution terms. Preparation stores the source URL, resolved Git revision, filenames, timestamp, schema mapping, validation removals, script statistics, and split strategy in `dataset_report.json`. Local input is accepted only when explicitly passed with `--local-path`; its license must be verified by the user.

## Commands

From the repository root:

```powershell
python -m ml.cli prepare
python -m ml.cli prepare --local-path path/to/verified/MahaSent_All
python -m ml.cli train --smoke
python -m ml.cli train --full --model-version muril-mahasent-md-v1
python -m ml.cli evaluate --artifact-dir ml/artifacts/sentiment/muril-mahasent-md-v1
python -m ml.cli inspect-artifact ml/artifacts/sentiment/muril-mahasent-md-v1
python -m ml.cli topics train --processed-dir ml/data/processed/mahasent-md --topic-version bertopic-mahasent-md-v1
python -m ml.cli topics inspect ml/artifacts/topics/bertopic-mahasent-md-v1
```

`prepare` downloads only the verified upstream repository and writes raw data
under ignored `ml/data/raw/` and prepared manifests under ignored
`ml/data/processed/`. Training requires the optional dependencies in
`ml/requirements.txt` and a model download/cache. Smoke mode uses a tiny
deterministic subset, one epoch, and a small step limit; it validates plumbing
only and is never project performance. Full mode uses all prepared records,
validation-based early stopping, and reports held-out test metrics separately.

## Training contract

The classifier consumes the shared `model_text` path from `ml.preprocessing`:
Unicode NFC, control/noise and whitespace normalization, with code mixing,
negation, punctuation, and emojis preserved. Stopwords are not removed and
text is not lemmatized. `analysis_text` remains a separate downstream path.

The artifact contains model/tokenizer files plus `config.json`,
`training_config.json`, `label_mapping.json`, `model_manifest.json`,
`dataset_report.json`, `metrics.json`, and `predictions.jsonl`. Artifacts are
ignored by Git. FastAPI loads the tokenizer/model and manifest while keeping
the existing `/v1/analyze` response contract unchanged.

## Phase 5 enrichment boundary

KeyBERT is optional and uses
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, a cached
multilingual embedding model selected for Marathi and Marathi-English support.
Install `ml/requirements-enrichment.txt` only for real enrichment work. The
backend loads it once per process with local-only model loading; it never
downloads an embedding model during a request. Keyword candidates use the
conservative `model_text` path rather than assuming the deeper analysis path is
always semantically better.

BERTopic training is offline and corpus-level. `ml/topics.py` consumes the
prepared train split and writes ignored artifacts under
`ml/artifacts/topics/<topic_version>/`, including a manifest, topic labels,
training configuration, and preprocessing/dataset provenance. The backend
`BertopicTopicService` only loads an existing artifact and calls `transform`.
Topic IDs are artifact-version-specific; an outlier (`-1`) is represented as a
null topic.

The extractive summary provider is deterministic, UTF-8 safe, and only returns
sentences selected from the input. One-sentence inputs return a null summary.
Its provider-isolated `SummaryService` interface leaves room for a future
generative provider without route or schema changes.

Use `KEYWORD_BACKEND=mock|keybert|disabled`,
`TOPIC_BACKEND=mock|bertopic|disabled`, and
`SUMMARY_BACKEND=mock|extractive|disabled`. Sentiment is required; enrichments
are optional. If an enrichment is unavailable or fails, sentiment still
succeeds and the API returns an empty/null enrichment with a safe warning.
Generated embedding caches and topic artifacts are ignored and must not be
committed. The current MuRIL smoke artifact remains integration-only, not final
project performance or production readiness.

## Final integration training gate

The verified `muril-mahasent-md-smoke-v4` artifact was trained only on the
small local fixture. Its metrics do not measure MahaPulse project performance.
Full official data is prepared separately under ignored
`ml/data/processed/mahasent-md`; never substitute fixture data for a full run.
Use the explicit full-run command after confirming CUDA and resources:

```powershell
python -m ml.cli train --full --processed-dir ml/data/processed/mahasent-md --artifact-root ml/artifacts --model-version muril-mahasent-md-v1 --train-batch-size 4 --eval-batch-size 8
```

The original working Python environment is preserved. GPU diagnosis uses an
isolated environment, and a full CPU training job is not launched when CUDA
is unavailable. See [docs/VALIDATION.md](../docs/VALIDATION.md) for recorded
hardware, full-data split checks, and the final training decision.
