# MahaPulse ML pipeline

The training pipeline prepares the official L3Cube MahaSent-MD Marathi sentiment dataset,
fine-tunes `google/muril-base-cased` for three classes, evaluates a selected
checkpoint, and records a versioned artifact. The FastAPI sentiment service
consumes these artifacts. Optional KeyBERT, BERTopic,
and extractive-summary services enrich results without changing the sentiment
artifact or API contract.

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
python -m ml.cli train --full --processed-dir ml/data/processed/mahasent-md --artifact-root ml/artifacts --model-version muril-mahasent-md-v1 --train-batch-size 4 --eval-batch-size 8
python -m ml.cli inspect-artifact ml/artifacts/sentiment/muril-mahasent-md-v1
python -m ml.cli topics train --processed-dir ml/data/processed/mahasent-md --topic-version bertopic-mahasent-md-v1
python -m ml.cli topics inspect ml/artifacts/topics/bertopic-mahasent-md-v1
```

`prepare` downloads only the verified upstream repository and writes raw data
under ignored `ml/data/raw/` and prepared manifests under ignored
`ml/data/processed/`. Training requires the optional dependencies in
`ml/requirements.txt` and a model download/cache. Smoke mode uses a tiny
deterministic subset, one epoch, and a small step limit; it validates plumbing
only and is never project performance. Full mode uses all prepared train
records and validation macro F1 for checkpoint selection/early stopping. After
selection completes, the same command loads the held-out test split and makes
one final prediction pass, recording its metrics separately from validation.
Do not run `evaluate` after a completed full run: the final test evaluation is
already stored. Reserve the separate command for artifacts without finalized
test evidence.

Recover an interrupted run with the same version/configuration and
`--resume-from-checkpoint ml/artifacts/sentiment/muril-mahasent-md-v1/_trainer/checkpoint-<step>`.
The CLI validates checkpoint location and Trainer state/optimizer files; a
finalized artifact cannot be resumed. Per-run records, effective arguments,
validation history and checkpoints remain under ignored `_trainer/`.

## Training contract

The classifier consumes the shared `model_text` path from `ml.preprocessing`:
Unicode NFC, control/noise and whitespace normalization, with code mixing,
negation, punctuation, and emojis preserved. Stopwords are not removed and
text is not lemmatized. `analysis_text` remains a separate downstream path.

The artifact contains model/tokenizer files plus `config.json`,
`training_config.json`, `label_mapping.json`, `model_manifest.json`,
`dataset_report.json`, `metrics.json`, and `predictions.jsonl`. Artifacts are
ignored by Git. FastAPI loads the tokenizer/model and manifest while keeping
the existing `/v1/analyze` response contract unchanged. A completed full run
starts with `smoke_test=false`, `production_ready=false`; training alone does
not attest that API integration passed.

Verify the manifest's SHA-256/byte metadata, independently reload with
`AutoTokenizer.from_pretrained(local_artifact, local_files_only=True)` and
`AutoModelForSequenceClassification.from_pretrained(local_artifact, local_files_only=True)`,
then validate real FastAPI inference, persistence, analytics and exports.
After these gates pass:

```powershell
python scripts/validate-production-model.py --artifact-dir ml/artifacts/sentiment/muril-mahasent-md-v1 --device cuda --keywords auto
$validationReport = Read-Host "Path to the passing prepromotion validation_report.json printed above"
python -m ml.cli promote-artifact --artifact-dir ml/artifacts/sentiment/muril-mahasent-md-v1 --validation-report $validationReport --api-integration-validated
python scripts/validate-production-model.py --artifact-dir ml/artifacts/sentiment/muril-mahasent-md-v1 --device cuda --keywords auto --expect-promoted
```

The validator saves an actual `validation_report.json` path in its output;
`$validationReport` must point to that passing **prepromotion** report, not a
handwritten attestation. It independently reloads the saved tokenizer/model,
checks SHA-256 coverage, exercises real FastAPI startup and OpenAPI, validates
four Marathi/code-mixed examples without treating them as metrics, and runs a
five-row CSV through fresh Alembic-migrated SQLite persistence, pagination,
analytics and CSV/JSON export. It also checks CORS and input limits. All model
loads are local-only; `--keywords auto` enables cached real KeyBERT when
available, never a mock substitute. Use `--device cpu` when CUDA is unavailable.
Evidence, the CSV and exports stay under ignored `ml/artifacts/validation/`
outside the immutable sentiment payload. The final command reruns integration
with the promoted flags; neither validation pass predicts on the held-out test.

Promotion checks complete held-out prediction evidence, all required top-level
artifact hashes, required metadata, the report's artifact/version/hash binding
and independent local reload before writing lifecycle gates and
`production_ready=true`. It recomputes classification metrics from existing
test predictions without running a second test inference pass. The manifest is
excluded from its own hash list. Required sentiment readiness is separate from overall
readiness: disabled optional topics leave `/ready` degraded with usable
sentiment. No request-time model download or mock fallback is introduced.

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
committed. The historical MuRIL smoke artifact remains integration-only, not final
project performance or production readiness.

## Historical smoke artifact and full-run lifecycle

The verified `muril-mahasent-md-smoke-v4` artifact was trained only on the
small local fixture. Its metrics do not measure MahaPulse project performance.
Full official data is prepared separately under ignored
`ml/data/processed/mahasent-md`. The verified prepared metadata contains
60,396 records: 47,730 train, 5,922 validation and 6,744 test, from upstream
revision `8ee29fa1329d6a841030eb46659d3c10614b5e59`. Every split contains
negative, neutral and positive, with IDs 0, 1 and 2 respectively; normalized
texts have zero intersections between split pairs. Use the explicit full-run
command after confirming CUDA and resources:

```powershell
python -m ml.cli train --full --processed-dir ml/data/processed/mahasent-md --artifact-root ml/artifacts --model-version muril-mahasent-md-v1 --train-batch-size 4 --eval-batch-size 8
```

The original working Python environment is preserved. The isolated training
executable is `C:\Users\athar\.codex\mahapulse-training-cuda\Scripts\python.exe`.
The completed full lifecycle used Python 3.13.4, PyTorch 2.11.0+cu128, CUDA 12.8 and
Transformers 5.15.0 on an RTX 3050 Laptop GPU with 6 GiB VRAM. Its recorded
configuration uses batch size 4, evaluation batch size 8, 256-token truncation,
three epochs, learning rate 2e-5 and seed 42. It completed 35,799 optimizer
steps and selected `_trainer/checkpoint-23866` (epoch 2) by validation macro
F1 `0.7997525380776924`; epoch 3 did not improve that metric. The single final
held-out test pass produced accuracy `0.8010083036773428`, macro F1
`0.8005925770515055` and weighted F1 `0.8005915632499007`.
Accepted training took approximately 62 minutes 14 seconds, including the
first two epochs and successful resumed third epoch, excluding discarded
replays and idle time. No CUDA OOM or batch-size reduction occurred.

The final ignored artifact is `ml/artifacts/sentiment/muril-mahasent-md-v1`,
with `smoke_test=false` and 6,744 final predictions containing canonical labels,
confidence and all three probabilities. Recorded SHA-256 for the
950,257,668-byte `model.safetensors` is
`c3c1855edc451d884c93ab5f360a60979ea46cf5e8fa2c62b9087ef028c1bdec`.
Its manifest now has `production_ready=true` after the independent reload,
integrity and real API validation gates passed. This is local artifact
promotion, not a claim that the free hosted mock demo serves the full model. See
[docs/VALIDATION.md](../docs/VALIDATION.md) for completed training, selected
checkpoint, separate validation/test metrics and runtime gates.
