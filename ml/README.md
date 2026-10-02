# MahaPulse ML pipeline

Phase 3 prepares the official L3Cube MahaSent-MD Marathi sentiment dataset,
fine-tunes `google/muril-base-cased` for three classes, evaluates a selected
checkpoint, and records a versioned artifact. The FastAPI service is not wired
to these artifacts yet; that is a Phase 4 concern.

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
ignored by Git. Phase 4 can load the tokenizer/model and manifest while keeping
the existing `/v1/analyze` response contract unchanged.
