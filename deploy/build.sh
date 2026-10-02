#!/usr/bin/env sh
set -eu
python -m pip install -r deploy/requirements-demo.txt
if [ "${SENTIMENT_BACKEND:-mock}" = muril ]; then
  python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
  python -m pip install -r ml/requirements.txt
  python deploy/fetch-model.py
fi
if [ "${KEYWORD_BACKEND:-mock}" = keybert ]; then
  python -m pip install 'keybert>=0.8,<1' 'sentence-transformers>=5,<7'
  python -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ.get("KEYWORD_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))'
fi
# BERTopic stays disabled until a genuine separately trained artifact exists.
