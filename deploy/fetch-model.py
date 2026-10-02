"""Fetch an immutable model archive at build time, never during inference."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen
import zipfile

root = Path(__file__).resolve().parents[1]
destination = (root / os.environ.get('SENTIMENT_MODEL_PATH', 'ml/artifacts/sentiment/muril-mahasent-md-smoke-v4')).resolve()
artifact_root = (root / 'ml/artifacts').resolve()
if not destination.is_relative_to(artifact_root) or destination == artifact_root:
    raise SystemExit('SENTIMENT_MODEL_PATH must identify a directory inside ml/artifacts')
url = os.environ.get('MODEL_ARTIFACT_URL', '')
expected = os.environ.get('MODEL_ARTIFACT_SHA256', '')
if url:
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.username or parsed.password:
        raise SystemExit('Model artifact must use HTTPS without credentials in the URL')
    if not re.fullmatch(r'[a-f0-9]{64}', expected):
        raise SystemExit('MODEL_ARTIFACT_SHA256 is required for an immutable download')
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mahapulse-artifact-') as temporary:
        archive = Path(temporary) / 'model.zip'
        digest = hashlib.sha256()
        size = 0
        with urlopen(url, timeout=120) as incoming, archive.open('wb') as outgoing:
            if urlparse(incoming.url).scheme != 'https':
                raise SystemExit('Artifact redirect must remain HTTPS')
            while chunk := incoming.read(1024 * 1024):
                size += len(chunk)
                if size > 2_000_000_000:
                    raise SystemExit('Artifact archive exceeds 2 GB')
                digest.update(chunk)
                outgoing.write(chunk)
        if digest.hexdigest() != expected:
            raise SystemExit('Artifact archive checksum mismatch')
        with zipfile.ZipFile(archive) as package:
            if sum(member.file_size for member in package.infolist()) > 2_000_000_000:
                raise SystemExit('Expanded model archive exceeds 2 GB')
            for member in package.infolist():
                name = member.filename
                target = (destination / name).resolve()
                if member.is_dir() or '/' in name or '\\' in name or target.parent != destination or member.external_attr >> 16 & 0o170000 == 0o120000:
                    raise SystemExit('Artifact archive must contain regular top-level files only')
            package.extractall(destination)
import sys
sys.path.insert(0, str(root))
from backend.app.services.sentiment.muril import _validate_artifact
metadata = _validate_artifact(destination, os.environ.get('ALLOW_SMOKE_MODEL', 'false').lower() == 'true')
print(f'Validated {metadata.model_version}; smoke_test={metadata.smoke_test}')
