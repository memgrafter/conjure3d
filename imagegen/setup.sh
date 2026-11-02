#!/usr/bin/env bash
# Optional image stage: builds the FLUX.2-klein-4B venv from pinned packages and downloads the model at a pinned
# revision, then checks every downloaded file's SHA-256 (flux2-klein-4b.sha256). Linux, NVIDIA GPU.
#   source config.env
#   bash imagegen/setup.sh env       # venv ($IMAGEGEN_VENV), every package as in requirements.txt
#   bash imagegen/setup.sh weights   # model files into $HF_HOME, SHA-256 checked (Apache-2.0; ~16 GB)
#   bash imagegen/setup.sh all
# Only the diffusers folders and the licence are downloaded; no code from the model repo runs (no trust_remote_code).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:-}
case "$MODE" in env|weights|all) ;; *) echo "usage: setup.sh env|weights|all"; exit 2 ;; esac
: "${IMAGEGEN_VENV:?set IMAGEGEN_VENV (see config.example.env)}" "${HF_HOME:?set HF_HOME}"
UV=${UV:-uv}
PY=$IMAGEGEN_VENV/bin/python
PYTHON_VERSION=3.12.13
REPO=black-forest-labs/FLUX.2-klein-4B
REV=e7b7dc27f91deacad38e78976d1f2b499d76a294   # Apache-2.0
SNAP=$HF_HOME/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/$REV

setup_env() {
  [ ! -e "$IMAGEGEN_VENV" ] || { echo "STOP: $IMAGEGEN_VENV exists; the venv is built only from scratch"; exit 2; }
  "$UV" python install $PYTHON_VERSION
  "$UV" venv --python $PYTHON_VERSION "$IMAGEGEN_VENV"
  # every package from its hashed lock (tools/lock_requirements.sh): each file from ONE index and checked by SHA-256
  "$UV" pip install --python "$PY" --require-hashes --no-deps --index-url https://pypi.org/simple -r "$HERE/requirements-pypi.lock"
  "$UV" pip install --python "$PY" --require-hashes --no-deps --index-url https://download.pytorch.org/whl/cu130 \
    -r "$HERE/requirements-pytorch.lock"
  "$UV" pip freeze --python "$PY" > "$IMAGEGEN_VENV/freeze.txt"
  diff <(grep -v '^#' "$HERE/requirements.txt" | sort) <(sort "$IMAGEGEN_VENV/freeze.txt") \
    || { echo "STOP: the installed packages differ from requirements.txt"; exit 1; }
  echo "PINS OK: every package as in requirements.txt"
}

setup_weights() {
  [ -x "$PY" ] || { echo "STOP: $IMAGEGEN_VENV does not exist; run: setup.sh env"; exit 2; }
  "$IMAGEGEN_VENV/bin/hf" download $REPO --revision $REV --include "model_index.json" --include "scheduler/*" \
    --include "text_encoder/*" --include "tokenizer/*" --include "transformer/*" --include "vae/*" --include "LICENSE.md"
  # every file of the snapshot must have the pinned SHA-256 (sha256sum follows the cache's symlinks)
  (cd "$SNAP" && sha256sum --check --strict --quiet "$HERE/flux2-klein-4b.sha256") \
    || { echo "STOP: a file in $SNAP is not the pinned file"; exit 1; }
  echo "FLUX.2-klein-4B@$REV: every file SHA-256 checked ($(wc -l < "$HERE/flux2-klein-4b.sha256") files)"
}

case "$MODE" in
  env) setup_env ;;
  weights) setup_weights ;;
  all) setup_env; setup_weights ;;
esac
