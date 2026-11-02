#!/usr/bin/env bash
# Writes the hashed lock files that trellis2/setup.sh and imagegen/setup.sh install from, from the pins in each
# requirements.txt (the same versions; no resolving). Each package comes from ONE index:
#   requirements-pytorch.lock  torch, torchvision, triton, xformers: https://download.pytorch.org/whl/cu130 only
#                              (its triton and xformers files differ from PyPI's files of the same name)
#   requirements-pypi.lock     every other package: https://pypi.org/simple only
# Every file of a pinned version on that index is listed by SHA-256; setup.sh installs with --require-hashes.
# Git packages (" @ git+" lines) are pinned by commit in requirements.txt and are not in the locks.
#   bash tools/lock_requirements.sh      # after a change to a requirements.txt; then rebuild and re-check the bytes
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
UV=${UV:-uv}
PYPI_INDEX=https://pypi.org/simple
PYTORCH_INDEX=https://download.pytorch.org/whl/cu130
PYTORCH_PACKAGES='^(torch|torchvision|triton|xformers)=='

lock() {   # $1 requirements.txt, $2 index, $3 grep -E or -v -E, $4 lock file
  grep -v -e '^#' -e ' @ ' "$1" | grep $3 "$PYTORCH_PACKAGES" \
    | "$UV" pip compile --quiet --no-deps --generate-hashes --no-annotate --index-url "$2" --python-version 3.12 \
        --python-platform x86_64-manylinux_2_28 --custom-compile-command "bash tools/lock_requirements.sh" - -o "$4"
}

for req in "$ROOT/trellis2/requirements.txt" "$ROOT/imagegen/requirements.txt"; do
  dir=$(dirname "$req")
  lock "$req" "$PYTORCH_INDEX" -E "$dir/requirements-pytorch.lock"
  lock "$req" "$PYPI_INDEX" -vE "$dir/requirements-pypi.lock"
  echo "$dir: $(grep -c '==' "$dir/requirements-pytorch.lock") + $(grep -c '==' "$dir/requirements-pypi.lock") packages"
done
