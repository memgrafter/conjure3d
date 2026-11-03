#!/usr/bin/env bash
# Apple Silicon route (EXPERIMENTAL, not production): builds trellis.cpp with Metal and the Python venv for the cut-out
# and the post step, and converts our pinned weights to GGUF. macOS on Apple Silicon, Xcode, cmake, uv. No sudo.
#   source config.env
#   bash mac/setup_mac.sh build    # trellis.cpp at TRELLIS_CPP_COMMIT (+ its ggml submodule) -> $TRELLIS_CPP_HOME/{src,build}
#   bash mac/setup_mac.sh env      # $MAC_VENV (mac/requirements.txt) and the pinned BiRefNet code in $T2_SRC/ext/BiRefNet
#   bash mac/setup_mac.sh gguf     # $T2_WEIGHTS + the pinned BiRefNet weights -> $TRELLIS_GGUF (mac/convert_weights.py)
# Weights: the same pinned files as Linux, in $T2_WEIGHTS (trellis2/setup.sh weights with T2_VENV=$MAC_VENV downloads
#   them; or copy a checked $T2_WEIGHTS from another machine: gguf checks every SHA-256).
# trellis.cpp is a download: its installer script is never run; its CMake files were read at the pinned commit. Build
#   choices: Release, Metal (ggml's default on Apple), WebP off (PNG textures, as the CUDA route; WebP would fetch
#   libwebp without a checksum), the optional quad retopology off (it is Linux/Windows only and fetches CGAL/Boost/GMP).
#   CMAKE_OSX_SYSROOT = the active Xcode's macOS SDK (a newer Command Line Tools SDK can carry library stubs that
#   Xcode's linker does not read).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:-}
case "$MODE" in build|env|gguf) ;; *) echo "usage: setup_mac.sh build|env|gguf"; exit 2 ;; esac
[ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ] || { echo "STOP: this route is for macOS on Apple Silicon"; exit 2; }
: "${TRELLIS_CPP_HOME:?set TRELLIS_CPP_HOME (config.example.env, Apple block)}"
TRELLIS_CPP_COMMIT=$(sed -n 's/^TRELLIS_CPP_COMMIT = "\([0-9a-f]*\)".*/\1/p' "$HERE/run_mac.py")
GGML_COMMIT=f8697b33795db2ee6a2569b40853ca6aaa69f9b9   # pwilkin/ggml, the submodule commit of TRELLIS_CPP_COMMIT
BIREFNET_SHA=$(sed -n 's/^BIREFNET_SHA = "\([0-9a-f]*\)".*/\1/p' "$HERE/../trellis2/run.py")
BIREFNET_REV=$(sed -n 's/^BIREFNET_REV = "\([0-9a-f]*\)".*/\1/p' "$HERE/../trellis2/run.py")
SRC=$TRELLIS_CPP_HOME/src BUILD=$TRELLIS_CPP_HOME/build

build() {
  if [ ! -d "$SRC/.git" ]; then
    git clone --no-checkout https://github.com/pwilkin/trellis.cpp "$SRC"
    git -C "$SRC" checkout -q "$TRELLIS_CPP_COMMIT"
  fi
  [ "$(git -C "$SRC" rev-parse HEAD)" = "$TRELLIS_CPP_COMMIT" ] && [ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] \
    || { echo "STOP: $SRC is not the clean commit $TRELLIS_CPP_COMMIT"; exit 1; }
  git -C "$SRC" submodule update --init thirdparty/ggml
  [ "$(git -C "$SRC/thirdparty/ggml" rev-parse HEAD)" = "$GGML_COMMIT" ] || { echo "STOP: ggml is not $GGML_COMMIT"; exit 1; }
  local sdk; sdk="$(xcode-select -p)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
  [ -d "$sdk" ] || { echo "STOP: no macOS SDK at $sdk (install Xcode, then xcode-select -s)"; exit 1; }
  cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DTRELLIS_WEBP=OFF -DCMAKE_OSX_SYSROOT="$sdk"
  grep -q "trellis.cpp: Metal backend enabled" "$BUILD/CMakeCache.txt" 2>/dev/null || grep -q "GGML_METAL:BOOL=ON" "$BUILD/CMakeCache.txt" \
    || { echo "STOP: the build is not using Metal"; exit 1; }
  cmake --build "$BUILD" --target trellis-cli trellis-devices -j "$(sysctl -n hw.ncpu)"
  local cli_sha; cli_sha=$(shasum -a 256 "$BUILD/trellis-cli" | cut -d' ' -f1)
  printf '{"commit": "%s", "ggml_commit": "%s", "cmake": "Release, Metal, TRELLIS_WEBP=OFF", "sdk": "%s", "cli_sha256": "%s"}\n' \
    "$TRELLIS_CPP_COMMIT" "$GGML_COMMIT" "$(basename "$(readlink "$sdk" || echo "$sdk")")" "$cli_sha" > "$BUILD/conjure3d-build.json"
  "$BUILD/trellis-devices" --init | grep -A3 "type        : GPU" || { echo "STOP: trellis.cpp sees no GPU"; exit 1; }
  cat "$BUILD/conjure3d-build.json"
}

env_setup() {
  : "${MAC_VENV:?set MAC_VENV}" "${T2_SRC:?set T2_SRC}"
  [ ! -e "$MAC_VENV" ] || { echo "STOP: $MAC_VENV exists (the venv is built only from scratch)"; exit 1; }
  ${UV:-uv} venv --python 3.13 "$MAC_VENV"
  ${UV:-uv} pip install --python "$MAC_VENV/bin/python" -r "$HERE/requirements.txt"
  local b=$T2_SRC/ext/BiRefNet
  if [ ! -d "$b/.git" ]; then git clone -q https://github.com/ZhengPeng7/BiRefNet.git "$b"; git -C "$b" checkout -q "$BIREFNET_SHA"; fi
  [ "$(git -C "$b" rev-parse HEAD)" = "$BIREFNET_SHA" ] && [ -z "$(git -C "$b" status --porcelain)" ] \
    || { echo "STOP: $b is not the clean pinned commit $BIREFNET_SHA"; exit 1; }
  ! "$MAC_VENV/bin/python" -I -c "import nvdiffrast" 2>/dev/null || { echo "STOP: nvdiffrast is importable in $MAC_VENV"; exit 1; }
  "$MAC_VENV/bin/python" -I -c "import torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
}

gguf() {
  : "${MAC_VENV:?set MAC_VENV}" "${T2_WEIGHTS:?set T2_WEIGHTS}" "${HF_HOME:?set HF_HOME}" "${TRELLIS_GGUF:?set TRELLIS_GGUF}"
  "$MAC_VENV/bin/python" -I "$HERE/convert_weights.py" --weights "$T2_WEIGHTS" --trellis-cpp "$SRC" --out "$TRELLIS_GGUF" \
    --birefnet "$HF_HOME/hub/models--ZhengPeng7--BiRefNet/snapshots/$BIREFNET_REV"
}

case "$MODE" in
  build) build ;;
  env) env_setup ;;
  gguf) gguf ;;
esac
