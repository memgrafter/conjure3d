#!/usr/bin/env bash
# Builds the TRELLIS.2 environment of this pipeline from pinned sources (Linux, NVIDIA GPU, CUDA toolkit 13.x).
#   source config.env
#   setsid -w bash trellis2/setup.sh env       # venv, pinned clones, CuMesh patch, CUDA extensions, checks
#   setsid -w bash trellis2/setup.sh weights   # pinned weight downloads (DINOv3 is gated: accept its licence on
#                                              # Hugging Face first), local SHA-256-checked copy in $T2_WEIGHTS
#   setsid -w bash trellis2/setup.sh all       # both
# Run with setsid so the RAM guard can stop the whole build. No conda, no sudo.
#
# env: the venv is built only from scratch (the script stops if $T2_VENV exists). No nvdiffrast, no nvdiffrec, no
#   RMBG-2.0 weights. The cut-out model is BiRefNet general (MIT): code = github.com/ZhengPeng7/BiRefNet cloned at
#   BIREFNET_SHA into $T2_SRC/ext/BiRefNet (run.py imports it from there; no .pth, because its top-level modules are
#   named config, models, dataset, utils). The GLB bake uses uvrast, installed from GitHub at the commit pinned in
#   requirements.txt. Stops if nvdiffrast can be imported. The GLB export's BVH is trianglebvh (MIT; run.py installs it
#   into CuMesh): built with cargo (Rust; the crate has no dependencies) from TRIANGLEBVH_GIT at TRIANGLEBVH_SHA.
# weights: the BiRefNet weights file and README only (never the model repo's Python files), checked by SHA-256; the
#   files the pipelines (512, 1024_cascade, 1536_cascade) load are copied to $T2_WEIGHTS by run.py --install-weights (SHA-256 checked;
#   stops if that disk would keep less than $T2_WEIGHTS_MIN_FREE_GB, default 20, free); run.py memory-maps them, so
#   all TRELLIS.2 processes on one machine share one host copy. Every repository is downloaded at a pinned revision.
#
# Differences from TRELLIS.2's own setup.sh:
# - torch 2.14.1+cu130 instead of 2.6.0+cu124; the CUDA extensions must be built with a toolkit that matches torch
#   (tested: nvcc 13.3).
# - xformers (TRELLIS.2's documented alternative attention backend) instead of flash-attn 2.7.3: flash-attn has no
#   prebuilt wheel for torch 2.14, and a source build needs a lot of RAM.
# - transformers 4.57.6 (DINOv3 support), so huggingface_hub < 1.0.
# - Not installed: nvdiffrast and nvdiffrec (non-commercial licence; uvrast replaces the two nvdiffrast calls of the
#   bake), gradio/tensorboard/pandas/lpips (app and training only), pillow-simd, briaai/RMBG-2.0 weights (CC-BY-NC;
#   run.py makes the cut-out with the pinned BiRefNet and stubs TRELLIS.2's own remover class).
# - Local patch: FlexGEMM, CuMesh and o-voxel setup.py ask for -std=c++17; torch 2.14's headers need C++20, so sed
#   changes it to -std=c++20 (git diff --stat in the log).
# - Local patch: patches/cumesh-deterministic.patch sorts each CSR adjacency segment of CuMesh after its atomic fill and
#   sums chart perimeters in a fixed order, so the same mesh gives the same GLB bytes every run. Checked with
#   git apply --check (stops if it fails); after the build, $T2_VENV/cumesh-build.json records the commit, the patch
#   SHA-256 and the SHA-256 of each built binary (run.py checks them).
# - CuMesh is built without its cubvh submodule (non-commercial NVIDIA licence): patches/cumesh-no-cubvh.patch;
#   trianglebvh (MIT) gives the BVH that TRELLIS.2's GLB export calls.
# - TRELLIS.2's own Python package is not installed: a .pth file in the venv puts $T2_SRC/TRELLIS.2 on the path.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:-}
case "$MODE" in env|weights|all) ;; *) echo "usage: setup.sh env|weights|all"; exit 2 ;; esac
: "${T2_VENV:?set T2_VENV (see config.example.env)}" "${T2_SRC:?set T2_SRC}" "${HF_HOME:?set HF_HOME}"
V=$T2_VENV SRC=$T2_SRC
PY=$V/bin/python
UV=${UV:-uv}
REQ=$HERE/requirements.txt
RUN=$HERE/run.py
CUMESH_PATCH=$HERE/patches/cumesh-deterministic.patch
PYTHON_VERSION=3.12.13
T2_SHA=75fbf0183001ed9876c8dbb35de6b68552ee08bd        # microsoft/TRELLIS.2 main, 2026-06-05 (MIT)
CUMESH_SHA=12289e1062f0603f2f0d0771b02e1395d247f26f    # JeffreyXiang/CuMesh (MIT); its cubvh submodule is never fetched
CUMESH_PATCH_SHA256=875efddae208bde1351c8061c7f7fb51c90898a89be02f9dacd700cc9d2c80d4  # = run.py CUMESH_PATCH_SHA256
CUMESH_NO_CUBVH=$HERE/patches/cumesh-no-cubvh.patch
CUMESH_NO_CUBVH_SHA256=365bd3713e0a438926dc2693bd50d8387fbe6e0921b818d6993d8f179fa8437a  # = run.py CUMESH_NO_CUBVH_SHA256
TIL_REV=25e0d31ffbebe4b5a97464dd851910efc3002d96        # microsoft/TRELLIS-image-large (MIT): sparse-structure decoder
DINO_REV=ea8dc2863c51be0a264bab82070e3e8836b02d51       # facebook/dinov3-vitl16-pretrain-lvd1689m (DINOv3 License)
FLEXGEMM_SHA=6dd94a859c26ee8246888502eada3dd8ad85532e  # JeffreyXiang/FlexGEMM (MIT)
HF_REV=af44b45f2e35a493886929c6d786e563ec68364d        # microsoft/TRELLIS.2-4B (MIT)
BIREFNET_SHA=f7079632d681ecf12161fcea8905682934e59e51  # ZhengPeng7/BiRefNet main, 2026-10-25 (MIT): cut-out code
BIREFNET_REV=e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4  # HF ZhengPeng7/BiRefNet (general), 2026-02-04 (MIT): weights
BIREFNET_SHA256=9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154  # its model.safetensors
TRIANGLEBVH_SHA=b8ef3415f38839f96969417e6a7e311ccfd73bc8  # trianglebvh (MIT), = run.py TRIANGLEBVH_SHA
# trianglebvh's repository; set TRIANGLEBVH_GIT to use a copy (e.g. file:///path/to/trianglebvh). The pin above still applies.
TRIANGLEBVH_GIT=${TRIANGLEBVH_GIT:-https://github.com/memgrafter/trianglebvh.git}

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} MAX_JOBS=${MAX_JOBS:-2} BUILD_TARGET=cuda
export PATH=$CUDA_HOME/bin:$PATH
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  # the compute capabilities of the visible GPUs, e.g. "8.6" (set TORCH_CUDA_ARCH_LIST yourself to build for others)
  TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd';' -)
fi
export TORCH_CUDA_ARCH_LIST
echo "MODE $MODE venv $V sources $SRC CUDA_HOME $CUDA_HOME TORCH_CUDA_ARCH_LIST $TORCH_CUDA_ARCH_LIST MAX_JOBS $MAX_JOBS $(date +%FT%T)"
if [ -n "${GPU_GUARD_EXPECT_POWER_W:-}" ]; then   # optional: stop unless every GPU has the expected power limit
  nvidia-smi --query-gpu=index,power.limit --format=csv,noheader,nounits | awk -F, -v w="$GPU_GUARD_EXPECT_POWER_W" \
    '{ if ($2+0 < w-0.5 || $2+0 > w+0.5) bad=1 } END { exit bad }' \
    || { echo "STOP: a GPU power limit is not $GPU_GUARD_EXPECT_POWER_W W"; exit 1; }
fi
free_gb() { mkdir -p "$1"; df --output=avail -BG "$1" | tail -1 | tr -dc 0-9; }
MIN_FREE=${SETUP_MIN_FREE_GB:-20}
[ "$(free_gb "$(dirname "$V")")" -gt "$MIN_FREE" ] || { echo "STOP: under $MIN_FREE GB free next to $V"; exit 1; }

# stop the build if the machine's available RAM drops under SETUP_MIN_RAM_MB (default 3000)
MIN_RAM=${SETUP_MIN_RAM_MB:-3000}
( while sleep 5; do
    a=$(free -m | awk '/^Mem:/{print $7}')
    if [ "$a" -lt "$MIN_RAM" ]; then echo "RAM GUARD: available $a MB < $MIN_RAM MB, stopping the build"; kill -TERM -- -$$; exit 1; fi
  done ) &
GUARD=$!
trap 'kill $GUARD 2>/dev/null || true' EXIT

clone() {  # url dir ref [nosub: leave the submodules out]
  [ -d "$2/.git" ] || git clone -q "$1" "$2"
  git -C "$2" checkout -q "$3"
  [ "${4:-}" = nosub ] || git -C "$2" submodule update -q --init --recursive
  echo "PIN $(basename "$2") $(git -C "$2" rev-parse HEAD)"
}
# CuMesh without cubvh: cubvh (CuMesh's submodule) carries a non-commercial NVIDIA licence (LICENSE_NVIDIA), so it is
# never fetched or compiled. patches/cumesh-no-cubvh.patch drops its extension from CuMesh's setup.py and adds a
# stand-in cumesh/_cubvh.py (CuMesh's Python code imports that module); run.py puts trianglebvh in its place.
exclude_cubvh() {
  local c=$SRC/ext/CuMesh
  [ "$(sha256sum < "$CUMESH_NO_CUBVH" | cut -c1-64)" = "$CUMESH_NO_CUBVH_SHA256" ] || { echo "STOP: $CUMESH_NO_CUBVH is not sha256 $CUMESH_NO_CUBVH_SHA256"; exit 1; }
  [ -z "$(ls -A "$c/third_party/cubvh" 2>/dev/null)" ] || { echo "STOP: $c/third_party/cubvh has files; use a fresh T2_SRC"; exit 1; }
  if [ ! -e "$c/cumesh/_cubvh.py" ]; then
    git -C "$c" apply --check "$CUMESH_NO_CUBVH" || { echo "STOP: the no-cubvh patch does not apply to $CUMESH_SHA"; exit 1; }
    git -C "$c" apply "$CUMESH_NO_CUBVH"
  fi
  ! grep -q cubvh "$c/setup.py" || { echo "STOP: $c/setup.py still builds cubvh"; exit 1; }
  echo "CuMesh: cubvh not fetched, not built ($(basename "$CUMESH_NO_CUBVH") sha256 $CUMESH_NO_CUBVH_SHA256)"
}
build() { echo "BUILD $1 start $(date +%T) avail $(free -m | awk '/^Mem:/{print $7}') MB"
  "$UV" pip install --python "$PY" --no-build-isolation --no-deps --reinstall-package "$2" "$1"
  echo "BUILD $1 done $(date +%T)"; }
# the CuMesh source diff, printed byte for byte as the patch file whatever the git config (= run.py CUMESH_DIFF)
cumesh_diff() { git -C "$SRC/ext/CuMesh" -c core.quotepath=on diff --no-color --no-ext-diff --no-renames --diff-algorithm=histogram \
  --abbrev=7 --src-prefix=a/ --dst-prefix=b/ -- src; }
patch_cumesh() {
  [ "$(sha256sum < "$CUMESH_PATCH" | cut -c1-64)" = "$CUMESH_PATCH_SHA256" ] || { echo "STOP: $CUMESH_PATCH is not sha256 $CUMESH_PATCH_SHA256"; exit 1; }
  if [ "$(cumesh_diff | sha256sum | cut -c1-64)" != "$CUMESH_PATCH_SHA256" ]; then
    [ -z "$(cumesh_diff)" ] || { echo "STOP: $SRC/ext/CuMesh/src has changes that are not the patch"; exit 1; }
    git -C "$SRC/ext/CuMesh" apply --check "$CUMESH_PATCH" || { echo "STOP: the CuMesh patch does not apply to $CUMESH_SHA"; exit 1; }
    git -C "$SRC/ext/CuMesh" apply "$CUMESH_PATCH"
  fi
  [ "$(cumesh_diff | sha256sum | cut -c1-64)" = "$CUMESH_PATCH_SHA256" ] && [ -z "$(git -C "$SRC/ext/CuMesh" ls-files --others --exclude-standard -- src)" ] \
    || { echo "STOP: $SRC/ext/CuMesh/src is not exactly the pinned commit plus the patch"; exit 1; }
  git -C "$SRC/ext/CuMesh" diff --stat -- src
  echo "PATCH CuMesh: $(basename "$CUMESH_PATCH") sha256 $CUMESH_PATCH_SHA256"
}
cumesh_record() {  # run.py's CuMesh check compares the binaries with this record
  "$PY" -I - "$SRC/ext/CuMesh" "$V/cumesh-build.json" "$CUMESH_SHA" "$CUMESH_PATCH_SHA256" "$CUMESH_NO_CUBVH_SHA256" <<'EOF'
import hashlib, importlib.util, json, os, sys, time
src, out, commit, patch, no_cubvh = sys.argv[1:6]
pkg = os.path.dirname(importlib.util.find_spec("cumesh").origin)
so = {f: hashlib.sha256(open(os.path.join(pkg, f), "rb").read()).hexdigest() for f in sorted(os.listdir(pkg)) if f.endswith(".so")}
if any(f.startswith("_cubvh") for f in so):
    sys.exit(f"STOP: cubvh was built into {pkg}")
json.dump(dict(commit=commit, patch="patches/cumesh-deterministic.patch", patch_sha256=patch, source=src, package=pkg,
               no_cubvh_patch="patches/cumesh-no-cubvh.patch", no_cubvh_patch_sha256=no_cubvh,
               so_sha256=so, built=time.strftime("%Y-%m-%dT%H:%M:%S")), open(out, "w"), indent=1)
print("CuMesh build record", out, so)
EOF
}

setup_env() {
  [ ! -e "$V" ] || { echo "STOP: $V exists; the venv is built only from scratch (remove it yourself, or set another T2_VENV)"; exit 2; }
  [ -n "$TRIANGLEBVH_GIT" ] || { echo "STOP: set TRIANGLEBVH_GIT (see config.example.env)"; exit 2; }
  command -v "${CARGO:-cargo}" > /dev/null || [ -x "$HOME/.cargo/bin/cargo" ] \
    || { echo "STOP: no Rust toolchain for trianglebvh (cargo on PATH, \$CARGO or ~/.cargo/bin/cargo)"; exit 2; }
  mkdir -p "$SRC/ext"
  clone https://github.com/microsoft/TRELLIS.2.git "$SRC/TRELLIS.2" $T2_SHA
  clone https://github.com/JeffreyXiang/CuMesh.git "$SRC/ext/CuMesh" $CUMESH_SHA nosub
  exclude_cubvh
  clone https://github.com/JeffreyXiang/FlexGEMM.git "$SRC/ext/FlexGEMM" $FLEXGEMM_SHA
  clone https://github.com/ZhengPeng7/BiRefNet.git "$SRC/ext/BiRefNet" $BIREFNET_SHA
  "$UV" python install $PYTHON_VERSION
  "$UV" venv --python $PYTHON_VERSION "$V"
  # every package from its hashed lock (tools/lock_requirements.sh): each file from ONE index and checked by SHA-256
  "$UV" pip install --python "$PY" --require-hashes --no-deps --index-url https://pypi.org/simple -r "$HERE/requirements-pypi.lock"
  "$UV" pip install --python "$PY" --require-hashes --no-deps --index-url https://download.pytorch.org/whl/cu130 \
    -r "$HERE/requirements-pytorch.lock"
  # git packages (uvrast, utils3d), pinned by commit in requirements.txt; built with the venv's pinned setuptools
  grep ' @ git+' "$REQ" | "$UV" pip install --python "$PY" --no-deps --no-build-isolation -r -
  # trianglebvh (Rust library, no crates): with the venv's pinned setuptools, at the pinned commit
  "$UV" pip install --python "$PY" --no-build-isolation --no-deps "trianglebvh @ git+$TRIANGLEBVH_GIT@$TRIANGLEBVH_SHA"
  local site; site=$("$PY" -I -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  echo "$SRC/TRELLIS.2" > "$site/trellis2-src.pth"
  # local patch: torch 2.14's headers need C++20; FlexGEMM, CuMesh and o-voxel ask for -std=c++17 (logged with git diff)
  for d in "$SRC/ext/FlexGEMM" "$SRC/ext/CuMesh" "$SRC/TRELLIS.2/o-voxel"; do
    sed -i 's/-std=c++17/-std=c++20/g' "$d/setup.py"; git -C "$d" diff --stat -- setup.py; done
  # CUDA extensions, one at a time, no build isolation (they import torch at build time)
  "$PY" -I -c "import flex_gemm" 2>/dev/null || build "$SRC/ext/FlexGEMM" flex_gemm
  "$PY" -I -c "import cumesh" 2>/dev/null || { patch_cumesh; build "$SRC/ext/CuMesh" cumesh; cumesh_record; }
  "$PY" -I -c "import uvrast; uvrast.install(); import o_voxel" 2>/dev/null || build "$SRC/TRELLIS.2/o-voxel" o_voxel

  echo "VERSIONS"
  "$UV" pip freeze --python "$PY" | tee "$V/freeze.txt"
  diff <(grep -v '^#' "$REQ" | sort) <(grep -v -E '^(cumesh|flex-gemm|o-voxel|trianglebvh) @' "$V/freeze.txt" | sort) \
    || { echo "STOP: the installed packages differ from requirements.txt"; exit 1; }
  echo "PINS OK: every package as in requirements.txt"
  "$PY" -I -c "import importlib.util as u, sys; sys.exit(any(u.find_spec(n) for n in ('nvdiffrast', 'nvdiffrec_render')))" \
    || { echo "STOP: nvdiffrast or nvdiffrec can be imported in $V"; exit 1; }
  echo "nvdiffrast present: no"
  [ "$(git -C "$SRC/ext/BiRefNet" rev-parse HEAD)" = "$BIREFNET_SHA" ] && [ -z "$(git -C "$SRC/ext/BiRefNet" status --porcelain)" ] \
    || { echo "STOP: $SRC/ext/BiRefNet is not the clean pinned commit $BIREFNET_SHA"; exit 1; }
  "$PY" -I -c "import importlib.metadata as m; d = m.distribution('uvrast'); print('uvrast', d.version, d.read_text('direct_url.json'))"
  "$PY" -I -c "import importlib.metadata as m, trianglebvh as t; d = m.distribution('trianglebvh'); print('trianglebvh', d.version, d.read_text('direct_url.json'), 'library sha256', t.library_sha256(), 'simd', bool(t._load().tb_simd()))"
  if [ "${SETUP_NO_GPU:-0}" = 1 ]; then
    # an image build has no GPU (Dockerfile), and flex_gemm reads the GPU's name when it is imported; run.py makes the
    # CuMesh and trianglebvh checks below at each start
    echo "GPU CHECKS SKIPPED (SETUP_NO_GPU=1): imports and the runner's CuMesh and trianglebvh checks; run.py makes them at each start"
    return
  fi
  "$PY" -I -c "import uvrast; uvrast.install(); import torch, xformers, flex_gemm, cumesh, o_voxel, trellis2, einops, kornia, timm; print('import ok', torch.__version__, torch.version.cuda, trellis2.__file__)"
  # the runner's own CuMesh and lean-remesh checks (stop if CuMesh is not the patched build)
  "$PY" -I - "$RUN" <<'EOF'
import runpy, sys, cumesh
ns = runpy.run_path(sys.argv[1])
print("CuMesh:", ns["cumesh_provenance"](cumesh))
print("lean remesh:", ns["load_sibling"]("cumesh_lean_remesh").install(cumesh, swap=False))
import trianglebvh
print("trianglebvh:", ns["trianglebvh_provenance"](trianglebvh))
EOF
}

setup_weights() {
  : "${T2_WEIGHTS:?set T2_WEIGHTS}"
  [ -x "$PY" ] || { echo "STOP: $V does not exist; run: setup.sh env"; exit 2; }
  local HF=$V/bin/hf   # the venv's pinned huggingface_hub
  [ ! -e "$HF_HOME/hub/models--briaai--RMBG-2.0" ] || { echo "STOP: briaai/RMBG-2.0 (CC-BY-NC) is in $HF_HOME"; exit 1; }
  # TRELLIS.2-4B decoders + flow models (no encoders), TRELLIS-image-large's sparse-structure decoder, DINOv3 ViT-L/16
  # (gated: your Hugging Face account must have accepted the DINOv3 License; log in with `hf auth login` or set HF_TOKEN)
  "$HF" download microsoft/TRELLIS.2-4B --revision $HF_REV --include pipeline.json --include README.md \
    --include "ckpts/*_dec_*" --include "ckpts/*_flow_*"
  "$HF" download microsoft/TRELLIS-image-large ckpts/ss_dec_conv3d_16l8_fp16.json ckpts/ss_dec_conv3d_16l8_fp16.safetensors README.md \
    --revision $TIL_REV
  "$HF" download facebook/dinov3-vitl16-pretrain-lvd1689m --include "*.json" --include "*.safetensors" --include "*.md" \
    --revision $DINO_REV
  # local, memory-mapped copy of the weights the pipeline loads, SHA-256 checked
  "$PY" -I "$RUN" --install-weights --weights "$T2_WEIGHTS"
  # BiRefNet general weights (MIT) at the pinned revision: the weights file and the README only, no Python files
  "$HF" download ZhengPeng7/BiRefNet model.safetensors README.md --revision $BIREFNET_REV
  local bw=$HF_HOME/hub/models--ZhengPeng7--BiRefNet/snapshots/$BIREFNET_REV/model.safetensors
  [ "$(sha256sum "$(readlink -f "$bw")" | cut -d' ' -f1)" = "$BIREFNET_SHA256" ] \
    || { echo "STOP: $bw is not the pinned BiRefNet weights (sha256 $BIREFNET_SHA256)"; exit 1; }
  [ ! -e "$HF_HOME/hub/models--briaai--RMBG-2.0" ] || { echo "STOP: briaai/RMBG-2.0 (CC-BY-NC) is in $HF_HOME"; exit 1; }
  echo "BiRefNet: code $BIREFNET_SHA, weights ZhengPeng7/BiRefNet@$BIREFNET_REV sha256 $BIREFNET_SHA256; RMBG-2.0 absent"
}

case "$MODE" in
  env) setup_env ;;
  weights) setup_weights ;;
  all) setup_env; setup_weights ;;
esac
df -h "$(dirname "$V")"
