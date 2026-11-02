#!/usr/bin/env bash
# Manual GPU check: run the pipeline again on objects that already have outputs, and compare the files byte for byte.
#   source config.env
#   bash tools/check_work_identical.sh PROPS.json GOLD_ROOT REF_OUT NEW_DIR [REF_MESH_DIR] [-- extra run.py args]
# - REF_OUT: an earlier run.py OUT folder (<id>/cut.png, raw.glb, record.json, maybe asset.glb), made on the SAME GPU
#   model with the same pins. The objects compared are the ids of PROPS.json that have <id>/raw.glb in REF_OUT.
# - The seed, pipeline type, texture size and GLB face target are read from each reference record.json and must be the
#   same for all. If every record names a profile with no overridden values and this run.py's profile of that name
#   has the same three values, run.py gets --profile NAME; otherwise the three go to run.py as explicit flags.
# - NEW_DIR/out gets the new outputs, NEW_DIR/mesh the decoded meshes (compared when REF_MESH_DIR is given),
#   NEW_DIR/compare.json the result. If REF_OUT has asset.glb files, the post step runs too (PROPS.json needs size_m).
# - Extra run.py args after "--" (for example --remesh upstream). Run it under tools/gpu_guard.sh to pick a GPU.
# - POST_EXTRAS_KEY: the post step's --extras-key, when the reference asset.glb files used another key.
# Exit status: compare_outputs.py's (0 = every compared file is the same).
set -euo pipefail
[ $# -ge 4 ] || { sed -n '2,14p' "$0"; exit 2; }
HERE=$(cd "$(dirname "$0")/.." && pwd)
PROPS=$1 GOLD=$2 REF=$3 NEW=$4; shift 4
REF_MESH=""
if [ $# -gt 0 ] && [ "$1" != "--" ]; then REF_MESH=$1; shift; fi
[ "${1:-}" = "--" ] && shift
: "${T2_VENV:?set T2_VENV}"
PY="$T2_VENV/bin/python -I"
read -r IDS SETTINGS < <($PY - "$PROPS" "$REF" "$HERE/trellis2/run.py" <<'EOF'
import json, os, runpy, sys
props, ref = json.load(open(sys.argv[1]))["props"], sys.argv[2]
ids = [i for i in props if os.path.isfile(os.path.join(ref, i, "raw.glb"))]
if not ids:
    sys.exit(f"STOP: no id of {sys.argv[1]} has a raw.glb in {ref}")
recs = [json.load(open(os.path.join(ref, i, "record.json"))) for i in ids]
settings = {(r["seed"], r["pipeline_type"], r["export"]["texture_size"], r["export"]["decimation_target"]) for r in recs}
if len(settings) != 1:
    sys.exit(f"STOP: the reference records differ in seed, pipeline type, texture size or face target: {settings}")
gpus = {r["gpu"] for r in recs}
if len(gpus) != 1:
    sys.exit(f"STOP: the reference was made on more than one GPU model: {gpus}")
seed, ptype, tex, faces = settings.pop()
profiles = {(r.get("profile") or {}).get("name") if not (r.get("profile") or {}).get("overridden") else None for r in recs}
name = profiles.pop() if len(profiles) == 1 else None
ours = runpy.run_path(sys.argv[3], run_name="check")["PROFILES"].get(name)
if name and ours == dict(pipeline_type=ptype, texture_size=tex, decimation_target=faces):
    print(",".join(ids), f"--seed {seed} --profile {name}")
else:
    print(",".join(ids), f"--seed {seed} --pipeline-type {ptype} --texture-size {tex} --decimation-target {faces}")
print(f"reference GPU: {gpus.pop()}", file=sys.stderr)
EOF
)
[ -n "${IDS:-}" ] || { echo "STOP: no objects to check"; exit 2; }
echo "ids: $IDS"; echo "settings: $SETTINGS"
mkdir -p "$NEW"
# shellcheck disable=SC2086
HF_HUB_OFFLINE=1 ATTN_BACKEND=xformers PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 \
  $PY "$HERE/trellis2/run.py" --job "$PROPS:$GOLD:$NEW/out" --only "$IDS" $SETTINGS --save-mesh "$NEW/mesh" "$@"
if ls "$REF"/*/asset.glb > /dev/null 2>&1; then
  $PY "$HERE/post/post_step.py" --props "$PROPS" --dir "$NEW/out" --only "$IDS" ${POST_EXTRAS_KEY:+--extras-key "$POST_EXTRAS_KEY"}
fi
MESH_ARGS=()
[ -n "$REF_MESH" ] && MESH_ARGS=(--mesh-a "$REF_MESH" --mesh-b "$NEW/mesh")
python3 -I "$HERE/tools/compare_outputs.py" "$REF" "$NEW/out" --only "$IDS" "${MESH_ARGS[@]}" --json "$NEW/compare.json"
