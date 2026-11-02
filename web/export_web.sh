#!/usr/bin/env bash
# Optional web export of posed assets. Per object, from SRC/<id>/asset.glb (or raw.glb with --raw), one object at a time:
#   web.glb    full level: seam-preserving simplify to the size class's triangle count, base colour + metallic-roughness
#              textures at the class's size and format, meshopt geometry (webvariant.mjs; options pinned there)
#   proxy.glb  ~1,000 triangles: simplified with seams collapsed, then new UVs and both textures re-baked from the
#              master in Blender (bake_proxy.py), base colour 128 px WebP, metallic-roughness as mean factors
# All files have no generator, names, extras or URIs.
# Size classes by size_m in the props file (longest side): CLASSES below.
#   SRC=... OUT=... WORK=... PROPS=... bash web/export_web.sh [--raw] [ID ...]   (no IDs = all objects with a source GLB)
#   -> $OUT/<id>/{web,proxy}.glb, $OUT/export-<src>.tsv; intermediates in $WORK/<id>/
# Needs: bun (with `bun install` run in this folder), Blender, basisu (env BUN, BLENDER, BASISU override the paths).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${SRC:?set SRC (folder with <id>/asset.glb)}" "${OUT:?set OUT}" "${WORK:?set WORK}" "${PROPS:?set PROPS (props file with size_m)}"
BUN=${BUN:-bun}
BLENDER=${BLENDER:-blender}
export BASISU=${BASISU:-basisu}
MIN_FREE=20     # % of RAM free, checked before each object
START_FREE=35   # % of RAM free before a Blender bake starts
# size class: max size_m (m), web triangles, base colour px, format (metallic-roughness = half the base size).
# One setting was found for all three classes in a size/quality sweep; the table stays so a class can move.
CLASSES=(
  "small 0.3 200000 1024 uastc"
  "medium 0.9 200000 1024 uastc"
  "large 1000 200000 1024 uastc"
)
PROXY_TRIS=1000  # ~26 KB; 1500 gave ~34 KB and +0.03 SSIM at 256 px in the proxy sweep
PROXY_BAKE_PX=512
PROXY_TEX=128

srcname=asset.glb
if [[ ${1:-} == --raw ]]; then srcname=raw.glb; shift; fi
free_pct() {  # % of RAM available (Linux: MemAvailable; macOS: memory_pressure)
  if [[ -r /proc/meminfo ]]; then awk '/^MemTotal/{t=$2} /^MemAvailable/{a=$2} END{print int(100*a/t)}' /proc/meminfo
  else memory_pressure -Q | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'; fi
}
wait_for_blender_slot() {
  local i
  for i in $(seq 1 40); do
    if ! pgrep -x Blender > /dev/null && ! pgrep -x blender > /dev/null && (( $(free_pct) >= START_FREE )); then return 0; fi
    sleep 30
  done
  return 1
}
size_m() { "$BUN" -e 'const p = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8")).props; console.log(p[process.argv[2]]?.size_m ?? "")' "$PROPS" "$1"; }
bytes() { wc -c < "$1" | tr -d ' '; }

ids=("$@")
if (( ${#ids[@]} == 0 )); then
  for d in "$SRC"/*/; do d=${d%/}; [[ -s $d/$srcname ]] && ids+=("$(basename "$d")"); done
fi
mkdir -p "$OUT"
tsv=${TSV:-$OUT/export-${srcname%.glb}.tsv}
[[ -s $tsv ]] || printf 'id\tsize_m\tclass\tweb_tris\tweb_tex\tweb_fmt\tweb_bytes\tproxy_bytes\n' > "$tsv"
cd "$HERE"
for id in "${ids[@]}"; do
  src=$SRC/$id/$srcname
  [[ -s $src ]] || { echo "skip $id: no $srcname"; continue; }
  (( $(free_pct) < MIN_FREE )) && { echo "stop: free memory $(free_pct)%"; exit 2; }
  s=$(size_m "$id")
  [[ -n $s ]] || { echo "skip $id: no size_m"; continue; }
  for c in "${CLASSES[@]}"; do
    read -r cname cmax ctris ctex cfmt <<< "$c"
    if awk -v s="$s" -v m="$cmax" 'BEGIN { exit !(s <= m) }'; then cls=$cname; tris=$ctris; tex=$ctex; fmt=$cfmt; break; fi
  done
  dst=$OUT/$id; work=$WORK/$id
  mkdir -p "$dst" "$work"
  timeout 1800 "$BUN" webvariant.mjs "$src" "$work" "web=${tris}:${tex}:${fmt}" 2>&1 | grep -v '^prune' || { echo "web build failed: $id"; exit 1; }
  timeout 1800 "$BUN" webvariant.mjs "$src" "$work" --render "plow=${PROXY_TRIS}:${PROXY_TEX}:webp:none:loose" 2>&1 | grep -v '^prune' \
    || { echo "proxy simplify failed: $id"; exit 1; }
  wait_for_blender_slot || { echo "stop: no Blender slot for 20 min"; exit 2; }
  timeout 1800 "$BLENDER" -b --factory-startup --python bake_proxy.py -- "$src" "$work/plow.render.glb" "$work/proxy-baked.glb" $PROXY_BAKE_PX factors 2>&1 \
    | grep -E '^(BAKED|COVERAGE|MR|PROXY)|Error' || { echo "bake failed: $id"; exit 1; }
  timeout 600 "$BUN" webvariant.mjs "$work/proxy-baked.glb" "$work" "proxy=full:${PROXY_TEX}:webp:none:lowq" 2>&1 | grep -v '^prune' \
    || { echo "proxy build failed: $id"; exit 1; }
  cp "$work/web.glb" "$dst/web.glb.tmp" && mv "$dst/web.glb.tmp" "$dst/web.glb"
  cp "$work/proxy.glb" "$dst/proxy.glb.tmp" && mv "$dst/proxy.glb.tmp" "$dst/proxy.glb"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$s" "$cls" "$tris" "$tex" "$fmt" "$(bytes "$dst/web.glb")" "$(bytes "$dst/proxy.glb")" >> "$tsv"
  echo "done $id $cls web $(bytes "$dst/web.glb") proxy $(bytes "$dst/proxy.glb")"
done
echo done
