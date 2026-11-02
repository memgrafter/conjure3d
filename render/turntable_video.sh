#!/usr/bin/env bash
# Quick-succession turntable video of TRELLIS.2 GLBs: each object turns once in TURN_S seconds with its name at
# the bottom (drawn by turntable.py), objects back to back, H.264. One Blender at a time (memory); frames are kept per object so a rerun
# only renders what is missing; segments are kept too, so calling it again with a longer list adds only the new
# objects (call it again after each new object).
#   bash render/turntable_video.sh OUT.mp4 WORK_DIR ID=GLB [ID=GLB ...]
#   env: FPS (24) TURN_S (2.0, seconds per object) RES (1080) CRF (30) BLENDER (blender); needs ffmpeg
set -euo pipefail
OUT=$1 WORK=$2; shift 2
FPS=${FPS:-24} SECONDS_PER=${TURN_S:-2.0} RES=${RES:-1080} CRF=${CRF:-30}
HERE=$(cd "$(dirname "$0")" && pwd)
B=${BLENDER:-blender}
mkdir -p "$WORK"
LIST="$WORK/segments.txt"; : > "$LIST"
# background: a slate-blue radial gradient (light behind the object, dark at the edges). The hue differs from the
# browns, blacks, whites and greys of the objects, and the rim light from turntable.py outlines dark objects.
BG="$WORK/background-$RES.png"
[ -f "$BG" ] || ffmpeg -nostdin -loglevel error -y -f lavfi -i "color=black:s=${RES}x${RES}" -frames:v 1 \
  -vf "geq=r='75+(27-75)*min(1\,hypot(X-W/2\,Y-H/2)/(W*0.72))':g='93+(34-93)*min(1\,hypot(X-W/2\,Y-H/2)/(W*0.72))':b='122+(48-122)*min(1\,hypot(X-W/2\,Y-H/2)/(W*0.72))'" "$BG"
for pair in "$@"; do
  id=${pair%%=*} glb=${pair#*=}
  fr="$WORK/frames/$id"
  if [ ! -f "$fr/f0000.png" ]; then
    for i in $(seq 1 120); do   # wait while another Blender runs (at most 10 min)
      pgrep -x Blender >/dev/null || pgrep -f "/blender .*-b" >/dev/null || break
      sleep 5
    done
    "$B" -b --factory-startup --python "$HERE/turntable.py" -- "$FPS" "$SECONDS_PER" "$RES" "$fr" "$(echo "$id" | tr - " ")" "$glb" > "$WORK/blender-$id.log" 2>&1
  fi
  seg="$WORK/seg-$id.mp4"
  [ -f "$seg" ] || ffmpeg -nostdin -loglevel error -y -loop 1 -framerate "$FPS" -i "$BG" -framerate "$FPS" -i "$fr/f%04d.png" \
    -filter_complex "[0][1]overlay=shortest=1,format=yuv420p" \
    -c:v libx264 -preset slow -crf "$CRF" -an "$seg"
  echo "file '$seg'" >> "$LIST"
done
# write to a temp file, then rename: a viewer that opens OUT mid-run always sees a complete video
ffmpeg -nostdin -loglevel error -y -f concat -safe 0 -i "$LIST" -c copy -movflags +faststart "$OUT.part.mp4"
mv "$OUT.part.mp4" "$OUT"
ls -l "$OUT"
