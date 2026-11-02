---
name: conjure3d
description: Make a textured, real-scale 3D object (GLB) from one reference image, or from a text prompt through the optional image stage, on a Linux machine with an NVIDIA GPU where this repository is already set up. Use it when the task is "make a 3D model / GLB / prop of X at N metres" or "re-make asset X and check it is identical".
---

# conjure3d in a session

Setup, flags, formats and env: [AGENTS.md](AGENTS.md). This page is the workflow.

## When to use

- One object per image. Not for scenes, characters with rigs, or animation.
- You know the object's real size (longest side, metres).
- The machine has the venv and weights from `trellis2/setup.sh all`. If `$T2_VENV/bin/python` or `$T2_WEIGHTS` is
  missing, stop and say that setup is needed (it builds CUDA code and downloads large weights; ask before you start it).

## Steps: "make a GLB of a brass lantern, 0.35 m"

1. `source config.env`. Check the GPU is free: `nvidia-smi`. If others share it, wrap GPU commands in
   `bash tools/gpu_guard.sh <gpu> "<what>" ...`.
2. Get a reference image: one object, whole, on a plain background, nothing cut off at the frame edge.
   - The user gave one: copy it to `images/brass-lantern.png`.
   - No image: write a job for the image stage (format in `imagegen/gen.py`), e.g. prompt "a brass oil lantern,
     whole object, plain grey background, studio light", 2 to 4 seeds, `w` 1024, `h` 1024. Run
     `HF_HUB_OFFLINE=1 $IMAGEGEN_VENV/bin/python -I imagegen/gen.py serve` with a `STOP` file in the queue. Look at
     the images and pick one; show them to the user if the choice matters.
3. Write a props file, e.g. `work/props.json`:
   `{"props": {"brass-lantern": {"gold": "brass-lantern.png", "size_m": 0.35}}}`.
   Lying objects (a watch, a knife): add `"rest_up": [0, 0, 1]` so the image's front faces up.
4. Make the 3D object. Several seeds give very different shapes; for a pick, run 2 to 3 seeds into separate folders:
   `HF_HUB_OFFLINE=1 ATTN_BACKEND=xformers PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $T2_VENV/bin/python -I
   trellis2/run.py --job work/props.json:images:work/out-s42 --seed 42`
5. `$T2_VENV/bin/python -I post/post_step.py --props work/props.json --dir work/out-s42`
6. Optional web files: `web/export_web.sh` (see AGENTS.md).

## Check after

- `work/out-s42/brass-lantern/cut.png`: the whole object is in the cut-out, with no background left. If not, use
  another image.
- `asset.json`: `checks.passed` is true; `bbox_m_gltf_xyz` has the expected longest side; `pose.chosen` and
  `pose.why` make sense (a tall lantern should stand: `trellis-up`).
- No `oom-*.json` or `cutout-error.json` in the object folder. On an out-of-memory file, see docs/LOW_VRAM.md.
- `record.json` `export_note` is "as configured". Any other value means the export retry ran and the bytes cannot be
  made again from the inputs alone.
- Look at the result: `blender -b --factory-startup --python render/render_views.py -- work/views work/out-s42/brass-lantern/raw.glb`
  (fixed cameras for the [-0.5, 0.5] box of raw.glb), then view the PNGs. Compare with the reference image; report what differs.

## Re-make or verify an asset

Same GPU model only: `bash tools/check_work_identical.sh PROPS GOLD_ROOT REF_OUT NEW_DIR`. Exit 0 = same bytes.

## Report to the user

The GLB path and its `object.json`, seed, GPU model, `size_m`, the pose chosen, anything from the checks above, and the licence note when
the use is commercial: the DINOv3 licence conditions (THIRD_PARTY_NOTICES.md).
