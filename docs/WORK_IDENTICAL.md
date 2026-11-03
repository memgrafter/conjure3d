# Work-identical outputs

The pipeline in this repository came out of a private production pipeline. The aim: with the same settings, input
image, seed, GPU model and pinned software, this repository gives the same bytes as that pipeline, so either one can
make or re-make an asset.

## What must be the same

- GPU model (an RTX 3090 and an RTX 3090 Ti give different files from the same input).
- The pins: `trellis2/requirements.txt`, the commits and revisions in `trellis2/setup.sh` and `trellis2/run.py`, the
  CuMesh patch, Python 3.12.13. Built with CUDA 13.3's nvcc and `TORCH_CUDA_ARCH_LIST=8.6` in the checks below.
- The settings: pipeline type, seed, texture size, export settings (`EXPORT` in run.py), `--remesh` (lean and
  upstream give the same bytes), the props file's `size_m` and pose options for the post step.
- The input image, byte for byte.

Changes that do not change output bytes: `--host-weights mapped|private`, `--offload persistent` (the default),
`--offload resident` and `--inflight 2` (9 draft GLBs each on an RTX 3090, the same as one runner at a time),
`--hash-stages`, `--save-mesh`, `--trace-decode` (6 GLBs on an RTX 3090 in the production pipeline), `--chunk-norm`
and `--export-from` (below), `--profile`
when it sets the same three values as the flags, warm runs (many objects in one process), the optional GPU and RAM
guards, and the BVH: trianglebvh gives the same bytes as CuMesh's cubvh (below).

## How to check

```sh
source config.env
bash tools/gpu_guard.sh 0 "work-identical check" \
  bash tools/check_work_identical.sh PROPS.json GOLD_ROOT REF_OUT NEW_DIR [REF_MESH_DIR]
```

`REF_OUT` is an earlier run's output folder made on the same GPU model. The script reads the seed, pipeline type and
texture size from the reference `record.json` files, runs the same objects again in a fresh process, runs the post
step when the reference has `asset.glb` files, and compares with `tools/compare_outputs.py` (MD5 of `cut.png`,
`mask-birefnet.png`, `mask.png`, `input-trellis.png`, the decoded mesh `.pt` when both sides saved it, `raw.glb`,
`asset.glb`). Exit status 0 means every compared file is the same.

`asset.glb` stores a short note in `asset.extras` under a key (`--extras-key`, default `post_step`). The production
pipeline used another key; `POST_EXTRAS_KEY=<key>` makes the check use it, so the whole file can be compared. With a
different key only the JSON chunk differs, and `compare_outputs.py` says "BIN chunk same".

## Checks against the production pipeline

Source: the private production pipeline. The runner is at production commit `8dad6d1` (trianglebvh as the BVH,
profiles, `1536_cascade`, the 512 texture weights, stage token counts and RNG hashes, `--trace-decode`), the lean
remesh at `254c7be`, the setup at `d53c7fb`, the post step at `62a2381`, the web export at `11f9587` and the image
stage at `23f22cf`. This repository's runner differs from production in code only where it must: the power check is
optional, the argument parser is a separate function, the weights folder comes from `$T2_WEIGHTS`, the nvdiffrast
bake and CuMesh's cubvh are not offered (`--bvh` has only `ours`), the run stops unless trianglebvh is at its pinned
commit and CuMesh has no compiled cubvh, and `record.json` also has the reference image's MD5 (`gold_md5`). Not
ported yet, because their output-byte checks have not run: production's `--export-from` and `--chunk-norm`.

Setup: this repository's `trellis2/setup.sh env` built a new venv on the production machine on 2026-11-01 (Linux,
CUDA 13.3, `TORCH_CUDA_ARCH_LIST=8.6`). Its package list matches `requirements.txt` (the setup's own check); CuMesh
was built without cubvh (its compiled modules are `_C` and `_cumesh_xatlas` only, and the cubvh submodule folder is
empty); trianglebvh came from a git mirror at `e90201c0`, and its library has the same SHA-256 (`4fe2855a`) as the
production one. trianglebvh's published commit `b8ef3415` (the pin now) has the same `src/` as `e90201c0`; built
into the same venv, its library has the same SHA-256 (`4fe2855a`). The CuMesh binaries differ in bytes from production's (build paths and no cubvh); the outputs do
not. Weights: the same SHA-256-checked files. Fresh processes, seed 42, default `--remesh lean`. Both GPUs at 250 W.
The production references were made with CuMesh's own remesh and with cubvh as the BVH, so this check also covers
the change of BVH.

MD5 match per file (`same` = identical bytes; first 8 hex digits of the MD5):

| Object | GPU | cut.png | masks | input-trellis.png | decoded mesh | raw.glb | asset.glb |
|---|---|---|---|---|---|---|---|
| iron box | RTX 3090 Ti | same `a8778eb7` | same | same | same `8f19521f` | same `a34f1cb2` | - |
| torch | RTX 3090 Ti | same `1df092d4` | same | same | same `346cc117` | same `cd24beb8` | - |
| bowl lamp | RTX 3090 Ti | same `292fc58f` | same | same | same `4255cb6a` | same `cb969cd5` | - |
| brine jar (25.0M decoded faces) | RTX 3090 Ti | same `1f01f0e6` | same | same | same `521e391d` | same `16fa7a66` | - |
| crutch | RTX 3090 Ti | same `c7847750` | same | same | - | same `c8a45230` | same `f97771a1` |
| torch | RTX 3090 | same `7abf4091` | same | same | same `18539ebe` | same `6bbc0d9d` | - |
| bowl lamp | RTX 3090 | same `292fc58f` | same | same | same `658cf571` | same `7e7d31dc` | - |
| grapnel | RTX 3090 | same `b936eb88` | same | same | - | same `5b9dd060` | same `f84a1bcf` |

48 of 48 compared files identical (30 on the 3090 Ti, 18 on the 3090), first on 2026-11-01 with cubvh and again on
2026-11-01 evening with this repository's trianglebvh build (same MD5s). `-`: the reference has no such file. The
`asset.glb` checks used `POST_EXTRAS_KEY` set to the production key.

Draft profile (512 pipeline, 1024 px textures, 100k faces), 2026-11-01 evening, 250 W:

| Object | GPU | Reference | cut.png | masks | input-trellis.png | raw.glb |
|---|---|---|---|---|---|---|
| clapper | RTX 3090 | production runner, `--profile draft`, same evening | same `d6d2d973` | same | same | same `2e00ebc6` |
| clapper | RTX 3090 Ti | production cap study, draft under a 7.63 GiB cap | same `71657668` | same | same | same `741a38bf` |
| bone bell | RTX 3090 Ti | as above | same `e101e43d` | same | same | same `bbbd3cb4` |

15 of 15 compared files identical, twice: once with the settings as flags and once with `--profile draft` (the
check tool passes the reference's profile). The cap does not change bytes: the study's draft files with a 7.63 GiB
cap and with a 15.70 GiB cap have the same MD5s.

`--chunk-norm` and `--export-from`, this repository, RTX 3090 at 250 W, 2026-11-01 (trianglebvh `b8ef3415`): funeral urn
with `--chunk-norm 1048576`: `decode_latent` SHA-256 `6794b010d2f64f26` and `raw.glb` `cdf829bb`, the same as the
production cap study's run without `--chunk-norm`. Grapnel exported alone (`--export-from`) from that study's saved
decoded mesh: `raw.glb` `5b9dd060`, the same as its full run.

BVH: in the production pipeline, trianglebvh against cubvh gave 27 of 27 GLBs byte-identical (21 against existing
production GLBs, 6 pairs; RTX 3090 and RTX 3090 Ti).

Other checks:

- Post step only (CPU): this repository's `post/post_step.py` and the production post step on copies of the same
  10 `raw.glb` files, including objects with `rest_up` and `pose_rule: simple`: 10 of 10 `asset.glb` identical.
- Web export (macOS, Blender 5.2.3, basisu 2.50.0, bun 1.4.4, the same `bun.lock` versions): `web.glb` and `proxy.glb`
  of one object identical to production's.
- Image stage (RTX 3090 Ti, a venv built by `imagegen/setup.sh env`): two images with the logged prompt, seed and
  size identical to production's PNGs (`fc1850ea`, `6c1072c1`). All 19 FLUX.2-klein-4B files matched
  `imagegen/flux2-klein-4b.sha256`.
