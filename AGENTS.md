# conjure3d: agent reference

Image of one object -> textured GLB at real size. Linux, NVIDIA, CUDA 13. Run Python with `-I`.

## Stages
- image (optional): `$IMAGEGEN_VENV/bin/python -I imagegen/gen.py serve`. Job JSON in `$IMAGEGEN_HOME/queue/` (format: gen.py header) -> `out/<round>/<id>.png`, `log.jsonl`. `queue/STOP` ends it.
- 3D: `$T2_VENV/bin/python -I trellis2/run.py --job PROPS:GOLD_ROOT:OUT` (env `HF_HUB_OFFLINE=1 ATTN_BACKEND=xformers PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`). Reads `GOLD_ROOT/<gold>` -> `OUT/<id>/cut.png`, `raw.glb`, `record.json`; failures `oom-*.json`, `cutout-error.json`. Flags: `--only a,b`, `--seed 42`, `--profile clean|draft|best` (sets `--pipeline-type 1024_cascade|512|1536_cascade`, `--texture-size`, `--decimation-target`; each flag overrides), `--texture-size 4096`, `--save-mesh DIR`, `--cutout-only`, `--vram-cap-gb G`, `--trace-decode FILE` (GPU memory per decoder block), `--chunk-norm ROWS` (off by default; decoders' LayerNorm in row blocks, same bytes, lower decode peak; with 1048576 at a 15.70 GiB cap 9 of 10 hard objects pass, 6 without), `--export-from MESH.pt... --export-out DIR` (GLB export only, from `--save-mesh` files; no cut-out, no generation, no weights; same GLB as the full run), `--queue QDIR --stream S --log DIR` (one runner per GPU).
- post: `$T2_VENV/bin/python -I post/post_step.py --props PROPS --dir OUT` -> `asset.glb`, `asset.json`, `object.json`.
- one object, both steps: `python3 -I conjure3d.py IMAGE SIZE_M [--profile clean|draft|best] [--seed N] [--out DIR] [--lie-flat] [--on-oom stop|draft] [--chunk-norm]` -> `OUT/<id>-<profile>[-chunk-norm]-seed<N>/<id>/asset.glb`. passes `--profile` to run.py; clean = run.py's default (same bytes); draft = 512 pipeline, 1024 px, 100k faces. Exit 3 = out of GPU memory; the hint offers `--chunk-norm` (generation OOM, clean) before draft.
- Apple Silicon (experimental, not production, other bytes): `mac/setup_mac.sh build|env|gguf`, `conjure3d.py ... --backend trellis-cpp` (mac/run_mac.py: same BiRefNet cut-out on MPS, trellis.cpp `c0bed38c` on Metal with GGUF converted from our pins, same post step; draft/clean only; seed >= 1). docs/MAC.md.
- machine check: `python3 -I tools/preflight.py [--container --data DIR]` (PASS/WARN/FAIL lines, exit 1 on FAIL). Container (no compiling on the host, weights in a mounted /data): docs/CONTAINER.md.
- web (optional): `cd web && bun install`; `SRC= OUT= WORK= PROPS= bash web/export_web.sh` -> `web.glb` (200k tris, KTX2), `proxy.glb` (~1k tris); check `bun web/verify_export.mjs`.
- renders (optional): `render/render_views.py`, `render/turntable_video.sh` (Blender).

## Formats
- props: `{"props": {id: {"gold": path, "size_m": m, "rest_up"?: [x,y,z], "pose_rule"?: "simple"}}}` (examples/props.example.json).
- raw.glb: glTF Y up, front +Z, inside [-0.5, 0.5]^3, 4096 px PNG textures.
- asset.glb: raw.glb BIN chunk unchanged + root node matrix: longest side = size_m (metres), stable rest pose, base centre at origin.
- object.json (public, ships with asset.glb; no local paths): `format`, `id`, `noun`/`description`/`material` (if in props), `size_m` + `size_rule`, `bbox_m` (glTF x,y,z), `height_m`, `frame` ("glTF: Y up, front +Z, origin = base centre"), `pose` (rule, rule_source, tilt_deg, yaw_deg), `seed`, `source_image_md5`, `gpu_model`, `settings` (profile, pipeline_type, texture_size, face_target, export_note), `pins` (TRELLIS.2 code and weights, weight SHA-256 first 12 hex, BiRefNet, CuMesh, trianglebvh, uvrast, remesh), `outputs_md5`, `licences`. Fields from record.json are null if it is missing.
- asset.json: internal post record with the checks (`checks.passed`).
- record.json: pins (incl. `bvh`: trianglebvh version, commit, library SHA-256), settings, times, memory peaks, GPU, local paths (strip before publishing).

## Setup
1. `cp config.example.env config.env`, edit, `source config.env`.
2. `setsid -w bash trellis2/setup.sh env` (fresh venv only, pinned clones, CuMesh patch, CUDA builds, trianglebvh with cargo from `$TRIANGLEBVH_GIT`, default https://github.com/memgrafter/trianglebvh.git, pinned commit checked).
3. Accept the DINOv3 License on Hugging Face; `hf auth login`.
4. `setsid -w bash trellis2/setup.sh weights` (pinned, SHA-256 checked, 13.6 GB copy).
5. Optional: `bash imagegen/setup.sh all`. Tests: see CONTRIBUTING.md.

## Env
Required: `T2_VENV T2_SRC T2_WEIGHTS HF_HOME CUDA_HOME`. Optional: `TRIANGLEBVH_GIT CARGO TORCH_CUDA_ARCH_LIST MAX_JOBS UV IMAGEGEN_VENV IMAGEGEN_HOME BLENDER BASISU BUN`. Guards (off unless used): `GPU_GUARD_EXPECT_POWER_W`, `tools/gpu_guard.sh N "<who>" cmd`, `tools/ram_guard.sh cmd`.

## Memory
24 GB tested (RTX 3090, 3090 Ti). Generation peaks up to 22.1 GB; some objects OOM even on 24 GB. Export peak <= 12.4 GiB on the measured meshes with the lean remesh (default). 16 GB or less: docs/LOW_VRAM.md. RAM: 3.6-6.4 GB per runner plus 13 GB shared mapped weights.

## Determinism
- Same GPU model, pins, image bytes, seed and settings -> identical cut.png, mesh, raw.glb, asset.glb.
- Other GPU model -> other bytes.
- The export OOM retry changes bytes (`record.json` `export_note`).
- Draft (512) vs clean (1024_cascade), same seed: identical cond_512, RNG state, sparse-structure coords and 512 shape latent; then clean upsamples, resamples shape with new noise (1024 flow) and textures with another model. Silhouette IoU 0.989-0.995, render SSIM 0.44-0.86 (RTX 3090, 4 hard objects). Pick seeds on form from drafts; judge surface and colour on clean.
- No effect on bytes: `--remesh`, `--host-weights`, `--save-mesh`, `--hash-stages`, `--trace-decode` (6 GLBs, RTX 3090), `--chunk-norm 1048576` (decode output of 1 object, GLBs of 3; RTX 3090 and 3090 Ti, 250 W), `--export-from` (2 GLBs, RTX 3090), `--profile clean`, warm runs, guards.

## Licence guards
run.py stops on: nvdiffrast importable, RMBG-2.0 in `$HF_HOME` or loaded, BiRefNet clone not at pin, weight SHA-256 mismatch, CuMesh not the patched build or with cubvh compiled in, trianglebvh not at pin, TRELLIS.2 or CuMesh remesh code not at pin. Never add non-commercial parts (cubvh stays out: THIRD_PARTY_NOTICES.md); never read nvdiffrast, nvdiffrec, cubvh or instant-ngp source.

## Check work-identical
`bash tools/check_work_identical.sh PROPS GOLD_ROOT REF_OUT NEW_DIR [REF_MESH_DIR]` on the reference's GPU model; `python3 -I tools/compare_outputs.py A B`. See docs/WORK_IDENTICAL.md. Before a release: `bash tools/scan.sh`.
