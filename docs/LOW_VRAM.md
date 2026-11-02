# Running on 16 GB GPUs (and smaller)

The pipeline was built and tested on 24 GB cards (RTX 3090 and RTX 3090 Ti). This page says where the GPU memory goes,
what has been measured with a 16 GB and an 8 GB limit, and where to start on a smaller card. Values marked "not
measured yet" come later; nothing on this page is an estimate.

Units: GiB (1024^3 bytes), as CUDA reports them. A "16 GB" card such as an RTX 4080 SUPER shows about 15.70 GiB to
CUDA under Linux; the desktop and other programs take part of that.

## Where the memory goes

One object goes through two GPU-heavy stages in `trellis2/run.py`:

1. **Generate** (TRELLIS.2 `pipeline.run`, 1024_cascade): the flow models run one at a time (low_vram mode moves each
   model to the GPU only for its stage), then the decoders. The peak depends on the object: detailed shapes decode to
   more voxels and faces.
2. **Export** (`o_voxel.postprocess.to_glb`): CuMesh fills holes, trianglebvh builds a BVH (on the CPU, no GPU memory), remeshes at resolution 1024 (narrow
   band, dual contouring), simplifies to 500k faces, unwraps UVs, then uvrast bakes the 4096 px textures.

The cut-out (BiRefNet) is small next to these and its weights leave the GPU after each image.

### Generate: measured peaks

| Hardware | Objects | Device-wide peak in generation |
|---|---|---|
| RTX 3090 Ti, 250 W, 2026-10-31 | 6 | 3.1 to 22.1 GB (22.1 GB: a fur pelt, 35.1M decoded faces; 20.4 GB: a coiled rope, 27.5M) |
| RTX 3090 x3 + 3090 Ti, 275 W, 2026-11-01 | 99 | over 15.7 GiB for 7 of 99 objects; torch's allocated peak over 15.7 GiB for 2 of 99 |

- The device-wide peak includes torch's cache of freed blocks, so it is higher than what the stage strictly needs; the
  allocated peak is lower than what it needs (allocator overhead). The real need is between the two.
- One object (a silk skein) ran out of memory in generation on a 24 GB RTX 3090 Ti: torch had 22.15 GiB allocated and
  asked for 2.81 GiB more. So some objects do not fit even on 24 GB with 1024_cascade.
- **On a 16 GB card, generation is the stage that can still fail on very detailed objects**, also with the lean remesh.
  `--chunk-norm` lowers that peak with the same bytes (below).

### Export: measured peaks, upstream remesh vs lean remesh

RTX 3090, 2026-11-01, `to_glb` with the pipeline's settings on saved decoded meshes; device peak sampled every 1 ms
with NVML in a separate process (GiB). The remesh is the peak of the whole export with CuMesh's own function:

| Decoded mesh (faces, after the 16.7M cap) | Remeshed faces | Remesh, upstream | Remesh, lean | Simplify | Bake |
|---|---|---|---|---|---|
| glass jar of brine, 25.0M (16.0M) | 47.4M | 19.56 | 12.41 | 11.14 | 7.38 |
| glowing jar, 23.3M (15.6M) | 43.5M | 19.08 | 12.19 | 10.64 | 8.33 |
| coiled rope, 27.5M (16.5M) | 32.0M | 15.08 | 9.80 | 8.50 | 7.06 |
| fur pelt, 35.1M (16.4M) | 20.6M | 11.69 | not measured | 6.30 | 7.68 |

- **The lean remesh is the main fix on the export side.** `trellis2/cumesh_lean_remesh.py` is CuMesh's
  `remesh_narrow_band_dc` with `del` of each intermediate after its last use; it gives byte-identical GLBs (checked on
  three meshes above) and is the default (`--remesh lean`). With it, the export peaks of these meshes are 12.4 GiB or
  less, and the next limit is CuMesh's simplify (about 155 bytes per remeshed face).
- Validation run, 2026-11-01 (RTX 3090 and RTX 3090 Ti at 275 W; 11 of the hardest objects of a 99-object batch):
  with the lean remesh the export peaked at 9.2-11.8 GiB. Two examples, upstream -> lean: 23.37 -> 11.84 GiB and
  21.98 -> 11.25 GiB. With the lean remesh, generation (`decode_latent`) is the device peak: 16.5-23.6 GiB in use,
  torch's cache included. 26 GLBs of 15 objects were byte-identical to the baselines on both cards, with lean and
  with upstream.
- The export peak grows with the number of remeshed faces (the surface at resolution 1024). The decoded face count
  does not predict it: the fur pelt has the most decoded faces and the smallest export peak.
- `run.py` empties torch's cache before every CuMesh call (CuMesh allocates GPU memory outside torch's cache). Without
  that, the export of the brine jar ran out of memory on 24 GB.
- The peaks above were measured with CuMesh's cubvh as the BVH. With trianglebvh (CPU) the export peaked at 8.0-9.8 GB
  against 9.4-10.7 GB with cubvh (5 objects, one run each, RTX 3090 at 250 W, same GLB bytes).
- To measure or run the export alone, save decoded meshes with `--save-mesh DIR`, then run
  `run.py --export-from DIR/<id>.pt ... --export-out OUT` (no cut-out, no generation, no TRELLIS.2 weights; the
  export uses the profile's texture size and face target, and `--vram-cap-gb` if given). Two saved meshes exported
  alone gave GLBs byte-identical to their full runs (RTX 3090, 250 W, 2026-11-01).
- If the export still runs out of memory, `run.py` retries once on the decoded mesh simplified to 4M faces. That GLB
  differs from a first-try export, and `record.json` says so (`export_note`). Use it as a safety net only.

## Measured with a 16 GB limit

`run.py --vram-cap-gb 15.70` makes a 24 GB card behave like a 16 GB one: before anything loads, one raw `cudaMalloc`
holds the rest of the card for the whole process, so torch and CuMesh see only 15.70 GiB. It is an
approximation: the same kernels and allocator, but not the smaller card's speed, driver reserve or display use. It
needs the GPU to itself.

Hard objects (the most decoded faces) from a 99-object batch, 2026-11-01. RTX 3090 and RTX 3090 Ti; each object
ran second in its process (warm); model load not included. Power limit per row.

| Arm | Cap | Card, power limit | Objects tried | Passed | Failed in generate | Failed in export | Time per passing object |
|---|---|---|---|---|---|---|---|
| clean (1024_cascade, 4096 px, 500k faces), lean remesh | 15.70 GiB | 3090 (4) and 3090 Ti (6), 275 W | 10 | 6, byte-identical to the same objects without a cap | 4, all in `decode_latent` | 0 | 203.0-244.4 s (without a cap: 205.2-251.2 s) |
| clean + `--chunk-norm 1048576`, lean remesh | 15.70 GiB | 3090 (2) and 3090 Ti (2), 250 W | 4: the 4 that failed above | 3, byte-identical to the same objects without a cap and without `--chunk-norm` | 1 (wicker hamper), in `decode_latent` | 0 | 263.7-322.7 s |
| clean, upstream remesh | 15.70 GiB | | not measured yet | | | | |
| 512, 4096 px, 500k faces, lean remesh | 15.70 GiB | 3090 (2) and 3090 Ti (6), 275 W | 8 | 8 | 0 | 0 | 48.0-74.6 s |
| draft (512, 1024 px, 100k faces), lean remesh | 7.63 GiB | 3090 Ti, 250 W | 6 | 6 | 0 | 0 | 31.2-43.2 s |

- Why clean fails under 16 GB: at the failure torch had 13.9-14.1 GiB allocated and only 0.2 GiB cached but unused,
  and asked for 1.7-1.8 GiB more. The shape decoder's last up block works on every voxel at 1024 at once (14.5M
  voxels on one object, 64 channels); one such fp16 tensor is 1.73 GiB. The memory is in use, not held by the cache.
- `--chunk-norm ROWS` (off by default) runs the decoders' LayerNorm32 on ROWS rows at a time, so its full-size fp32
  copies are not made at once. The norm works per row, so the bytes do not change: with 1048576 rows, one object's
  decode output and GLB were identical to the run without it (RTX 3090, no cap), and so were the GLBs of the 3
  objects that passed under the cap. It lowered the last up block's torch peak by 0.96 GiB in each decoder on that
  object; the block's other large tensors stay. **Clean under the 15.70 GiB cap: 9 of 11 hard objects with
  `--chunk-norm`, 6 of 11 without.** The wicker hamper (20.1M voxels at 1024) still ran out of memory in
  `decode_latent` (it asked for 2.02 GiB more), and so did the capstan (0.90 GiB more).
- Peaks in the passing runs, generation / export: clean under the cap 15.43-15.69 / 9.25-11.64 GiB; 512 at 4096 px
  4.72-7.05 / 4.28-5.68 GiB; draft under the 7.63 GiB cap 4.72-7.05 / 2.75-3.32 GiB.
- 7.63 GiB is what an 8 GB card such as an RTX 4060 or RTX 3070 shows to CUDA under Linux (7.63-7.79 GiB).

What 512 loses against 1024_cascade (same objects and seed; the 512 arm at 4096 px, 500k faces; 4 views rendered at
1024 px and 384 px with the camera fitted to the 1024_cascade object):

- The shape outline stays: silhouette IoU 0.989-0.995.
- The surface and colour are drawn again: SSIM 0.44-0.91 at 1024 px and 0.51-0.92 at 384 px; mean colour difference
  per pixel 0.04-0.10.
- It shows in the web export too (200k-triangle `web.glb`): SSIM 0.49-0.89 at 1024 px and 0.51-0.85 at 384 px. For
  scale, a 1024_cascade object's own web export against its full GLB: 0.89-0.98 and 0.94-0.99.
- Same seed: the coarse structure and the 512 shape are byte-identical in the two pipelines (`--hash-stages`), so
  512 is good to pick a form or a seed, not to judge the surface.

## Starting settings for 16 GB

Tested only on a 24 GB card with the 15.70 GiB cap (table above). No real 16 GB card was tested.

```sh
HF_HUB_OFFLINE=1 ATTN_BACKEND=xformers PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 \
  $T2_VENV/bin/python -I trellis2/run.py --job props.json:images:out --chunk-norm 1048576   # 1024_cascade, --remesh lean
```

- Keep the defaults: `--remesh lean`, `--offload persistent`, low_vram mode (TRELLIS.2's default). Add
  `--chunk-norm 1048576`: same bytes, lower decode peak (`conjure3d.py --chunk-norm` does the same).
- Nothing else may use the GPU: no image stage, no desktop session on that card if you can avoid it. Run the image
  stage (FLUX.2-klein-4B, 15.9 GB resident on the GPU) in its own process, before or after, never next to the 3D stage.
- An object that runs out of memory in generation writes `out/<id>/oom-1024_cascade.json` and the run goes on with the
  next object. With the 15.70 GiB cap, 5 of 11 hard objects did without `--chunk-norm` and 2 of 11 with it (table
  above). Run those objects again with `--profile draft` (fewer details, other bytes).

## Under 16 GB

No real card under 16 GB was tested. With a 7.63 GiB cap (an 8 GB card's CUDA total) on an RTX 3090 Ti, the draft
profile made all 6 hard objects tried (table above). A starting point:

1. `--profile draft` (the 512 pipeline, 1024 px textures, 100k faces).
2. One object per process (`--only ID`), so no cached memory from an earlier object is left.
3. Run each stage in its own process: the image stage, the 3D stage, the post step and the web export (the post
   step and the web export use no GPU).
4. Check `out/<id>/oom-*.json` and `stages.jsonl` to see which stage ran out of memory.

`--vram-cap-gb` can try a smaller limit on a bigger card first (for example 11.5 or 7.5), with the same caveats as
above. Different settings give different files: outputs made with them are not byte-identical to the defaults.
