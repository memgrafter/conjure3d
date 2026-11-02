# conjure3d

This repository makes 3D objects from images. Give it a picture of one object and the object's real size, and you get a
textured GLB at that size, standing in a stable rest pose. It can also make the pictures: an optional stage prompts
an image model (FLUX.2-klein-4B) for reference images.

## Who it is for

You need many 3D props from images for a game or a web scene, and you want to use them commercially. You
have a Linux machine with an NVIDIA GPU (24 GB tested; a guide for 16 GB is in [docs/LOW_VRAM.md](docs/LOW_VRAM.md)).

## What is different

- **Licences that allow commercial use, end to end.** TRELLIS.2 runs with a BiRefNet cut-out (MIT) instead of
  RMBG-2.0 (non-commercial), with uvrast (MIT) instead of nvdiffrast (non-commercial) for the texture bake, and
  with trianglebvh (MIT) instead of CuMesh's cubvh (non-commercial) for the export's BVH; cubvh is never downloaded.
  The runner stops if an excluded part is installed.
- **Byte-identical output.** On the same GPU model with the same pins, the same image and seed give the same bytes
  for the cut-out, the mesh and the GLB. A patch to CuMesh removes its run-to-run changes. You can make an asset again
  later and compare files ([docs/WORK_IDENTICAL.md](docs/WORK_IDENTICAL.md)).
- **Real scale and pose.** The post step sets the longest side to your `size_m`, puts the object on a stable face if
  it would tip over, and sets the origin at the base centre. It changes one node matrix; the mesh and texture bytes
  stay as they are. `object.json` next to `asset.glb` records the size, frame, pose, seed, settings, pinned versions,
  file MD5s and licences, with no local paths, so it can ship with the GLB.
- **Web export.** An optional step makes a 200k-triangle GLB with KTX2 textures and a 1k-triangle proxy for the web.
- **Lower memory.** A memory-lean copy of CuMesh's remesh (same bytes) lowers the export's GPU peak by about 7 GiB on
  large meshes, and the weights are memory-mapped, so several runners on one machine share one copy in RAM.

## Generation profiles

A profile sets the whole recipe: mesh resolution, texture size and face count. Mesh resolution sets the GPU memory you
need and most of the time per object. Use draft to explore shapes and seeds, clean for assets you ship, best for
close-ups.

| Profile | Mesh resolution | Textures | Faces | GPU memory, peak seen | Fits |
|---|---|---|---|---|---|
| draft | 512 | 1024 px | 100k | 4.7-7.1 GiB (hard objects) | 16 GB: 8 of 8 (512 mesh, 4096 px). 8 GB: 6 of 6 |
| clean (default) | 1024 (`1024_cascade`) | 4096 px | 500k | 16.5-23.6 GiB (11 hard objects) | 24 GB: 11 of 11. 16 GB: 6 of 10 hard objects, 9 of 10 with `--chunk-norm` |
| best | 1536 (`1536_cascade`) | 4096 px | 500k | up to 23.3 GiB | see below |

Hard objects on an RTX 3090 / 3090 Ti at 275 W (warm process, model load of 33-64 s not included):

| | 512 mesh, 4096 px, 500k faces | clean |
|---|---|---|
| Time per object | 48-75 s | 157-296 s |
| GPU energy per object | 2.6-3.9 Wh | 10-21 Wh |

- The saving grows with how detailed the object is. A simple object saves much less (about 1.2-1.7x in first
  runs).
- The 16 GB failures happen in the decoders, which work on every voxel at 1024 at once. That memory is in use,
  not only held by the cache.
- `--chunk-norm` (`run.py --chunk-norm 1048576` or `conjure3d.py --chunk-norm`) runs the decoders' normalization on
  1048576 rows at a time. It gives the same bytes and a lower decode peak. With it, clean made 9 of the 10 hard
  objects under the 16 GB cap (6 without); a wicker hamper still ran out of memory (RTX 3090 and 3090 Ti, 250 W).
  It is off by default.
- To run only the GLB export again, save the decoded meshes with `run.py --save-mesh DIR`, then run
  `run.py --export-from DIR/<id>.pt --export-out OUT`. The GLB is the same as the full run's (2 objects, RTX 3090).
- Best: TRELLIS.2 lowers the resolution when an object needs more than 49,152 tokens. On the hard objects tested,
  best fell back to 1024 (same bytes as clean), ran at 1152 (a different result, 289 s against 219 s), or ran out
  of memory at 24 GB. It gave no gain on those objects. Simpler objects are not tested yet.
- With the same seed, draft and clean share the same coarse structure and the same 512 shape (checked byte for
  byte), so the outline matches (silhouette overlap 0.989-0.995). Clean then draws the shape detail and the texture
  again, so the surface differs: render SSIM 0.44-0.91 against clean. The difference shows in the web export too.
  Use drafts to pick form and seeds, not to judge surface or colour.
- The 512 column above used 4096 px textures and 500k faces. Draft itself (1024 px, 100k faces), RTX 3090 Ti at
  250 W, 6 hard objects: 31.2-43.2 s and 1.96-2.79 Wh per object. Its GLB export took 5.8-7.9 s, against
  19.5-35.4 s for the same objects at 4096 px and 500k faces (at 275 W).
- "16 GB" and "8 GB" in the table: runs on 24 GB cards with the GPU memory capped at 15.70 GiB and 7.63 GiB
  (`--vram-cap-gb`), what such cards show to CUDA under Linux. No real 16 GB or 8 GB card was tested. Details:
  [docs/LOW_VRAM.md](docs/LOW_VRAM.md).
- Use it: `run.py --profile draft|clean|best` or `conjure3d.py --profile ...`. Clean is the default and gives the
  same bytes as before profiles existed. `--pipeline-type`, `--texture-size` and `--decimation-target` override one
  value of a profile; `record.json` lists the profile and what was overridden.

## Limits

- One object per image. If the cut-out finds several separate pieces, it keeps the largest one.
- TRELLIS.2 makes its own proportions and back side; the result can differ from the picture. Different seeds give
  very different shapes, so make several and pick one. It does not pick images or seeds for you.
- Glass, water and flames come out poorly (in tests: a lamp's flame as a flat card, the water in a glass jar nearly
  invisible). Fur can break into thousands of loose pieces. Meshes are not watertight.
- Bytes match only on the same GPU model. An RTX 3090 and an RTX 3090 Ti give different files from the same input.
- Linux and NVIDIA only (CUDA 13 build). Only one machine was tested; other GPUs, drivers and CUDA versions are
  untested. Some very detailed objects run out of memory even on 24 GB.

## Measured

Numbers come from logged runs on the hardware named.

99 objects in one batch (2026-11-01; 3 x RTX 3090 and 1 x RTX 3090 Ti at 275 W, four runners on one 31 GB machine
sharing one queue; 1024_cascade, 4096 px textures, seed 42):

| GPU | Objects | Per object, median (min-max) | Generate, median | GLB export, median | Model load |
|---|---|---|---|---|---|
| RTX 3090 | 82 | 108.0 s (47.8-265.7) | 57.7 s | 37.1 s | 39.9-60.8 s |
| RTX 3090 Ti | 17 | 106.9 s (51.1-221.7) | 62.2 s | 33.4 s | 47.8 s |

- These runs used CuMesh's cubvh for the BVH. trianglebvh, which replaces it, runs on the CPU: in 5 pairs on an RTX
  3090 at 250 W (same GLB bytes), its BVH queries took 9.3-10.9 s against 0.6-0.7 s, and a whole object took
  4.7-10.0 s longer.
- GPU memory in generation, RTX 3090 Ti at 250 W (6 objects, device-wide, sampled every 0.2 s): 3.1 to 22.1 GB.
  Export peaks and the lean remesh: [docs/LOW_VRAM.md](docs/LOW_VRAM.md).
- Cut-out: 0.61-0.70 s per image on an RTX 3090 Ti at 250 W, after a first image of 5.9 s.
- Image stage (FLUX.2-klein-4B, RTX 3090 Ti, whole pipeline on the GPU, 15.9 GB): 6.1 s per 1088x1920 image from
  text (n = 1,134), about 10 s with one reference image.

## Licence

The code here is MIT ([LICENSE](LICENSE)). It ships no weights and no third-party binaries. `setup.sh` downloads the
parts below at pinned versions under their own licences ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)):

- MIT: TRELLIS.2 code and weights, the TRELLIS-image-large decoder, BiRefNet code and weights, CuMesh, FlexGEMM,
  o-voxel, utils3d, uvrast, trianglebvh. Apache-2.0: FLUX.2-klein-4B.
- DINOv3 License (the image features TRELLIS.2 uses): gated on Hugging Face. Accept it with your own account before
  setup. It has conditions, for example no use subject to ITAR and no military or weapons use.
- Blender (optional renders and the proxy bake) is a GPL program that the scripts run as a separate program.
- Excluded: nvdiffrast and nvdiffrec, cubvh (CuMesh's BVH submodule), briaai/RMBG-2.0, FLUX.2-klein-9B, Qwen-Image.

## First object

These steps make one object on your own machine. You need Linux, an NVIDIA GPU, and the disk space in the table
below. Run the commands in the folder of this repository.

1. Set the paths and check the machine. Each line of the check starts with PASS, WARN or FAIL and tells you what to
   do. Correct each FAIL before you continue.

   ```sh
   cp config.example.env config.env      # then write your five paths in config.env
   source config.env
   python3 -I tools/preflight.py
   ```

2. Make the 3D environment. This compiles CUDA code and one Rust library, so it needs the CUDA 13 toolkit, g++,
   uv and Rust (cargo). To compile nothing on this machine, use the container: [docs/CONTAINER.md](docs/CONTAINER.md).

   ```sh
   setsid -w bash trellis2/setup.sh env
   ```

3. Get the weights. First, open https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m with your own
   Hugging Face account. Read the DINOv3 License and accept it if you agree. Approval can take some time. Then:

   ```sh
   $T2_VENV/bin/hf auth login
   setsid -w bash trellis2/setup.sh weights
   python3 -I tools/preflight.py         # no FAIL lines now
   ```

4. Make the object. Use a picture of one whole object on a plain background. Give the length of its longest side
   in metres.

   ```sh
   python3 -I conjure3d.py my-lantern.png 0.35
   ```

   The command prints the path of the GLB, for example `out/my-lantern-clean-seed42/my-lantern/asset.glb`. Other
   seeds give other shapes: add `--seed 7`. For an object that lies flat, such as a watch, add `--lie-flat`. All
   options: `python3 -I conjure3d.py --help`.

5. Look at the object. Open `asset.glb` in Blender (File > Import > glTF 2.0) or in another glTF 2.0 viewer. Then
   look at `cut.png` in the same folder. It must show the whole object and no background. If not, use another
   picture.

If the command says OUT OF GPU MEMORY, the object needs more GPU memory than your GPU has free. Close other programs
that use the GPU, or read [docs/LOW_VRAM.md](docs/LOW_VRAM.md). If generation ran out of memory, first try
`--chunk-norm`: it gives the same files with a lower GPU peak. The draft profile (`--profile draft`) uses less GPU
memory and gives a rougher surface. The preflight check tells you if its model files are installed.

### Disk, downloads and time

| Step | Disk | Download | Time |
|---|---|---|---|
| `setup.sh env` | venv 5.9 GB, sources 1.3 GB, uv's cache (not measured) | Python packages and pinned sources (not measured) | 23 min, measured once (below) |
| `setup.sh weights` | 16.6 GB Hugging Face cache, 13.6 GB weight copy | 16.6 GB | not measured |
| Container build | not measured | CUDA base image 3.6 GB (compressed) and the packages | not measured |
| One object | 93 MB for the test object (two 46 MB GLBs) | none | 148 s for the 3D step and 8 s for the post step (below) |

- The 23 minutes: one run of `setup.sh env` on an Intel Core i7-14700KF with `MAX_JOBS=2`, the venv on a network
  share, and uv's cache already holding torch. The three CUDA builds took 14 minutes of it.
- One object: `conjure3d.py` with the clean profile, one simple object (a burning torch prop, 1.1M decoded
  faces), RTX 3090 Ti at 250 W, 2026-11-01. The 148 s include the model load (37 s). Its files were
  byte-identical to an earlier run of the same object. Detailed objects take longer: 47.8-265.7 s per object in
  the batch above, plus the model load.

## Start

Setup, commands, inputs and outputs: [AGENTS.md](AGENTS.md). Using it from an agent session: [SKILL.md](SKILL.md).

```sh
cp config.example.env config.env && source config.env      # edit the paths first
setsid -w bash trellis2/setup.sh all                       # after you accept the DINOv3 License and log in to Hugging Face
HF_HUB_OFFLINE=1 ATTN_BACKEND=xformers PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $T2_VENV/bin/python -I trellis2/run.py --job examples/props.example.json:images:out
$T2_VENV/bin/python -I post/post_step.py --props examples/props.example.json --dir out
```

[CONTRIBUTING.md](CONTRIBUTING.md) · [SECURITY.md](SECURITY.md) · [SECURITY_SCAN.md](SECURITY_SCAN.md)
