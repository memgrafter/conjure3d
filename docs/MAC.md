# Apple Silicon (experimental)

Status: **experimental, not production.** The output is not byte-identical to the CUDA route and was not checked for
production use. Use it to try images and seeds on a Mac; make the assets you ship on the CUDA route.

The CUDA route cannot run on a Mac: TRELLIS.2's sparse convolution (FlexGEMM), CuMesh and o-voxel are CUDA only. This
route uses [trellis.cpp](https://github.com/pwilkin/trellis.cpp) (MIT, one developer), a C++/GGML port of TRELLIS.2
that runs on Metal, with our own parts around it:

1. Cut-out: the same code as the CUDA route (`run.py`'s BiRefNetCutout: pinned BiRefNet code and weights, the
   largest-piece rule), on the GPU through PyTorch MPS.
2. 3D: `trellis-cli` at the pinned commit, built with Metal, on `cut.png`, with GGUF files converted from our pinned
   weights (`mac/convert_weights.py`). trellis.cpp keeps the cut-out's alpha; its own background remover is not used.
3. Post step: `post/post_step.py`, unchanged (real size, rest pose, `object.json`).

## Setup

```sh
cp config.example.env config.env   # set the Apple block, T2_SRC, T2_WEIGHTS, HF_HOME
source config.env
python3 -I tools/preflight.py      # prints the Apple route checks
bash mac/setup_mac.sh build        # trellis.cpp at its pinned commit, Metal, PNG textures
bash mac/setup_mac.sh env          # $MAC_VENV and the pinned BiRefNet code
# weights: the same pinned files as Linux in $T2_WEIGHTS, and BiRefNet in $HF_HOME (see below)
bash mac/setup_mac.sh gguf         # checks every SHA-256, converts to $TRELLIS_GGUF
python3 -I conjure3d.py IMAGE SIZE_M --backend trellis-cpp --profile draft
```

- Needs: macOS on Apple Silicon, Xcode (its macOS SDK), cmake, git, uv. No sudo.
- Weights: `T2_VENV=$MAC_VENV bash trellis2/setup.sh weights` downloads them as on Linux (not tested on a Mac). The
  test machine got a checked copy of `$T2_WEIGHTS` from another machine instead; `setup_mac.sh gguf` checks every
  SHA-256 against `run.py`'s pins either way.
- DINOv3: its licence applies to the GGUF copy too. `DINOv3-LICENSE.md` is copied next to the GGUF files, and the run
  stops without it.
- Disk: GGUF files 16 GB, `$MAC_VENV` 0.9 GB, trellis.cpp source and build 0.3 GB.

## Profiles

| Profile | trellis.cpp | Textures | Faces |
|---|---|---|---|
| draft | `--res 512` | 1024 px | 150,000 (fixed by trellis.cpp) |
| clean | `--res 1024` (512 -> 1024 cascade) | 4096 px | 300,000 (fixed by trellis.cpp) |

The CUDA profiles use 100k (draft) and 500k (clean) faces; trellis.cpp has no option for the face count.

## Measured

Apple M6, 24 GiB unified memory, macOS 27, trellis.cpp `c0bed38c` with Metal, 2026-11-01. The 8 objects of the
image-to-3D bake-off, draft (`--res 512`), the cut-outs of the CUDA route (RTX 3090 Ti) as input, one object at a
time, nothing else heavy running (one object had another program's Blender for 35 s of its run). Wall time is the
whole `trellis-cli` process, model loading from an external SSD included. Peak memory: the process's maximum resident
set size (`/usr/bin/time -l`); the GPU shares this memory. Free memory: the lowest `memory_pressure` free percentage
of the whole machine during the run (other apps included).

| Object | Wall time | Peak memory | Lowest free memory | GLB | Faces | Texture |
|---|---|---|---|---|---|---|
| fruit cart | 413.7 s | 4.05 GiB | 56% | 9.4 MB | 147,904 | 1024 px |
| dragon mask | 389.6 s | 3.77 GiB | 57% | 10.5 MB | 145,382 | 1024 px |
| carousel horse | 202.1 s | 2.65 GiB | 51% | 9.3 MB | 145,700 | 1024 px |
| panel chest | 476.4 s | 4.58 GiB | 52% | 9.1 MB | 139,882 | 1024 px |
| frost wolf | 203.7 s | 2.68 GiB | 54% | 8.5 MB | 149,708 | 1024 px |
| gloom stalker | 228.5 s | 3.13 GiB | 54% | 9.4 MB | 146,870 | 1024 px |
| mimic | 803.2 s | 5.30 GiB | 39% | 9.9 MB | 141,068 | 1024 px |
| cuckoo clock | 234.4 s | 2.74 GiB | 49% | 10.5 MB | 148,162 | 1024 px |

- Median 312 s per object (202-803 s). The mimic decoded to an 11.9M-face mesh before simplification; the CPU parts
  of the post-processing (remesh, QEM simplification, bake) grow with that.
- The same cut-out step on MPS: 2.6 s per image after loading.
- These runs used `--seed 0`, which trellis.cpp turns into a random seed (each log records the seed it used); the
  route now stops on seed 0. Seeds never give the same shape as the CUDA route: other random number code.
- Clean (`--res 1024`), fruit cart only, seed 1: 3439 s (shape flow 1754 s, texture flow 1056 s), peak memory
  9.64 GiB (peak footprint 10.34 GiB), lowest free memory 28% (21% while another program's Blender ran), GLB 47.3 MB,
  279,032 faces, 4096 px. The CUDA route on an RTX 3090 takes 215 s for the same object (clean), plus a 55 s model
  load. Four earlier attempts were stopped when another program's Blender started and free memory fell below 20%.

Bake-off S5 score of the same objects (silhouette IoU from the fitted gold camera, higher is better / mean colour
difference, lower is better; one seed each, and the seeds differ between routes):

| Object | CUDA route, clean | trellis.cpp 512 | trellis.cpp 1024 |
|---|---|---|---|
| fruit cart | 0.883 / 0.145 | 0.861 / 0.148 | 0.871 / 0.151 |
| dragon mask | 0.903 / 0.200 | 0.908 / 0.195 | |
| carousel horse | 0.884 / 0.125 | 0.857 / 0.148 | |
| panel chest | 0.966 / 0.144 | 0.965 / 0.155 | |
| frost wolf | 0.896 / 0.132 | 0.870 / 0.138 | |
| gloom stalker | 0.826 / 0.100 | 0.830 / 0.146 | |
| mimic | not scored | 0.948 / 0.148 | |
| cuckoo clock | not scored | 0.907 / 0.160 | |

In 384 px review renders the Mac outputs look like the CUDA route's: the same parts, no holes or wrong colours. Fine
detail at full size was not compared.

## Differences from the CUDA route

- Other code for every 3D stage: the flows and decoders in GGML, weights stored as f16 (bf16 weights lose range) and
  f32, its own remesh (narrow-band dual contouring), a port of CuMesh's QEM simplifier (on the CPU on Apple), stock
  xatlas, its own texture bake with a BVH snap, and a port of OpenCV's Telea inpaint for the texture gutter.
- Same cut-out code and weights; on MPS the cut-out matched an RTX 3090 Ti's (alpha IoU 1.0, at most 1/255 different on
  0.0001% of the pixels, 1 object).
- GLB: the same frame (glTF Y up, front +Z, inside [-0.5, 0.5]^3), PNG textures, `alphaMode` OPAQUE.
- `record.json` has `backend: trellis-cpp`; `object.json` names the route, its pins (trellis.cpp commit, GGUF SHA-256)
  and its licences.

## Licences

trellis.cpp and the parts it vendors are MIT (ggml, xatlas, meshoptimizer, Fast-Quadric-Mesh-Simplification,
cpp-httplib) or public domain/MIT (stb). Its post-processing was read at the pinned commit for non-commercial parts:
none found. Its BVH (`src/tri_bvh.cpp`) is a median-split tree with Ericson's closest-point test; its notes say the
reference's cubvh was not vendored and only CuMesh's Python interface to it was read. The UV raster that replaces
nvdiffrast is its own CPU code. Not built here: its optional quad retopology (Linux and Windows only), which would
fetch CGAL (GPL/LGPL), Boost, LEMON, GMP and MPFR (LGPL).
