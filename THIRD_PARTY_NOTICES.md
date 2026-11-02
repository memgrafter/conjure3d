# Third-party notices

This repository's own code is under the MIT License ([LICENSE](LICENSE)). It ships no third-party weights and no
third-party binaries. `setup.sh` downloads the parts below at pinned versions; each keeps its own licence, and you
accept that licence when you download and use it. This list does not give legal advice. Check the licences
yourself before commercial use.

Each entry was checked against the source named in "Checked in" (the pinned commit's or revision's own licence file
or metadata, or the project's licence page) on 2026-11-01.

## Models (downloaded by `setup.sh weights`, never shipped here)

| Part | Pin | Licence | Conditions | Checked in |
|---|---|---|---|---|
| TRELLIS.2-4B weights (microsoft/TRELLIS.2-4B) | `af44b45f` | MIT | keep the copyright and licence notice | model card metadata `license: mit` at the pinned revision |
| TRELLIS-image-large sparse-structure decoder (microsoft/TRELLIS-image-large) | `25e0d31f` | MIT | as above | model card metadata at the pinned revision |
| DINOv3 ViT-L/16 (facebook/dinov3-vitl16-pretrain-lvd1689m) | `ea8dc286` | DINOv3 License (Meta, last updated 2025-08-19) | see below | `LICENSE.md` at the pinned revision |
| BiRefNet general weights (ZhengPeng7/BiRefNet `model.safetensors`) | `e2bf8e44`, SHA-256 `9ab37426...a154` | MIT | keep the notice | model card metadata `license: mit` at the pinned revision |
| FLUX.2-klein-4B (black-forest-labs/FLUX.2-klein-4B; optional image stage; includes its Qwen3 text encoder) | `e7b7dc27` | Apache-2.0 | keep the licence and notices; state changes if you redistribute modified files | `LICENSE.md` at the pinned revision |

**DINOv3 License.** The weights are gated on Hugging Face: each user must accept the licence with their own account;
this repository never accepts it for you. Conditions in the licence text (section 1.b): if you distribute the DINO
Materials or derivative works, do it under the same agreement and include a copy of it; acknowledge DINO Materials in
published research that uses them; comply with applicable laws, including trade control and privacy laws; do not
reverse engineer, decompile or discover the underlying components; you must not be a target of trade controls; no
use for activities subject to ITAR or end uses prohibited by trade controls, including military or warfare purposes,
nuclear industries or applications, espionage, or the development or use of guns or illegal weapons. Also: no
warranty; your licence ends if you sue Meta over the materials; Meta may modify the agreement; California law. There
is no revenue limit in the licence text.

## Code downloaded and built by `setup.sh env`

| Part | Pin | Licence | Checked in |
|---|---|---|---|
| TRELLIS.2 (microsoft/TRELLIS.2), including o-voxel | `75fbf018` | MIT, Copyright (c) Microsoft Corporation | `LICENSE` in the pinned clone |
| Eigen (o-voxel third_party, header-only) | TRELLIS.2's submodule commit | primarily MPL-2.0; some files carry BSD or other MPL-2.0-compatible licences (Eigen's COPYING.README) | `COPYING.*` in the clones |
| CuMesh (JeffreyXiang/CuMesh) + `trellis2/patches/cumesh-deterministic.patch` and `cumesh-no-cubvh.patch` (ours, MIT); built without its cubvh submodule (below) | `12289e10` | MIT, Copyright (c) 2025 Jianfeng XIANG | `LICENSE` in the pinned clone |
| `trellis2/cumesh_lean_remesh.py`: a copy of CuMesh's `remesh_narrow_band_dc` with `del` statements added | CuMesh `12289e10` | MIT, Copyright (c) 2025 Jianfeng XIANG | as above |
| xatlas (CuMesh third_party) | CuMesh's copy | MIT, Copyright (c) 2018-2020 Jonathan Young | `third_party/xatlas/LICENSE` |
| FlexGEMM (JeffreyXiang/FlexGEMM) | `6dd94a85` | MIT, Copyright (c) 2025 Jianfeng Xiang | `LICENSE` in the pinned clone |
| BiRefNet code (ZhengPeng7/BiRefNet) | `f7079632` | MIT, Copyright (c) 2024 ZhengPeng | `LICENSE` in the pinned clone |
| uvrast (memgrafter/uvrast) | `5dcc54a3` | MIT | its `LICENSE` |
| trianglebvh (clean-room BVH; replaces cubvh in the GLB export; built with cargo, no Rust crates) | `b8ef3415` | MIT | its `LICENSE` at the pinned commit |
| utils3d (EasternJournalist/utils3d) | `9a4eb15e` | MIT | package metadata (table below) |

### cubvh: not downloaded, not built

CuMesh's own `setup.py` compiles its submodule cubvh (github.com/JeffreyXiang/cubvh @ `ce92267a`) into CuMesh as
`cumesh._cubvh`. `setup.sh` leaves it out: it clones CuMesh without submodules, and
`trellis2/patches/cumesh-no-cubvh.patch` removes cubvh's extension from CuMesh's `setup.py` and adds a small
stand-in module that stops on any use. `run.py` stops if a compiled cubvh module is in the CuMesh package. The BVH
that TRELLIS.2's GLB export calls (the remesh's distance queries and the bake's attribute lookup) comes from
trianglebvh (MIT, clean-room) instead.

Why: the pinned cubvh has two licence files, `LICENSE` (MIT, Copyright (c) 2022 kiui) and `LICENSE_NVIDIA`, the
"NVIDIA Source Code License for instant neural graphics primitives" (section 3.3: use "only ... non-commercially",
that is "for research or evaluation purposes only"). Upstream cubvh (github.com/ashawkey/cubvh) removed that file in
commit `191b020` (https://github.com/ashawkey/cubvh/commit/191b0206, "[ROCm] Remove LICENSE_NVIDIA, no code it
covers remains"); its message says that `src/bvh.cu`, `include/gpu/bvh.cuh`, `include/gpu/triangle.cuh`,
`include/gpu/bounding_box.cuh` and part of `include/gpu/common.h` borrowed code from NVlabs/instant-ngp and were
later replaced. The pinned copy has those files, and CuMesh compiles them.

## Python packages (installed by `setup.sh`, pinned in `requirements.txt`)

| Package | Version | Licence (from the installed package's metadata) |
|---|---|---|
| certifi | 2026.10.23 | MPL-2.0 |
| charset-normalizer | 3.5.2 | MIT |
| cuda-bindings | 13.4.4 | Apache-2.0 (package metadata; it wraps NVIDIA's CUDA) |
| cuda-pathfinder | 1.8.5 | Apache-2.0 |
| cuda-toolkit | 13.0.3.0 | no licence in its metadata (NVIDIA) |
| easydict | 1.13 | **LGPL-3.0** (imported by TRELLIS.2's flow sampler; installed, not shipped) |
| einops | 0.8.2 | MIT |
| filelock | 4.2.5 | MIT |
| fsspec | 2026.9.0 | BSD-3-Clause |
| glcontext | 3.1.0 | MIT |
| hf-xet | 1.8.0 | Apache-2.0 |
| huggingface_hub | 0.36.2 | Apache-2.0 |
| idna | 3.20 | BSD-3-Clause |
| imageio | 2.38.1 | BSD-2-Clause |
| imageio-ffmpeg | 0.7.0 | BSD-2-Clause (its wheel bundles an ffmpeg binary under ffmpeg's own licence) |
| jinja2 | 3.1.6 | BSD-3-Clause |
| kornia | 0.9.0 | Apache-2.0 |
| kornia_rs | 0.2.0 | Apache-2.0 |
| markupsafe | 3.0.4 | BSD-3-Clause |
| moderngl | 5.13.0 | MIT |
| mpmath | 1.3.0 | BSD-3-Clause |
| networkx | 3.7 | BSD-3-Clause |
| ninja | 1.13.2 | Apache-2.0 |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| nvidia-* CUDA libraries (cublas, cuda-cupti, cuda-nvrtc, cuda-runtime, cudnn-cu13, cufft, cufile, curand, cusolver, cusparse, cusparselt-cu13, nccl-cu13, nvjitlink, nvshmem-cu13) | see requirements.txt | NVIDIA proprietary licence (LicenseRef-NVIDIA-Proprietary) |
| nvidia-nvtx | 13.0.85 | Apache-2.0 |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pillow | 13.0.0 | MIT-CMU |
| plyfile | 1.1.5 | **GPL-3.0-or-later** (a TRELLIS.2 dependency; imported only by `trellis2/utils/mesh_utils.py`, which this pipeline does not call; installed, not shipped) |
| pyyaml | 6.0.3 | MIT |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python |
| requests | 2.34.2 | Apache-2.0 |
| safetensors | 0.8.0 | Apache-2.0 |
| scipy | 1.18.1 | BSD-3-Clause |
| setuptools | 84.2.1 | MIT |
| sympy | 1.14.0 | BSD-3-Clause |
| timm | 1.0.30 | Apache-2.0 |
| tokenizers | 0.22.2 | Apache-2.0 |
| torch | 2.14.1+cu130 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| torchvision | 0.29.1+cu130 | BSD-3-Clause |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| transformers | 4.57.6 | Apache-2.0 |
| trimesh | 5.2.0 | MIT |
| triton | 3.8.0 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| urllib3 | 2.8.0 | MIT |
| utils3d | 0.0.2 (git 9a4eb15e) | MIT |
| uvrast | 0.1.0 (git 5dcc54a3) | MIT |
| wheel | 0.48.0 | MIT |
| xformers | 0.0.35 | BSD-3-Clause (metadata: BSD License) |
| zstandard | 0.25.0 | BSD-3-Clause |

Also built from source in the venv: cumesh, flex_gemm, o_voxel (above). The optional image stage's venv
(`imagegen/requirements.txt`) uses the same torch, CUDA wheels and common packages, and adds (checked the same way):
diffusers 0.42.1, accelerate 1.15.0, transformers 5.22.1, huggingface-hub 2.4.0, tokenizers 0.23.3,
importlib-metadata 10.1.0 (Apache-2.0); rich 15.0.0, typer 0.27.3, anyio 4.15.1, h11 0.16.0, markdown-it-py 4.2.0,
mdurl 0.1.2, zipp 4.1.2, truststore 0.10.5, annotated-doc 0.0.5 (MIT); click 8.5.0, psutil 7.2.2, httpx2 2.16.0,
httpcore2 2.16.0 (BSD-3-Clause); pygments 2.21.0 (BSD-2-Clause); shellingham 1.5.4 (ISC).

NVIDIA's CUDA wheels (`nvidia-*`, `cuda-toolkit`, `cuda-bindings`) and the CUDA toolkit you install for building are
under NVIDIA's own licence agreements; you download them from NVIDIA's and PyTorch's package indexes yourself.

## Build tools and images (downloaded by you or by setup; not shipped here)

Checked on 2026-11-01 in each project's own licence statement (the source named in the table).

| Part | Used by | Licence | Checked in |
|---|---|---|---|
| CUDA base image `nvidia/cuda:13.3.0-devel-ubuntu24.04` (pinned by digest in `Dockerfile`) | the container (`docs/CONTAINER.md`) | NVIDIA End User License Agreements: "By pulling and using the CUDA images, you accept the terms and conditions of these licenses" (the CUDA toolkit: License Agreement for NVIDIA Software Development Kits; the NVIDIA Deep Learning Container License). The image also holds Ubuntu packages and open-source parts under their own licences (for example GPL; NVIDIA archives their sources at developer.download.nvidia.com/compute/cuda/opensource/image). Its Dockerfiles: BSD-3-Clause | the image's Docker Hub description (End User License Agreements section); docs.nvidia.com/cuda/eula |
| Ubuntu packages `build-essential`, `ca-certificates`, `git`, `procps` | the container (apt) | each package's own licence (its `/usr/share/doc/<package>/copyright` in the image) | not checked one by one |
| uv 0.11.21 (`ghcr.io/astral-sh/uv` image, pinned by digest; or your own uv) | `setup.sh`, the container | MIT or Apache-2.0, at your option. The image is `FROM scratch` with only the `uv` and `uvx` programs | uv's `README.md` (License) and `Dockerfile` (main branch) |
| Python 3.12.13 from python-build-standalone (`uv python install`) | `setup.sh` | Python and its dependencies under several licences, "most ... fairly permissive"; the builds avoid GPL readline (libedit instead) and leave out `_gdbm`. Each archive holds its licence texts and licence metadata | python-build-standalone documentation ("Licensing") |
| Rust toolchain (cargo, rustc) | `setup.sh env` builds trianglebvh with it | the Rust project: "generally dual-licensed" Apache-2.0 and MIT | rust-lang.org/policies/licenses |
| bun (tested 1.4.4) | `web/` (the optional web export) | "Bun itself is MIT-licensed"; it statically links JavaScriptCore and WebKit (LGPL-2) and other parts listed in its `LICENSE.md` (MIT, BSD, Apache-2.0, LGPL-2.1 tinycc, others) | oven-sh/bun `LICENSE.md` (main branch) |

## Apple Silicon route (`mac/setup_mac.sh`; experimental, docs/MAC.md)

| Part | Pin | Licence | Checked in |
|---|---|---|---|
| trellis.cpp (pwilkin/trellis.cpp) | `c0bed38c` (v0.8.1) | MIT, Copyright (c) 2026 Piotr Wilkin | `LICENSE` at the pinned commit |
| ggml (pwilkin/ggml, trellis.cpp's submodule) | `f8697b33` | MIT | `LICENSE` in the submodule |
| xatlas, meshoptimizer, Fast-Quadric-Mesh-Simplification (vendored in trellis.cpp) | trellis.cpp's copies | MIT | the licence headers and `LICENSE.md` files in `thirdparty/` |
| stb (vendored in trellis.cpp) | trellis.cpp's copy | public domain or MIT (your choice) | the end of each header |
| gguf (Python, writes the GGUF files) | 0.19.0 | MIT | package metadata |
| `mac/requirements.txt` packages not listed above: anyio 4.16.0, h11 0.16.0, truststore 0.10.5 (MIT); click 8.5.0, httpcore2 2.16.0, httpx2 2.16.0 (BSD-3-Clause) | as listed | as listed | package metadata |

trellis.cpp's post-processing (remesh, QEM simplification, xatlas unwrap, bake, BVH) was read at the pinned commit
for non-commercial parts: none found (docs/MAC.md, Licences). `setup_mac.sh` does not build its optional quad
retopology, which would download CGAL (GPL-3.0 or LGPL-3.0 by package), Boost and LEMON (Boost Software License),
GMP (LGPL-3.0 or GPL-2.0) and MPFR (LGPL-3.0).

## Tools called as external programs (optional; not shipped, not linked)

| Tool | Used for | Licence | Checked in |
|---|---|---|---|
| Blender (tested 5.2.3) | `render/`, the proxy bake in `web/bake_proxy.py` | GNU GPL (source GPL-2.0-or-later; binaries GPL-3.0-or-later). Blender's licence page: "What you create with Blender is your sole property." | blender.org/about/license |
| basis_universal `basisu` (tested 2.50.0) | KTX2 texture encoding in `web/` | Apache-2.0 | `LICENSE` in BinomialLLC/basis_universal |
| ffmpeg | `render/turntable_video.sh` | LGPL-2.1-or-later or GPL, depending on the build | your build's own notice |

## Web export packages (`web/package.json`, locked in `web/bun.lock`)

| Package | Version | Licence |
|---|---|---|
| @gltf-transform/core, /extensions, /functions | 4.5.1 | MIT |
| meshoptimizer | 1.2.0 | MIT |
| ktx-parse | 1.1.0 | MIT |
| sharp | 0.35.5 | Apache-2.0 |
| @img/sharp-libvips-* (libvips binaries that sharp installs for your platform) | 1.3.4 | LGPL-3.0-or-later |
| ndarray, ndarray-lanczos, ndarray-ops, ndarray-pixels, cwise-compiler, iota-array, uniq, is-buffer, property-graph, @img/colour, @types/ndarray | (bun.lock) | MIT |
| detect-libc | 2.1.2 | Apache-2.0 |
| semver | 7.8.5 | ISC |

Checked in each installed package's `package.json`. sharp loads libvips as a shared library at run time (LGPL-3.0:
you may replace that library).

## Not used (excluded on purpose)

| Part | Licence | Why excluded |
|---|---|---|
| cubvh (CuMesh's submodule, JeffreyXiang/cubvh @ `ce92267a`) | MIT and NVIDIA Source Code License (instant-ngp): research or evaluation only | never fetched or compiled; trianglebvh replaces its BVH (see "cubvh" above) |
| nvdiffrast, nvdiffrec (NVlabs) | NVIDIA Source Code License: research or evaluation only | uvrast replaces the two nvdiffrast calls of the bake (clean room); `run.py` and `setup.sh` stop if nvdiffrast can be imported |
| briaai/RMBG-2.0 | CC-BY-NC-4.0 | TRELLIS.2's default background remover; replaced by BiRefNet general (MIT). `run.py` stubs the remover class and stops if RMBG-2.0 is in the Hugging Face cache or any load names it |
| FLUX.2-klein-9B | FLUX Non-Commercial License | not commercial |
| Qwen-Image | research licence | not commercial |
