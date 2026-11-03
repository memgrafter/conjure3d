# Container

The `Dockerfile` builds the 3D environment that `trellis2/setup.sh env` makes: Python, the pinned packages and the
compiled CUDA extensions. With the image, nothing compiles on the machine that runs it. The image holds no weights.
You build the image yourself. This repository publishes no image.

Status: the image was built and checked without a GPU on 2026-11-02 (an x86_64 Debian 13 machine, Docker 26.1,
all the GPU targets below). It has not run on a GPU yet, and its outputs are not compared with a setup on the host
yet. The bytes of the compiled libraries differ from a host setup (another compiler and another system), so check
before you mix the two ([WORK_IDENTICAL.md](WORK_IDENTICAL.md)). For example, trianglebvh's library in the image has
another SHA-256 than the host build (`1802822e`, host `4fe2855a`): its code sections are byte-identical, and only the
C start-up code of the system compiler differs (Ubuntu's gcc 13 in the image, Debian's gcc 14 in the host build).

You can run the image on your own Linux machine (below), on vast.ai ([deploy/vast/README.md](../deploy/vast/README.md))
or on Modal ([deploy/modal/README.md](../deploy/modal/README.md)).

## What you need

- To run it: Linux, an NVIDIA GPU, and a driver that supports CUDA 13.3 or newer (`nvidia-smi` shows "CUDA Version"
  13.3 or higher). The image does not start on an older driver.
- Docker Engine with BuildKit (the default builder since Docker 23) and the NVIDIA Container Toolkit.
- To build it: an x86_64 Linux machine with Docker. The build needs no GPU.
- Disk for Docker: the image is 14.9 GB with all GPU targets. During the clean build below, BuildKit used 30.8 GB
  (the last sample before the end; sampled every 10 minutes), and then Docker loaded the 14.9 GB image next to it.
  Keep 50 GB free. `setup.sh` stops if 20 GB or less is free next to the venv, and in the build that is Docker's disk.
- Disk for the data folder: 16.6 GB of Hugging Face downloads and a 13.6 GB copy of the weights.

Check the host:

```sh
python3 -I tools/preflight.py --container --data data
```

## Build

Set `TORCH_CUDA_ARCH_LIST` to your GPU's compute capability. The preflight check prints it, for example 8.6 for an
RTX 3090 or 8.9 for an RTX 4090:

```sh
docker build -t conjure3d:local --build-arg TORCH_CUDA_ARCH_LIST=8.6 .
```

For all the GPUs in the table below:

```sh
docker build -t conjure3d:local --build-arg "TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0;10.0;10.3;12.0+PTX" .
```

- The compiled code runs on GPUs of the same major version and the same or a higher minor version: a build for 8.6
  runs on 8.6 to 8.9, but not on 8.0 (A100), 9.0 (H100) or 12.0 (RTX 5090).
- Several values (`"8.6;8.9"`) build for several GPU generations. The build takes longer and the image is larger.
- `--build-arg MAX_JOBS=4` compiles faster and uses more RAM (default 2).
- trianglebvh (the BVH of the GLB export) comes from https://github.com/memgrafter/trianglebvh.git at the commit that
  `trellis2/setup.sh` pins (`TRIANGLEBVH_SHA`). To use another copy, give `--build-arg TRIANGLEBVH_GIT=<git URL>`;
  the build must be able to reach it, and the pin still applies.
- Rust 1.97.1 compiles trianglebvh during the build. It comes from the official `rust` image, pinned by digest, and
  is not in the final image.
- The Ubuntu packages come from snapshot.ubuntu.com at a fixed time (`UBUNTU_SNAPSHOT`), so a later build gets the
  same versions. The base images are pinned by digest, the Python packages by version and file SHA-256 (the hashed
  locks in `trellis2/`), the source clones by commit.
- The Dockerfile copies `trellis2/` before it runs `setup.sh env`. A change there (`setup.sh`, `patches/`, the
  locks) makes the next build run the whole environment step again, with the CUDA builds.
- The work-identical checks used `TORCH_CUDA_ARCH_LIST=8.6` and the CUDA 13.3 toolkit, the same as this base image.
- What the build downloads, and from where: [BUILD_NETWORK.md](BUILD_NETWORK.md).

### Measured build

One clean build (`tools/build_netscan.sh`: new builder, empty cache, `--no-cache --pull`) on 2026-11-02 on a machine
without a GPU, with `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;10.3;12.0+PTX"` and `MAX_JOBS=2`. The builder had 4
of the 6 CPU cores (Intel Core i5-8500T, 23 GB RAM) at the lowest CPU priority, because the machine did other work at
the same time.

| Step | Time |
|---|---|
| whole build, start to image in Docker | 99 min |
| downloads, Ubuntu packages, venv | 30 min (the downloads were slow; 9 min in an earlier build) |
| FlexGEMM, CuMesh, o-voxel (CUDA builds) | 14, 33 and 10 min |
| export of the image and load into Docker | 12 min |

| Part of the image | Size |
|---|---|
| whole image | 14.9 GB |
| layer of `setup.sh env` | 7.67 GB |
| venv (`/opt/conjure3d/venv-t2`) | 6.0 GB |
| sources (`/opt/conjure3d/src`) | 1.2 GB |
| Python (`/opt/conjure3d/python`) | 108 MB |

An earlier build without 10.3 took 79 min, and its CUDA builds took 13, 32 and 10 min. Its image was 22 MB smaller.
The sizes of the venv, the sources and Python are from that build.

Checked in the image without a GPU (`cuobjdump --list-elf` and `--list-ptx`): FlexGEMM, CuMesh and o-voxel each have
machine code for sm_80, sm_86, sm_89, sm_90, sm_100, sm_103 and sm_120 in every CUDA object, and PTX for sm_120.
CuMesh's xatlas library has no CUDA code. No cubvh library is in the image. Rust is not in the image.

### GPUs

The cards come from NVIDIA's CUDA GPU list (https://developer.nvidia.com/cuda-gpus, read 2026-11-02). "In the image"
is the code of the CUDA extensions that the build above compiles. PyTorch 2.14.1+cu130 has machine code for sm_75,
sm_80, sm_86, sm_90, sm_100 and sm_120; xformers 0.0.35 for sm_75, sm_80, sm_90a, sm_100a, sm_103a, sm_110a, sm_120a,
sm_121a and PTX for sm_80. Both cover each row.

| Compute capability | Cards | In the image | Tested |
|---|---|---|---|
| 8.0 | A100, A30 | sm_80 | built only |
| 8.6 | RTX 3090, RTX 3090 Ti, other RTX 30 cards, A10, A40, RTX A6000 | sm_86 | host setup on RTX 3090 and RTX 3090 Ti; the image: built only |
| 8.9 | RTX 4090, other RTX 40 cards, RTX 5000 Ada, RTX 6000 Ada, L4, L40S | sm_89 | built only |
| 9.0 | H100, H200 | sm_90 | built only |
| 10.0 | B200 | sm_100 | built only |
| 10.3 | B300 | sm_103 | built, untested |
| 12.0 | RTX 5090, 5080, 5070 Ti, 5070, 5060 Ti, 5060, 5050, RTX PRO 6000 Blackwell (Workstation and Server Editions) | sm_120 and PTX | built only |

- 7.5 (T4, RTX 20 cards) is not in this build. CUDA 13 still supports it: add 7.5 to `TORCH_CUDA_ARCH_LIST`.
- The image is for x86_64 only. GH200, GB200, GB300 and DGX Spark (GB10) have Arm CPUs.
- In the container, `python -I tools/preflight.py` names each card, its memory class and the profile to start
  with, and fails for a GPU that the image's code does not run on. Without a GPU it fails with
  `FAIL  nvidia-smi not found`.

## GPU numbers

In the container, GPU N is the same GPU as GPU N of `nvidia-smi` on the host if two things are true:

1. The container gets all GPUs: `--gpus all`.
2. CUDA counts the GPUs in PCI bus order, the same as `nvidia-smi`. The image sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
   Without it, CUDA puts the fastest GPU first. On a machine with three RTX 3090 and one RTX 3090 Ti, that can make
   the 3090 Ti GPU 0 in PyTorch while it is GPU 3 in `nvidia-smi`.

To use one GPU, give the container all GPUs and select the GPU with the host's number in `CUDA_VISIBLE_DEVICES`:

```sh
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=2 ...      # host GPU 2
```

- With `--gpus device=2`, the container sees only that GPU, so it is GPU 0 in the container.
- `CUDA_VISIBLE_DEVICES` also takes the GPU's UUID from `nvidia-smi -L`, for example `GPU-8932f937-...`. A UUID does
  not depend on the order.
- With `--gpus all` the container can use every GPU. `CUDA_VISIBLE_DEVICES` hides the other GPUs from CUDA programs,
  but `nvidia-smi` in the container still shows them. If the container must see only its own GPU, use
  `--gpus device=2`, and in the container use GPU 0.

Check the numbers. On the host, this runs the image with all GPUs and compares the UUID of each GPU inside and
outside:

```sh
python3 -I tools/preflight.py --container --image conjure3d:local
```

Each GPU gives a line such as `PASS  in conjure3d:local: CUDA device 3 is host GPU 3 (GPU-e53bc0b5-..., PCI
0000:06:00.0)`. In the container, `python -I tools/preflight.py` prints the same lines for the GPUs that it sees.

## Weights (once)

First, open https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m with your own Hugging Face account. Read
the DINOv3 License and accept it if you agree. Approval can take some time. Then:

```sh
mkdir -p data
docker run --rm -it --gpus all --user "$(id -u):$(id -g)" -v "$PWD/data:/data" conjure3d:local \
  bash -c 'hf auth login && setsid -w bash trellis2/setup.sh weights'
```

- `hf auth login` keeps your token in `data/hf/token`. Keep that file private.
- `setup.sh weights` also checks the free space next to the venv. In the container that is Docker's disk, but the
  weights go to `data/`. If it stops there, add `-e SETUP_MIN_FREE_GB=1` after `--gpus all`.

## One object

Put the picture in a folder, for example `work/`:

```sh
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/data:/data" -v "$PWD/work:/work" conjure3d:local \
  python -I conjure3d.py /work/lantern.png 0.35 --out /work/out
```

The result is `work/out/lantern-clean-seed42/lantern/asset.glb`. To use host GPU 1, add `-e CUDA_VISIBLE_DEVICES=1`
after `--gpus all`. The preflight check also runs in the container:
`docker run --rm --gpus all -v "$PWD/data:/data" conjure3d:local python -I tools/preflight.py`.

## Rented GPUs

- vast.ai: [deploy/vast/README.md](../deploy/vast/README.md). An instance runs the image with SSH. The weights stay on
  a vast.ai volume.
- Modal: [deploy/modal/README.md](../deploy/modal/README.md). One command sends a picture and returns `asset.glb`. The
  weights stay on a Modal Volume.

Both need the image in a registry that they can pull from, for example your own private Amazon ECR repository. The
image name and the registry login are settings of the vast.ai template and of `app.py`.
