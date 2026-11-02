# Container

The `Dockerfile` builds the 3D environment that `trellis2/setup.sh env` makes: Python, the pinned packages and the
compiled CUDA extensions. With the image, nothing compiles on the machine that runs it. The image holds no weights.
You build the image yourself. This repository publishes no image.

Status: the image build is not tested yet, and its outputs are not compared with a setup on the host yet. The bytes
can differ from a host setup (another compiler and another system), so check before you mix the two
([WORK_IDENTICAL.md](WORK_IDENTICAL.md)).

## What you need

- Linux, an NVIDIA GPU and a driver for CUDA 13.
- Docker Engine with BuildKit (the default builder since Docker 23) and the NVIDIA Container Toolkit.
- Disk for Docker: the CUDA base image (3.6 GB download, compressed) and the build (venv 5.9 GB, sources 1.3 GB, uv's
  download cache). The total is not measured. `setup.sh` stops if 20 GB or less is free next to the venv, and in the
  build that is Docker's disk.
- Disk for the data folder: 16.6 GB of Hugging Face downloads and a 13.6 GB copy of the weights.

Check the host:

```sh
python3 -I tools/preflight.py --container --data data
```

## Build

Set `TORCH_CUDA_ARCH_LIST` to your GPU's compute capability. The preflight check prints it, for example 8.6 for an
RTX 3090 or 8.9 for an RTX 4090.

```sh
docker build -t conjure3d:local --build-arg TORCH_CUDA_ARCH_LIST=8.6 .
```

- Several values (`"8.6;8.9"`) build for several GPU generations. The build takes longer and the image is larger.
- `--build-arg MAX_JOBS=4` compiles faster and uses more RAM (default 2).
- The work-identical checks used `TORCH_CUDA_ARCH_LIST=8.6` and the CUDA 13.3 toolkit, the same as this base image.

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

The result is `work/out/lantern-clean-seed42/lantern/asset.glb`. With several GPUs, select one with
`--gpus device=1`. The preflight check also runs in the container:
`docker run --rm --gpus all -v "$PWD/data:/data" conjure3d:local python -I tools/preflight.py`.
