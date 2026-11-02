# conjure3d on Modal

`app.py` runs `conjure3d.py` on a Modal GPU: you send one picture and get `asset.glb` back. Status: not tested on Modal
yet. The steps follow Modal's documentation and the source of its Python client.

## What you need

- The image in a registry. This repository publishes no image: build it ([docs/CONTAINER.md](../../docs/CONTAINER.md))
  and push it to your own registry. Modal needs a `linux/amd64` image.
- A build for the Modal GPU you use. A build with `TORCH_CUDA_ARCH_LIST=8.6` runs on compute capability 8.6 to 8.9:
  Modal's `A10`, `L4` and `L40S`. For `A100` (8.0), `H100` (9.0) or newer GPUs, build with their values too.
- A Modal account and the `modal` Python package on your computer (`pip install modal`, then `modal setup`).
- A Hugging Face account that has accepted the DINOv3 License (see "Weights" in [docs/CONTAINER.md](../../docs/CONTAINER.md)).

Modal's documentation gives its GPU hosts driver 610.57.04 with CUDA 13.3, so the image's CUDA 13.3 base can start
there.

## 1. Secrets

Make the Modal Secrets in the Modal dashboard (Secrets) or with the CLI:

```sh
modal secret create huggingface HF_TOKEN=hf_...                 # read access is enough
modal secret create conjure3d-ecr AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1
```

- The registry Secret is for a private image. For Amazon ECR use the keys above, from an IAM user that can only pull
  this image. For other registries use `REGISTRY_USERNAME` and `REGISTRY_PASSWORD`.
- A command line with a secret goes into your shell history. The dashboard does not.

## 2. Settings

```sh
export CONJURE3D_IMAGE=ACCOUNT.dkr.ecr.REGION.amazonaws.com/conjure3d:cc86
export CONJURE3D_REGISTRY_SECRET=conjure3d-ecr   # leave unset for a public image
export CONJURE3D_GPU=L40S                         # optional; default L40S (48 GB)
```

## 3. Weights (once)

```sh
modal run deploy/modal/app.py::download_weights
```

This runs `setup.sh weights` in a container without a GPU. The files go to the Modal Volume `conjure3d-data`
(Hugging Face cache in `/data/hf`, 16.6 GB; weight copy in `/data/weights`, 13.6 GB). The Volume stays after the run.

## 4. One object

```sh
modal run deploy/modal/app.py --image lantern.png --size 0.35
```

The result is `lantern-clean-seed42.glb` in the current folder. Options: `--seed 7`, `--profile draft`,
`--out FILE`. The log of `conjure3d.py` shows in your terminal. Exit status 3 in that log means the GPU ran out of
memory.

## How it works

- `modal.Image.from_registry` uses your image and adds Modal's own Python 3.12 (`add_python`). Modal runs `app.py`
  with that Python. The pipeline runs in the image's own environment, as built (`T2_VENV`).
- Modal runs containers as root, and ignores the image's `USER`.
- `make_object` writes the picture to the container's disk, runs `conjure3d.py`, and returns the bytes of
  `asset.glb`. The other files of the run (`raw.glb`, `cut.png`, `record.json`) stay in the container and are lost
  when it ends.
- `download_weights` sets `SETUP_MIN_FREE_GB=1` and `T2_WEIGHTS_MIN_FREE_GB=2`, because the Volume holds only these
  files. Without that, setup stops unless 20 GB stay free.

## Not measured

Time per object, the model load from a Modal Volume, and the cost per object. Other GPU models give other bytes than
the RTX 3090 results in the README.
