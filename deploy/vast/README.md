# conjure3d on vast.ai

These steps run the conjure3d container on a GPU that you rent on vast.ai. Status: not tested on vast.ai yet. The
steps follow vast.ai's documentation.

## What you need

- The image in a registry that vast.ai can pull from. This repository publishes no image: build it
  ([docs/CONTAINER.md](../../docs/CONTAINER.md)) and push it to your own registry.
- A build for the GPUs you rent. The image runs only on GPUs that match its `TORCH_CUDA_ARCH_LIST`: a build for 8.6
  runs on compute capability 8.6 to 8.9 (RTX 3090, RTX A6000, RTX 4090, L40S). For an H100 (9.0) or an RTX 5090
  (12.0), build with that value too, for example `--build-arg TORCH_CUDA_ARCH_LIST="8.6;9.0;12.0"`.
- A machine whose driver supports CUDA 13.3 or newer. The image does not start on an older driver.
- A Hugging Face account that has accepted the DINOv3 License (see "Weights" in [docs/CONTAINER.md](../../docs/CONTAINER.md)).

## 1. Login data for a private registry

vast.ai pulls the image when it creates the instance. For a private registry, give it a login:

- Amazon ECR: user name `AWS`, password from `aws ecr get-login-password --region REGION`, server
  `ACCOUNT.dkr.ecr.REGION.amazonaws.com`. This password expires after 12 hours, so get a new one before you create an
  instance.
- Other registries: your user name, an access token with read access only, and the registry's server name
  (none for Docker Hub).

## 2. Hugging Face token

In your vast.ai account settings, add the environment variable `HF_TOKEN` with a Hugging Face token that has read
access. Account variables go to all your instances. Do not put the token in a template.

## 3. Template

Make a template (Templates > New) with these values:

| Field | Value |
|---|---|
| Image Path:Tag | your image, for example `ACCOUNT.dkr.ecr.REGION.amazonaws.com/conjure3d:cc86` |
| Docker Repository Authentication | the login from step 1 |
| Launch Mode | Interactive shell server, SSH |
| On-start Script | `bash /opt/conjure3d/repo/deploy/vast/onstart.sh` |
| Extra Filters | `cuda_vers>=13.3 gpu_ram>=22 cpu_arch=amd64` and the GPUs of your build, for example `compute_cap>=860 compute_cap<900` |
| Disk Space | the image size plus room for your outputs; the image size is in [docs/CONTAINER.md](../../docs/CONTAINER.md) |
| Volume | "Add recommended volume settings", 40 GB, mounted at `/data` |

- In SSH mode vast.ai replaces the image's start command and starts SSH. The on-start script makes the image's
  settings visible in SSH sessions and, if `HF_TOKEN` is set, starts the weight download in the background
  (log: `/data/setup-weights.log`).
- The volume keeps the weights (16.6 GB of downloads and a 13.6 GB copy) after you destroy the instance. A vast.ai
  volume stays on its machine: a new instance can use it only on the same machine. Its size cannot change later.
- The disk size of an instance also cannot change later.
- The on-start script sets `SETUP_MIN_FREE_GB=1` and `T2_WEIGHTS_MIN_FREE_GB=2`, because the volume holds only these
  files. Without that, setup stops unless 20 GB stay free.

With the vast.ai CLI, the same instance without a volume is (OFFER_ID from `vastai search offers`):

```sh
vastai search offers 'cuda_vers>=13.3 gpu_ram>=22 cpu_arch=amd64 compute_cap>=860 compute_cap<900 num_gpus=1'
vastai create instance OFFER_ID --image ACCOUNT.dkr.ecr.REGION.amazonaws.com/conjure3d:cc86 \
  --login "-u AWS -p $(aws ecr get-login-password --region REGION) ACCOUNT.dkr.ecr.REGION.amazonaws.com" \
  --ssh --direct --disk DISK_GB --onstart-cmd 'bash /opt/conjure3d/repo/deploy/vast/onstart.sh'
```

Without a volume, `/data` is on the instance disk: add 31 GB to the disk size for the weights. For a volume from the
CLI, see `vastai create instance --help` (`--create-volume` with an ID from `vastai search volumes`, and
`--mount-path /data`).

## 4. First object

Connect with the SSH command that the instance page shows. Then:

```sh
cd /opt/conjure3d/repo
tail -f /data/setup-weights.log          # wait for the end of the weight download (first instance only)
python -I tools/preflight.py             # no FAIL lines; it also shows which GPU is which
```

Copy a picture to the instance (`scp -P PORT lantern.png root@HOST:/data/`), then:

```sh
python -I conjure3d.py /data/lantern.png 0.35 --out /data/out
```

Copy the result back with `scp -P PORT root@HOST:/data/out/lantern-clean-seed42/lantern/asset.glb .`.

## Costs and limits

- vast.ai bills the instance while it runs, and the disk and the volume while they exist, also when the instance
  is stopped. Destroy the instance and the volume when you are done.
- Time per object and the model load on vast.ai GPUs: not measured.
- Other GPU models give other bytes than the RTX 3090 results in the README.
