# What the container build downloads

This page lists every internet host that a clean build of the `Dockerfile` contacted, what it downloaded there, and
how that download is pinned. It comes from one recorded build (below). Make the same record yourself with
`tools/build_netscan.sh`.

## Hosts

One clean build on 2026-11-02 on an x86_64 Linux machine without a GPU (Docker 26.1): a new builder with an empty
cache, `--no-cache --pull`, `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0+PTX"`. The build finished (exit 0).
A second clean build the same day, with 10.3 added to that list, contacted the same hosts for the same downloads.

| Host | Port | What the build downloads there | Pinned |
|---|---|---|---|
| registry-1.docker.io, production.cloudfront.docker.com | 443 | the `nvidia/cuda` base image and the `rust` image (Docker Hub) | yes, by digest |
| ghcr.io, pkg-containers.githubusercontent.com | 443 | the `uv` image (GitHub Container Registry) | yes, by digest |
| snapshot.ubuntu.com | 443 | the Ubuntu package lists and the 18 packages that the build installs | yes, by snapshot time; apt checks the signed lists and each package's hash |
| archive.ubuntu.com, security.ubuntu.com | 80 | today's Ubuntu package lists. `apt-get update --snapshot` also gets them, but the install uses only the snapshot | lists only, not pinned; no package comes from here |
| developer.download.nvidia.com | 443 | the package list of NVIDIA's CUDA apt repository, which the base image lists | list only, not pinned; no package comes from here |
| github.com | 443 | git clones of TRELLIS.2 (with its submodules), CuMesh (without its cubvh submodule), FlexGEMM and BiRefNet; utils3d, uvrast and trianglebvh (packages from git) | yes, by commit |
| gitlab.com | 443 | Eigen, a submodule of TRELLIS.2's o-voxel | yes, by the commit that TRELLIS.2 records |
| api.github.com, raw.githubusercontent.com | 443 | uv asks GitHub for the commit and the package metadata of the git packages | yes, by commit |
| releases.astral.sh | 443 | Python 3.12.13 for the venv (uv's Python download) | by version |
| pypi.org, files.pythonhosted.org | 443 | Python packages (`trellis2/requirements-pypi.lock`) | yes, by version and file SHA-256 |
| download.pytorch.org, download-r2.pytorch.org | 443 | torch, torchvision, triton and xformers for CUDA 13.0 (`trellis2/requirements-pytorch.lock`) | yes, by version and file SHA-256 |

Not contacted: pypi.nvidia.com (the earlier record, without the hashed locks, got NVIDIA's CUDA library packages
there), Hugging Face (the weights are not in the image), crates.io (trianglebvh has no Rust dependencies, and
cargo runs offline), static.rust-lang.org (Rust comes from the `rust` image).

## Points to know

- Each Python package comes from one index, with `--require-hashes`: a file that is not in the lock stops the build.
- `apt-get update` also gets today's Ubuntu lists over plain HTTP and NVIDIA's list. The build installs nothing from
  them.
- Not checked: whether uv checks a hash of the Python download.
- The record leaves out the image of the builder itself (`moby/buildkit:buildx-stable-1`, a tag, not a digest) and
  the small Debian image that runs tcpdump; `tools/build_netscan.sh` pulls both before it starts to record.

## Make the record

Run it on the Linux machine that runs Docker, as a user who can use Docker; root is not needed. With Colima or
another virtual machine, run it in the virtual machine. It needs Docker with buildx, and Python 3.

```sh
bash tools/build_netscan.sh scan-out --build-arg TORCH_CUDA_ARCH_LIST=8.6
```

- tcpdump runs in a container in the builder's network namespace, so the record holds only the build's traffic.
- `BUILDER_OPTS="--driver-opt cpuset-cpus=2-5 --driver-opt cpu-shares=2"` keeps CPU cores free for other work (the
  recorded build used this and `--driver-opt memory=16g`).

- It always makes a clean build: a new buildx builder, `--no-cache --pull` and empty cache mounts. It removes that
  builder at the end. Add `-t conjure3d:local --load` to keep the image.
- `scan-out/report.md` lists each host with its ports, addresses and number of connections, and what the build log
  shows (images, Ubuntu packages, git commits, Python packages). `scan-out/build.pcap` holds only the start of each
  connection, the DNS answers, the TLS hello and the first line of each HTTP request, not the downloaded data.
- A host name comes from the TLS hello (SNI) of the connection, else from the DNS answer for its address.
