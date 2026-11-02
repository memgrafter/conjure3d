# What the container build downloads

This page lists every internet host that a clean build of the `Dockerfile` contacted, what it downloaded there, and
how that download is pinned. It comes from one recorded build (below). Make the same record yourself with
`tools/build_netscan.sh`.

## Hosts

One clean build on 2026-11-02, on an x86_64 emulator: a new builder with an empty cache, `--no-cache --pull`,
trianglebvh from a local git clone (`--build-context`), `TORCH_CUDA_ARCH_LIST=8.6`. It used the commit before
`SETUP_NO_GPU`, so it downloaded everything and then stopped in setup's last import check, which needs a GPU. The
downloads are the same with `SETUP_NO_GPU`.

| Host | Port | What the build downloads there | Pinned |
|---|---|---|---|
| registry-1.docker.io, auth.docker.io, production.cloudfront.docker.com | 443 | the `nvidia/cuda` base image and the `rust` image (Docker Hub) | yes, by digest |
| ghcr.io, pkg-containers.githubusercontent.com | 443 | the `uv` image (GitHub Container Registry) | yes, by digest |
| snapshot.ubuntu.com | 443 | the Ubuntu package lists and the 18 packages that the build installs | yes, by snapshot time; apt checks the signed lists and each package's hash |
| archive.ubuntu.com, security.ubuntu.com | 80 | today's Ubuntu package lists. `apt-get update --snapshot` also gets them, but the install uses only the snapshot | lists only, not pinned; no package comes from here |
| developer.download.nvidia.com | 443 | the package list of NVIDIA's CUDA apt repository, which the base image lists | list only, not pinned; no package comes from here |
| github.com | 443 | git clones of TRELLIS.2 (with its submodules), CuMesh (without its cubvh submodule), FlexGEMM and BiRefNet; utils3d and uvrast (Python packages from git) | yes, by commit |
| gitlab.com | 443 | Eigen, a submodule of TRELLIS.2's o-voxel | yes, by the commit that TRELLIS.2 records |
| api.github.com, raw.githubusercontent.com | 443 | uv asks GitHub for the commit and the package metadata of utils3d and uvrast | yes, by commit |
| releases.astral.sh | 443 | Python 3.12.13 for the venv (uv's Python download) | by version |
| pypi.org, files.pythonhosted.org | 443 | Python packages (`trellis2/requirements.txt`) | by version, no file hashes |
| download.pytorch.org, download-r2.pytorch.org | 443 | PyTorch packages for CUDA 13.0 | by version, no file hashes |
| pypi.nvidia.com | 443 | NVIDIA's CUDA library packages that PyTorch needs | by version, no file hashes |

Not contacted: Hugging Face (the weights are not in the image), crates.io (trianglebvh has no Rust dependencies, and
cargo runs offline), static.rust-lang.org (Rust comes from the `rust` image).

## Points to know

- The Python packages are pinned by version only. A package index can add a new file to a version that exists, and
  `setup.sh` would not see it. `uv pip install` uses two indexes (pypi.org and download.pytorch.org) with
  `--index-strategy unsafe-best-match`, so a name on both indexes can come from either.
- `apt-get update` also gets today's Ubuntu lists over plain HTTP and NVIDIA's list. The build installs nothing from
  them.
- Not checked: whether uv checks a hash of the Python download.
- The record leaves out port 22 (the virtual machine's own SSH) and the image of the builder itself
  (`moby/buildkit`), which `tools/build_netscan.sh` pulls before it starts to record.

## Make the record

Run it as root on the Linux machine that runs Docker. With Colima or another virtual machine, run it in the virtual
machine. It needs `tcpdump`, Docker with buildx, and Python 3.

```sh
sudo bash tools/build_netscan.sh scan-out --build-arg TORCH_CUDA_ARCH_LIST=8.6 --build-context trianglebvh=../trianglebvh
```

- It always makes a clean build: a new buildx builder, `--no-cache --pull` and empty cache mounts. It removes that
  builder at the end. Add `-t conjure3d:local --load` to keep the image.
- `scan-out/report.md` lists each host with its ports, addresses and number of connections, and what the build log
  shows (images, Ubuntu packages, git commits, Python packages). `scan-out/build.pcap` holds only the start of each
  connection, the DNS answers, the TLS hello and the first line of each HTTP request, not the downloaded data.
- A host name comes from the TLS hello (SNI) of the connection, else from the DNS answer for its address.
