# The 3D stage of this repository in a container: trellis2/setup.sh env runs during the image build, so nothing
# compiles on the machine that runs the container. The weights are not in the image: DINOv3 is gated, so you log in
# to Hugging Face and download the weights yourself into a folder you mount at /data. Steps: docs/CONTAINER.md.
#   docker build -t conjure3d:local --build-arg TORCH_CUDA_ARCH_LIST=8.6 --build-context trianglebvh=PATH/TO/trianglebvh .

# trianglebvh (the GLB export's BVH) is not published yet: give a git clone of it as the build context "trianglebvh"
# (it must hold the commit that setup.sh pins). Once it is published, use --build-arg TRIANGLEBVH_GIT=<its git URL>
# instead, and this stage stays empty.
FROM scratch AS trianglebvh

# CUDA 13.3.0 toolkit (the version the work-identical checks used), Ubuntu 24.04, pinned by digest
FROM nvidia/cuda:13.3.0-devel-ubuntu24.04@sha256:ef2203909e80b8b976cfc672f7e2ae2b00bc0e25c404ee86d89e10a3802f1c52

# uv 0.11.21 (the version setup was tested with), pinned by digest
COPY --from=ghcr.io/astral-sh/uv:0.11.21@sha256:ff07b86af50d4d9391d9daf4ff89ce427bc544f9aae87057e69a1cc0aa369946 /uv /uvx /usr/local/bin/

# C++ compiler for the CUDA extensions, git for the pinned clones, procps for setup.sh's RAM guard. The Ubuntu packages
# come from snapshot.ubuntu.com at a fixed time, so every build gets the same versions (this pins them, not "=version")
ARG UBUNTU_SNAPSHOT=20261101T000000Z
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --snapshot "$UBUNTU_SNAPSHOT" && \
    apt-get install -y --no-install-recommends --snapshot "$UBUNTU_SNAPSHOT" build-essential ca-certificates git procps

# Compute capabilities to build the CUDA extensions for: tools/preflight.py prints your GPU's (e.g. 8.6, 8.9, 12.0)
ARG TORCH_CUDA_ARCH_LIST=8.6
ARG MAX_JOBS=2
ENV T2_VENV=/opt/conjure3d/venv-t2 T2_SRC=/opt/conjure3d/src CUDA_HOME=/usr/local/cuda \
    HF_HOME=/data/hf T2_WEIGHTS=/data/weights TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS \
    UV_PYTHON_INSTALL_DIR=/opt/conjure3d/python UV_CACHE_DIR=/var/cache/uv UV_LINK_MODE=copy UV_NO_PROGRESS=1 \
    HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
WORKDIR /opt/conjure3d/repo

# setup.sh env needs only trellis2/; copied first so that changes elsewhere do not rebuild the environment
COPY trellis2/ trellis2/
# Rust 1.97.1 (the version setup was tested with; official image, pinned by digest) builds trianglebvh. It is mounted
# only for this step, so it is not in the image. trianglebvh's crate has no dependencies, and cargo runs offline.
ARG TRIANGLEBVH_GIT=file:///opt/conjure3d/trianglebvh
RUN --mount=type=cache,target=/var/cache/uv \
    --mount=type=bind,from=trianglebvh,target=/opt/conjure3d/trianglebvh \
    --mount=type=bind,from=rust:1.97.1-slim-trixie@sha256:8e8cf8f7fd54a2d23d5a743b3a03f56e26b6c774276c33fa0595111704ebb15c,source=/usr/local,target=/opt/rust,rw \
    if [ "$TRIANGLEBVH_GIT" = file:///opt/conjure3d/trianglebvh ] && [ ! -d /opt/conjure3d/trianglebvh/.git ]; then \
      echo "STOP: give trianglebvh's git repository: --build-context trianglebvh=PATH/TO/trianglebvh"; exit 1; fi && \
    PATH=/opt/rust/cargo/bin:$PATH RUSTUP_HOME=/opt/rust/rustup CARGO_HOME=/opt/rust/cargo \
    TRIANGLEBVH_GIT=$TRIANGLEBVH_GIT setsid -w bash trellis2/setup.sh env

COPY . .
# CUDA numbers the GPUs in PCI bus order, the same order as nvidia-smi (CUDA's default is fastest first)
ENV PATH=/opt/conjure3d/venv-t2/bin:$PATH CUDA_DEVICE_ORDER=PCI_BUS_ID
CMD ["bash"]
