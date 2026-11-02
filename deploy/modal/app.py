"""Runs conjure3d on Modal: one picture in, asset.glb out. Steps: deploy/modal/README.md.

  export CONJURE3D_IMAGE=123456789012.dkr.ecr.us-east-1.amazonaws.com/conjure3d:cc86   # your image in a registry
  modal run deploy/modal/app.py::download_weights                    # once, into the Modal Volume "conjure3d-data"
  modal run deploy/modal/app.py --image lantern.png --size 0.35      # writes lantern-clean-seed42.glb here

Settings (environment variables on your computer when you run `modal run`):
  CONJURE3D_IMAGE            the image (required). Build it yourself (docs/CONTAINER.md) and push it to your registry.
  CONJURE3D_REGISTRY_SECRET  the Modal Secret with the registry login (default: none, for a public image). For AWS ECR:
                             AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION. Other registries: REGISTRY_USERNAME,
                             REGISTRY_PASSWORD.
  CONJURE3D_GPU              Modal GPU type (default L40S). The GPU must match the TORCH_CUDA_ARCH_LIST of the image.
  CONJURE3D_HF_SECRET        the Modal Secret with HF_TOKEN, for download_weights only (default: huggingface)
"""
import os, pathlib, re, subprocess, sys

import modal

IMAGE = os.environ.get("CONJURE3D_IMAGE", "")
REGISTRY_SECRET = os.environ.get("CONJURE3D_REGISTRY_SECRET", "")
GPU = os.environ.get("CONJURE3D_GPU", "L40S")
HF_SECRET = os.environ.get("CONJURE3D_HF_SECRET", "huggingface")
REPO = "/opt/conjure3d/repo"
# Modal's own Python (add_python) runs this file; the image's venv stays as built and runs the pipeline (T2_VENV)
SYSTEM_PATH = "/usr/local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
# The Volume is only for the pipeline's files, so setup's free-space floors (20 GB each by default) are lowered
SETUP_ENV = {"SETUP_MIN_FREE_GB": "1", "T2_WEIGHTS_MIN_FREE_GB": "2"}

if modal.is_local() and not IMAGE:
    sys.exit("set CONJURE3D_IMAGE to your conjure3d image in a registry (deploy/modal/README.md)")

image = modal.Image.from_registry(
    IMAGE or "unset",
    secret=modal.Secret.from_name(REGISTRY_SECRET) if REGISTRY_SECRET else None,
    add_python="3.12",
    setup_dockerfile_commands=[f"ENV PATH={SYSTEM_PATH}"],
).workdir(REPO)
data = modal.Volume.from_name("conjure3d-data", create_if_missing=True, version=2)
app = modal.App("conjure3d")


@app.function(image=image, volumes={"/data": data}, secrets=[modal.Secret.from_name(HF_SECRET)], timeout=3 * 3600)
def download_weights():
    """setup.sh weights into the Volume: Hugging Face cache in /data/hf, weight copy in /data/weights."""
    subprocess.run(["setsid", "-w", "bash", "trellis2/setup.sh", "weights"], cwd=REPO, check=True,
                   env={**os.environ, **SETUP_ENV})
    data.commit()


@app.function(image=image, gpu=GPU, volumes={"/data": data}, timeout=3600)
def make_object(picture: bytes, name: str, size_m: float, seed: int, profile: str) -> bytes:
    """conjure3d.py on one picture. Returns the bytes of asset.glb."""
    work = pathlib.Path("/tmp/work")
    work.mkdir(parents=True, exist_ok=True)
    src = work / pathlib.Path(name).name
    src.write_bytes(picture)
    cmd = [sys.executable, "-I", "conjure3d.py", str(src), str(size_m), "--seed", str(seed), "--profile", profile,
           "--out", str(work / "out")]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print(p.stdout, p.stderr, sep="\n")
    done = re.search(r"^DONE\s+(\S+asset\.glb)$", p.stdout, re.M)
    if p.returncode != 0 or not done:
        raise RuntimeError(f"conjure3d.py ended with exit status {p.returncode} (3 = out of GPU memory); see the log above")
    return pathlib.Path(done.group(1)).read_bytes()


@app.local_entrypoint()
def main(image: str, size: float, seed: int = 42, profile: str = "clean", out: str = ""):
    picture = pathlib.Path(image)
    glb = make_object.remote(picture.read_bytes(), picture.name, size, seed, profile)
    dest = pathlib.Path(out or f"{picture.stem}-{profile}-seed{seed}.glb")
    dest.write_bytes(glb)
    print(f"{dest} ({len(glb)} bytes)")
