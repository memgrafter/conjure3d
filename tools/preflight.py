"""Checks a machine before setup and before the first object. Reads only; it changes nothing and accepts no licence.

  source config.env && python3 -I tools/preflight.py          # setup on this machine (trellis2/setup.sh)
  python3 -I tools/preflight.py --container --data DIR        # the host of the container (docs/CONTAINER.md)
  python3 -I tools/preflight.py --container --image conjure3d:local   # also: GPU numbers inside the image = host's

Each line starts with PASS, WARN or FAIL. A WARN or FAIL line says what to do. Exit status 1 if any line is FAIL.
Python 3.8 or newer, standard library only. The Hugging Face token is read from HF_TOKEN, HF_TOKEN_PATH or
$HF_HOME/token, sent only to huggingface.co, and never printed.
"""
import argparse, ctypes, json, os, platform, re, runpy, shutil, subprocess, sys, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIB = 2 ** 30
# Measured on the test machine (docs and README): sizes in GB (10^9 bytes)
VENV_GB, SRC_GB = 5.9, 1.3          # trellis2/setup.sh env: venv and pinned sources
HF_DOWNLOAD_GB = 16.6               # trellis2/setup.sh weights: pinned Hugging Face downloads of the 3D stage
WEIGHTS_GB = 13.6                   # the local weight copy (run.py --install-weights)
TESTED = dict(nvcc="13.3", driver="610.43.02", gcc="14.2", uv="0.11.21", compute_cap="8.6", cargo="1.97.1")
IMAGE_CUDA = (13, 3)                # the container's base image starts only with a driver for CUDA 13.3 or newer
MIN_COMPUTE_CAP = 7.5               # CUDA 13 has no support for older GPUs (Maxwell, Pascal, Volta)
RAM_TESTED_GIB = 30                 # the test machine has 31 GB
RAM_ONE_RUNNER_GB = 19.4            # one runner: 3.6-6.4 GB plus 13 GB of memory-mapped weights (AGENTS.md "Memory")
CONFIG_VARS = ("T2_VENV", "T2_SRC", "T2_WEIGHTS", "HF_HOME", "CUDA_HOME")

results = []


def say(level, text, fix=""):
    results.append(level)
    print(f"{level:4}  {text}" + (f"  ->  {fix}" if fix else ""), flush=True)


def run(cmd, timeout=30):
    """(exit status, stdout) or (None, error text) when the program is missing or hangs."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, str(e)


def pipeline_constants():
    """run.py's pins (weights, models per pipeline type, DINOv3 repo): one source of truth. run.py is not executed."""
    return runpy.run_path(os.path.join(ROOT, "trellis2", "run.py"), run_name="preflight")


def check_os():
    system, machine = platform.system(), platform.machine()
    if system != "Linux":
        return say("FAIL", f"operating system: {system}", "use Linux; Windows and macOS are not supported")
    if machine != "x86_64":
        return say("WARN", f"Linux on {machine}", "only x86_64 is tested")
    say("PASS", f"Linux {platform.release()} on {machine}")


def driver_cuda_version(header):
    """The highest CUDA version the driver supports, from nvidia-smi's header (old and new header formats)."""
    m = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", header)
    return (int(m.group(1)), int(m.group(2))) if m else None


def profile_advice(vram_gib):
    """(level, text) for one GPU's memory, from the measured results in README.md and docs/LOW_VRAM.md."""
    if vram_gib >= 22:
        return "PASS", ("clean (the default) fits: 11 of 11 hard objects on 24 GB. A few very detailed objects ran out "
                        "of memory even on 24 GB")
    if vram_gib >= 15:
        return "WARN", ("16 GB class: clean ran 6 of 11 hard objects under a 15.70 GiB limit, draft 8 of 8. Start with "
                        "clean; after an out-of-memory error use --profile draft")
    if vram_gib >= 7.5:
        return "WARN", ("under 16 GB: draft ran 6 of 6 hard objects under a 7.63 GiB limit (an 8 GB card); clean under 16 GB "
                        "is not tested. Use --profile draft and read docs/LOW_VRAM.md")
    return "WARN", ("under 8 GB is not tested. Draft used 4.7-7.1 GiB on hard objects. Use --profile draft and read "
                    "docs/LOW_VRAM.md")


def check_gpus():
    """Returns the driver's CUDA version (major, minor) or None, and nvidia-smi's [(index, UUID, PCI bus id)]."""
    if not shutil.which("nvidia-smi"):
        say("FAIL", "nvidia-smi not found", "install the NVIDIA driver (it includes nvidia-smi)")
        return None, []
    code, header = run(["nvidia-smi"])
    cuda = driver_cuda_version(header) if code == 0 else None
    if cuda is None:
        say("FAIL", f"nvidia-smi does not work: {header.strip().splitlines()[0] if header.strip() else code}",
            "check the NVIDIA driver installation")
        return None, []
    code, out = run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,compute_cap,driver_version,uuid,"
                     "pci.bus_id", "--format=csv,noheader,nounits"])
    rows = [[c.strip() for c in line.split(",")] for line in out.strip().splitlines()] if code == 0 else []
    if not rows:
        say("FAIL", "no NVIDIA GPU found", "check that the GPU and its driver work: nvidia-smi")
        return None, []
    driver = rows[0][5]
    if cuda < (13, 0):
        say("FAIL", f"driver {driver} supports CUDA {cuda[0]}.{cuda[1]}",
            f"install a driver for CUDA 13 or newer (tested: {TESTED['driver']})")
    else:
        say("PASS", f"driver {driver}, supports CUDA {cuda[0]}.{cuda[1]}")
    for index, name, total, used, cap, *_ in rows:
        check_one_gpu(index, name, int(total) / 1024, int(used) / 1024, cap)
    smi = [(index, uuid, bus_key(bus)) for index, *_, uuid, bus in rows]
    check_gpu_numbers(cuda_devices(), smi)
    return cuda, smi


def bus_key(bus_id):
    """A PCI bus id in one form: nvidia-smi writes 00000000:01:00.0, CUDA writes 0000:01:00.0."""
    domain, rest = bus_id.strip().lower().split(":", 1)
    return f"{int(domain, 16):04x}:{rest}"


def uuid_text(raw):
    h = bytes(raw).hex()
    return f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def cuda_devices():
    """[(CUDA device number, UUID, PCI bus id)] in the order that CUDA programs such as PyTorch use with this
    environment (CUDA_DEVICE_ORDER, CUDA_VISIBLE_DEVICES). Read from the driver library; None if it cannot be read."""
    try:
        cu = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    count = ctypes.c_int()
    if cu.cuInit(0) or cu.cuDeviceGetCount(ctypes.byref(count)):
        return None
    get_uuid = getattr(cu, "cuDeviceGetUuid_v2", None) or cu.cuDeviceGetUuid
    devices = []
    for i in range(count.value):
        dev, raw, bus = ctypes.c_int(), (ctypes.c_ubyte * 16)(), ctypes.create_string_buffer(32)
        if cu.cuDeviceGet(ctypes.byref(dev), i) or get_uuid(raw, dev) or cu.cuDeviceGetPCIBusId(bus, len(bus), dev):
            return None
        devices.append((i, uuid_text(raw), bus_key(bus.value.decode())))
    return devices


def gpu_number_lines(cuda, smi, visible=None):
    """(level, text, fix) per CUDA device: which nvidia-smi GPU it is. smi: [(index, UUID, PCI bus id)]."""
    by_uuid = {uuid: index for index, uuid, _ in smi}
    lines = []
    for number, uuid, bus in cuda:
        index = by_uuid.get(uuid)
        text = f"CUDA device {number} is nvidia-smi GPU {index} ({uuid}, PCI {bus})"
        if index is None:
            lines.append(("WARN", f"CUDA device {number} ({uuid}, PCI {bus}) is not in nvidia-smi's list",
                          "check the driver installation"))
        elif str(number) == index:
            lines.append(("PASS", text, ""))
        elif visible is not None:
            lines.append(("PASS", f"{text}; CUDA_VISIBLE_DEVICES={visible} selects it", ""))
        else:
            lines.append(("WARN", f"{text}: the numbers differ",
                          "export CUDA_DEVICE_ORDER=PCI_BUS_ID, then CUDA uses nvidia-smi's numbers"))
    return lines


def check_gpu_numbers(cuda, smi):
    if cuda is None:
        return say("WARN", "CUDA device numbers not readable (libcuda.so.1)", "check the NVIDIA driver installation")
    for level, text, fix in gpu_number_lines(cuda, smi, os.environ.get("CUDA_VISIBLE_DEVICES")):
        say(level, text, fix)


def arch_list_runs_on(arch_list, cap):
    """True if CUDA code built for TORCH_CUDA_ARCH_LIST runs on a GPU of compute capability cap: machine code runs on the
    same major version with the same or a higher minor version; PTX ("+PTX") also runs on every later version; an
    arch-specific target ("9.0a") runs only on that version."""
    want = tuple(int(x) for x in cap.split("."))
    for entry in re.split(r"[;\s]+", arch_list.strip()):
        m = re.fullmatch(r"(\d+)\.(\d+)(a?)(\+PTX)?", entry)
        if not m:
            continue
        have = (int(m.group(1)), int(m.group(2)))
        if m.group(3):
            if have == want:
                return True
        elif (have[0] == want[0] and have[1] <= want[1]) or (m.group(4) and have <= want):
            return True
    return False


def check_one_gpu(index, name, total_gib, used_gib, cap):
    label = f"GPU {index} {name}, {total_gib:.1f} GiB, compute capability {cap}"
    if not re.fullmatch(r"\d+\.\d+", cap):
        return say("WARN", label, "compute capability unknown: check that CUDA 13 supports this GPU")
    if float(cap) < MIN_COMPUTE_CAP:
        return say("FAIL", label, f"CUDA 13 does not support GPUs under compute capability {MIN_COMPUTE_CAP}")
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    if arch_list and not arch_list_runs_on(arch_list, cap):
        say("FAIL", f"GPU {index}: the CUDA code is built for TORCH_CUDA_ARCH_LIST={arch_list}, which does not run on "
            f"compute capability {cap}", f"build again with {cap} in TORCH_CUDA_ARCH_LIST (docs/CONTAINER.md)")
    if cap != TESTED["compute_cap"]:
        say("WARN", f"GPU {index}: compute capability {cap}",
            f"this GPU generation is not tested (tested: {TESTED['compute_cap']}, RTX 3090 and 3090 Ti)")
    level, text = profile_advice(total_gib)
    say(level, f"{label}: {text}")
    if used_gib > 1:
        say("WARN", f"GPU {index}: {used_gib:.1f} GiB in use by other programs now",
            "stop them before you make objects, or use another GPU (CUDA_VISIBLE_DEVICES)")


def check_config():
    missing = [v for v in CONFIG_VARS if not os.environ.get(v)]
    if missing:
        say("FAIL", f"not set: {' '.join(missing)}", "cp config.example.env config.env, edit the paths, then: source config.env")
        return False
    say("PASS", "config: " + ", ".join(f"{v}={os.environ[v]}" for v in CONFIG_VARS))
    return True


def venv_state(venv):
    """'built', 'absent' or 'incomplete'."""
    if not os.path.exists(venv):
        return "absent"
    built = os.path.isfile(os.path.join(venv, "bin", "python")) and os.path.isfile(os.path.join(venv, "cumesh-build.json"))
    return "built" if built else "incomplete"


def check_venv(venv):
    state = venv_state(venv)
    if state == "built":
        say("PASS", f"3D environment built: {venv}")
    elif state == "incomplete":
        say("FAIL", f"{venv} exists but is not a finished build",
            "setup builds the environment only from scratch: remove that folder yourself, then run setup.sh env again")
    else:
        say("PASS", f"3D environment not built yet (next: setup.sh env): {venv}")
    return state


def version_of(cmd, pattern):
    code, out = run(cmd)
    m = re.search(pattern, out) if code == 0 else None
    return m.group(1) if m else None


def check_build_tools(driver_cuda):
    """Only needed by setup.sh env (it compiles CUDA code)."""
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    nvcc = version_of([os.path.join(cuda_home, "bin", "nvcc"), "--version"], r"release (\d+\.\d+)")
    if nvcc is None:
        say("FAIL", f"no nvcc in {cuda_home}/bin", f"install the CUDA toolkit 13.x (tested {TESTED['nvcc']}) and set CUDA_HOME")
    elif not nvcc.startswith("13."):
        say("FAIL", f"nvcc {nvcc} in {cuda_home}", f"setup needs CUDA toolkit 13.x to match torch's CUDA 13.0 (tested {TESTED['nvcc']})")
    elif driver_cuda and tuple(int(x) for x in nvcc.split(".")) > driver_cuda:
        say("WARN", f"nvcc {nvcc} is newer than the driver's CUDA {driver_cuda[0]}.{driver_cuda[1]}",
            "update the driver, or use a toolkit not newer than the driver")
    else:
        say("PASS", f"nvcc {nvcc} in {cuda_home}" + ("" if nvcc == TESTED["nvcc"] else f" (tested: {TESTED['nvcc']})"))
    gcc = version_of(["g++", "--version"], r"(\d+\.\d+)\.\d+")
    if gcc is None:
        say("FAIL", "no C++ compiler (g++)", "install your distribution's build tools, for example: build-essential")
    else:
        say("PASS", f"g++ {gcc} (tested: {TESTED['gcc']})")
    if not shutil.which("git"):
        say("FAIL", "git not found", "install git (setup clones pinned sources)")
    uv = os.environ.get("UV", "uv")
    uv_version = version_of([uv, "--version"], r"uv (\S+)")
    if uv_version is None:
        say("FAIL", f"uv not found ({uv})", "install uv (https://docs.astral.sh/uv/) or set UV to its path")
    else:
        say("PASS", f"uv {uv_version} (tested: {TESTED['uv']})")
    check_trianglebvh_build()


def check_trianglebvh_build():
    """setup.sh env builds trianglebvh (Rust) from TRIANGLEBVH_GIT with cargo ($CARGO, PATH or ~/.cargo/bin)."""
    cargo = os.environ.get("CARGO") or shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    cargo_version = version_of([cargo, "--version"], r"cargo (\S+)")
    if cargo_version is None:
        say("FAIL", "no Rust toolchain (cargo)", "install Rust (https://rustup.rs) or set CARGO to the cargo program")
    else:
        say("PASS", f"cargo {cargo_version} (tested: {TESTED['cargo']})")
    if not os.environ.get("TRIANGLEBVH_GIT"):
        say("PASS", "TRIANGLEBVH_GIT not set: setup.sh uses https://github.com/memgrafter/trianglebvh.git")
    else:
        say("PASS", f"TRIANGLEBVH_GIT={os.environ['TRIANGLEBVH_GIT']}")


def missing_weight_files(weights, pins):
    return [rel for rel in pins["WEIGHT_FILES"] if not os.path.isfile(os.path.join(weights, rel))]


def check_weights(weights, pins):
    """Returns True if every pinned weight file is in place (SHA-256 is checked by run.py at each start)."""
    missing = missing_weight_files(weights, pins)
    if missing:
        say("PASS", f"weights not installed yet ({len(missing)} of {len(pins['WEIGHT_FILES'])} files missing; "
                    f"next: setup.sh weights): {weights}")
        return False
    say("PASS", f"weights installed: {weights}")
    check_profiles(weights, pins)
    return True


def profile_model_files(weights, pins):
    """{pipeline type: [model files it needs that are not in the weights folder]} from pipeline.json and run.py's NEEDED."""
    t2 = os.path.join(weights, pins["T2_DIR"])
    models = json.load(open(os.path.join(t2, "pipeline.json")))["args"]["models"]
    return {ptype: [models[n] for n in names if not os.path.isfile(os.path.join(t2, models[n] + ".safetensors"))]
            for ptype, names in pins["NEEDED"].items()}


def check_profiles(weights, pins):
    pinned = {rel for rel in pins["WEIGHT_FILES"]}
    for ptype, absent in profile_model_files(weights, pins).items():
        name = {"512": "draft", "1024_cascade": "clean", "1536_cascade": "best"}.get(ptype, ptype)
        if not absent:
            say("PASS", f"profile {name} ({ptype}): all model files present")
            continue
        in_setup = all(f"{pins['T2_DIR']}/{a}.safetensors" in pinned for a in absent)
        fix = ("run setup.sh weights again" if in_setup else
               "this version of setup does not install it yet; use the clean profile")
        say("WARN", f"profile {name} ({ptype}) not available: {', '.join(absent)} not in the weights folder", fix)


def hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    path = os.environ.get("HF_TOKEN_PATH") or os.path.join(os.environ.get("HF_HOME", ""), "token")
    if os.path.isfile(path):
        return open(path).read().strip()
    return ""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect means access; never follow it, so the token is not sent to another host."""
    def redirect_request(self, *args, **kwargs):
        return None


def check_dinov3_access(pins, how_to_login):
    repo, rev = pins["DINO_REPO"], pins["DINO_REV"]
    page = f"https://huggingface.co/{repo}"
    token = hf_token()
    if not token:
        return say("WARN", "no Hugging Face login yet (needed before setup.sh weights)", f"log in: {how_to_login} (or set HF_TOKEN)")
    req = urllib.request.Request(f"{page}/resolve/{rev}/config.json", method="HEAD",
                                 headers={"Authorization": f"Bearer {token}"})
    try:
        status = urllib.request.build_opener(NoRedirect).open(req, timeout=20).status
    except urllib.error.HTTPError as e:
        status, message = e.code, (e.headers.get("X-Error-Message") or "")[:200]
        if status in (301, 302, 303, 307, 308):
            return say("PASS", f"Hugging Face login works and has access to {repo}")
        if status == 401:
            return say("FAIL", f"Hugging Face does not accept the token ({message})", f"log in again: {how_to_login}")
        if status == 403:
            return say("FAIL", f"your Hugging Face account has no access to {repo} yet ({message})",
                       f"open {page}, read the DINOv3 License and accept it yourself if you agree; access can wait for approval")
        return say("WARN", f"Hugging Face answered {status} for {repo}", "try again later")
    except (urllib.error.URLError, OSError) as e:
        return say("WARN", f"could not reach huggingface.co ({getattr(e, 'reason', e)})", "check the network, then run this again")
    say("PASS" if status == 200 else "WARN", f"Hugging Face login works and has access to {repo} (HTTP {status})")


def existing_parent(path):
    path = os.path.abspath(path)
    while not os.path.exists(path):
        path = os.path.dirname(path)
    return path


def free_gb(path):
    st = os.statvfs(existing_parent(path))
    return st.f_bavail * st.f_frsize / 1e9


def check_disk(items, venv=None, weights=None):
    """items: [(label, path, GB still to write)]. One line per filesystem, then setup.sh's own free-space floors."""
    by_fs = {}
    for label, path, gb in items:
        dev = os.stat(existing_parent(path)).st_dev
        by_fs.setdefault(dev, []).append((label, path, gb))
    for group in by_fs.values():
        need = sum(gb for _, _, gb in group)
        free = free_gb(group[0][1])
        names = ", ".join(f"{label} {gb:.1f} GB" for label, _, gb in group if gb) or "nothing new"
        where = existing_parent(group[0][1])
        if free < need:
            say("FAIL", f"disk at {where}: {free:.1f} GB free, needs {need:.1f} GB ({names})", "free space or set other paths")
        else:
            say("PASS", f"disk at {where}: {free:.1f} GB free, needs {need:.1f} GB ({names})")
    check_setup_floors(items, venv, weights)


def check_setup_floors(items, venv, weights):
    min_venv = float(os.environ.get("SETUP_MIN_FREE_GB", "20"))
    min_weights = float(os.environ.get("T2_WEIGHTS_MIN_FREE_GB", "20"))
    if venv and not os.path.exists(venv) and free_gb(os.path.dirname(os.path.abspath(venv))) <= min_venv:
        say("FAIL", f"setup.sh env stops when {min_venv:.0f} GB or less is free next to {venv}",
            "free space, or set T2_VENV (and SETUP_MIN_FREE_GB) yourself")
    if not weights or not any(label == "weights" and gb for label, _, gb in items):
        return
    dev = os.stat(existing_parent(weights)).st_dev
    need_on_fs = sum(gb for _, path, gb in items if os.stat(existing_parent(path)).st_dev == dev)
    left = free_gb(weights) - need_on_fs
    if left < min_weights:
        say("FAIL", f"after setup, {left:.1f} GB would be free at {existing_parent(weights)}; "
                    f"setup.sh weights stops under {min_weights:.0f} GB", "free space or set T2_WEIGHTS to another disk")


def check_ram():
    info = {}
    for line in open("/proc/meminfo"):
        key, value = line.split(":", 1)
        info[key] = int(value.split()[0]) * 1024
    total, avail = info["MemTotal"] / GIB, info.get("MemAvailable", 0) / GIB
    text = f"RAM {total:.1f} GiB, {avail:.1f} GiB available"
    if total * GIB / 1e9 < RAM_ONE_RUNNER_GB:
        say("WARN", text, f"under one runner's measured need (about {RAM_ONE_RUNNER_GB} GB); not tested")
    elif total < RAM_TESTED_GIB:
        say("WARN", text, "tested only with 31 GB; one runner used 3.6-6.4 GB plus 13 GB of memory-mapped weights")
    else:
        say("PASS", text)
    min_mb = int(os.environ.get("SETUP_MIN_RAM_MB", "3000"))
    if avail * 1024 < min_mb:
        say("WARN", f"only {avail * 1024:.0f} MB RAM available", f"setup's RAM guard stops the build under {min_mb} MB")


def check_container_host(driver_cuda):
    if driver_cuda and driver_cuda < IMAGE_CUDA:
        say("FAIL", f"the driver supports CUDA {driver_cuda[0]}.{driver_cuda[1]}; the container needs CUDA "
                    f"{IMAGE_CUDA[0]}.{IMAGE_CUDA[1]} or newer", "update the NVIDIA driver")
    if not shutil.which("docker"):
        return say("FAIL", "docker not found", "install Docker Engine, or use the setup without a container")
    code, out = run(["docker", "info", "--format", "{{json .Runtimes}} {{.DockerRootDir}}"])
    if code != 0:
        return say("FAIL", f"docker does not answer: {out.strip().splitlines()[-1] if out.strip() else code}",
                   "start Docker, and check that your user may use it")
    runtimes, root = out.rsplit(" ", 1)
    toolkit = "nvidia" in runtimes or shutil.which("nvidia-ctk") or shutil.which("nvidia-container-cli")
    if toolkit:
        say("PASS", "NVIDIA Container Toolkit found")
    else:
        say("FAIL", "NVIDIA Container Toolkit not found", "install it, so that docker run --gpus all works")
    root = root.strip()
    try:
        say("WARN", f"Docker's disk at {root}: {free_gb(root):.1f} GB free", "the image size is not measured yet")
    except OSError:
        say("WARN", f"Docker's disk at {root}: free space not readable as your user", "check it; the image size is not measured yet")


def check_image_gpu_numbers(image, smi):
    """Runs the image with all GPUs and compares its CUDA device numbers with the host's nvidia-smi numbers."""
    code, out = run(["docker", "run", "--rm", "--gpus", "all", image, "python", "-I", "tools/preflight.py",
                     "--cuda-devices"], timeout=300)
    try:
        cuda = json.loads(out.strip().splitlines()[-1]) if code == 0 else None
    except (ValueError, IndexError):
        cuda = None
    if cuda is None:
        last = out.strip().splitlines()[-1] if out and out.strip() else code
        return say("FAIL", f"could not read the GPUs in {image}: {last}",
                   "check: docker run --rm --gpus all " + image + " nvidia-smi")
    for level, text, fix in gpu_number_lines(cuda, smi):
        say(level, f"in {image}: {text.replace('nvidia-smi GPU', 'host GPU')}",
            fix and "build the image again from this repository's Dockerfile (it sets CUDA_DEVICE_ORDER=PCI_BUS_ID)")


def main():
    ap = argparse.ArgumentParser(description="Check this machine before setup and before the first object.")
    ap.add_argument("--container", action="store_true", help="check the host of the container instead of a local setup")
    ap.add_argument("--data", default="data", help="--container: the folder you mount at /data (Hugging Face cache, weights)")
    ap.add_argument("--image", help="--container: also run this image and check that its GPU numbers are the host's")
    ap.add_argument("--cuda-devices", action="store_true", help="print only the CUDA devices as JSON (number, UUID, PCI bus id)")
    a = ap.parse_args()
    if a.cuda_devices:
        devices = cuda_devices()
        print(json.dumps(devices))
        return 0 if devices is not None else 1
    pins = pipeline_constants()
    check_os()
    if platform.system() != "Linux":
        return 1
    driver_cuda, smi = check_gpus()
    check_ram()
    if a.container:
        check_container_host(driver_cuda)
        if a.image:
            check_image_gpu_numbers(a.image, smi)
        os.environ.setdefault("HF_HOME", os.path.join(a.data, "hf"))
        weights = os.path.join(a.data, "weights")
        have_weights = check_weights(weights, pins)
        check_disk([("Hugging Face downloads", os.environ["HF_HOME"], 0 if have_weights else HF_DOWNLOAD_GB),
                    ("weights", weights, 0 if have_weights else WEIGHTS_GB)], weights=weights)
        if not have_weights:
            check_dinov3_access(pins, "in the container: hf auth login")
    elif check_config():
        venv, weights = os.environ["T2_VENV"], os.environ["T2_WEIGHTS"]
        state = check_venv(venv)
        if state != "built":
            check_build_tools(driver_cuda)
            say("WARN", "uv's download cache (UV_CACHE_DIR, default ~/.cache/uv) also grows during setup; its size is not measured",
                "keep extra free space there")
        have_weights = check_weights(weights, pins)
        check_disk([("venv", venv, 0 if state != "absent" else VENV_GB),
                    ("sources", os.environ["T2_SRC"], 0 if os.path.exists(os.environ["T2_SRC"]) else SRC_GB),
                    ("Hugging Face downloads", os.environ["HF_HOME"], 0 if have_weights else HF_DOWNLOAD_GB),
                    ("weights", weights, 0 if have_weights else WEIGHTS_GB)], venv=venv, weights=weights)
        if not have_weights:
            check_dinov3_access(pins, "$T2_VENV/bin/hf auth login (after setup.sh env)")
    fails, warns = results.count("FAIL"), results.count("WARN")
    print(f"\n{fails} FAIL, {warns} WARN, {results.count('PASS')} PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
