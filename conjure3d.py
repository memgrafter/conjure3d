"""Make one 3D object from one image with one command.

  source config.env
  python3 -I conjure3d.py IMAGE SIZE_M [--profile clean|draft|best] [--seed 42] [--out out] [--id NAME] [--lie-flat]
                        [--on-oom stop|draft] [--chunk-norm]

SIZE_M is the real length of the object's longest side in metres. The command writes a props file, runs
trellis2/run.py (cut-out, TRELLIS.2, GLB export) and post/post_step.py (real size, rest pose), and prints the path of
asset.glb. Files: OUT/<id>-<profile>-seed<N>/<id>/ (asset.glb, raw.glb, cut.png, record.json, asset.json), logs in
OUT/<id>-<profile>-seed<N>/.

The profile goes to run.py (--profile; its values are run.py's PROFILES). clean is run.py's default, so its files are
the same bytes as a plain run.py run with the same image and seed. draft: 512 pipeline, 1024 px textures, 100k faces
(less GPU memory, a rougher surface, other bytes). best: 1536_cascade (more GPU memory; see README).
--chunk-norm: run.py --chunk-norm CHUNK_NORM_ROWS (lower GPU peak in the decoders, same bytes); folder name gets
"-chunk-norm". Out of GPU memory with clean: --on-oom stop (default) prints the --chunk-norm command (when the
generation ran out and --chunk-norm was not used) and the draft command; --on-oom draft makes the draft at once in its
own folder. Python 3.8 or newer, standard library only; run.py and the post step run in $T2_VENV.
--backend trellis-cpp: the Apple Silicon route (EXPERIMENTAL, not production, not byte-identical to the CUDA route):
mac/run_mac.py (the same BiRefNet cut-out on MPS, then trellis.cpp with Metal) and the same post step, both in
$MAC_VENV; profiles draft and clean only; folder OUT/<id>-<profile>-trellis-cpp-seed<N>/. See docs/MAC.md.
"""
import argparse, importlib.util, json, os, re, shlex, signal, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
PROFILES = ("clean", "draft", "best")   # run.py --profile; the settings are run.py's PROFILES
RUN_ENV = {"HF_HUB_OFFLINE": "1", "ATTN_BACKEND": "xformers", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
EXIT_OK, EXIT_ERROR, EXIT_OOM = 0, 1, 3
CHUNK_NORM_ROWS = 1048576   # the row count proven byte-identical (docs/LOW_VRAM.md)
MAC_ENV = {"HF_HUB_OFFLINE": "1"}
MAC_PROFILES = ("clean", "draft")   # mac/run_mac.py PROFILES


class Stop(Exception):
    """A plain message for the user; the command ends with EXIT_ERROR."""


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Make one 3D object (asset.glb) from one image.")
    ap.add_argument("image", help="a picture of one whole object on a plain background (PNG or JPEG)")
    ap.add_argument("size_m", type=float, help="the real length of the object's longest side, in metres (e.g. 0.35)")
    ap.add_argument("--profile", default="clean", choices=sorted(PROFILES), help="clean (default), draft or best")
    ap.add_argument("--seed", type=int, default=42, help="other seeds give other shapes (default 42)")
    ap.add_argument("--out", default="out", help="output folder (default: out)")
    ap.add_argument("--id", default="", help="object name (default: from the image file name)")
    ap.add_argument("--lie-flat", action="store_true", help="the object lies on its back with the image's front up (a watch, a knife)")
    ap.add_argument("--on-oom", default="stop", choices=["stop", "draft"],
                    help="if clean runs out of GPU memory: stop and print the draft command (default), or make a draft")
    ap.add_argument("--chunk-norm", action="store_true",
                    help="run.py --chunk-norm: lower GPU peak in the decoders, same files; for a clean run that ran out of memory")
    ap.add_argument("--backend", default="cuda", choices=["cuda", "trellis-cpp"],
                    help="cuda (default: Linux, NVIDIA) or trellis-cpp (Apple Silicon, experimental; docs/MAC.md)")
    a = ap.parse_args(argv)
    if a.backend == "trellis-cpp" and a.profile not in MAC_PROFILES:
        ap.error(f"--backend trellis-cpp supports --profile {' or '.join(MAC_PROFILES)}")
    if a.backend == "trellis-cpp" and a.chunk_norm:
        ap.error("--chunk-norm is for the CUDA route (trellis.cpp has no such option)")
    if not a.size_m > 0:
        ap.error("SIZE_M must be more than 0 (metres)")
    a.image, a.out = os.path.abspath(a.image), os.path.abspath(a.out)
    a.id = a.id or object_id(a.image)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", a.id):
        ap.error(f"--id {a.id!r}: use letters, digits, '-', '_' and '.'")
    return a


def object_id(image):
    stem = os.path.splitext(os.path.basename(image))[0]
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:64] or "object"


def load_preflight():
    spec = importlib.util.spec_from_file_location("preflight", os.path.join(ROOT, "tools", "preflight.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_setup(profile):
    """Stops with a plain message if setup is not finished or the profile's model files are missing."""
    missing = [v for v in ("T2_VENV", "T2_WEIGHTS", "HF_HOME") if not os.environ.get(v)]
    if missing:
        raise Stop(f"not set: {' '.join(missing)}. Run: source config.env (see README.md, First object)")
    python = os.path.join(os.environ["T2_VENV"], "bin", "python")
    if not os.path.isfile(python):
        raise Stop(f"no 3D environment at {os.environ['T2_VENV']}. Run: python3 -I tools/preflight.py, then setup.sh")
    preflight = load_preflight()
    pins = preflight.pipeline_constants()
    weights = os.environ["T2_WEIGHTS"]
    if preflight.missing_weight_files(weights, pins):
        raise Stop(f"the weights are not installed in {weights}. Run: setsid -w bash trellis2/setup.sh weights")
    absent = preflight.profile_model_files(weights, pins)[pins["PROFILES"][profile]["pipeline_type"]]
    if absent:
        raise Stop(f"the {profile} profile needs model files that are not in {weights}: {', '.join(absent)}. "
                   "Run: setsid -w bash trellis2/setup.sh weights")
    return python


def check_setup_mac():
    """The Apple route: stops with a plain message if mac/setup_mac.sh (build, env, gguf) is not finished."""
    names = ("MAC_VENV", "TRELLIS_CPP_HOME", "TRELLIS_GGUF", "T2_SRC", "HF_HOME")
    missing = [v for v in names if not os.environ.get(v)]
    if missing:
        raise Stop(f"not set: {' '.join(missing)}. Run: source config.env (see docs/MAC.md)")
    python = os.path.join(os.environ["MAC_VENV"], "bin", "python")
    for path, step in ((python, "env"), (os.path.join(os.environ["TRELLIS_CPP_HOME"], "build", "trellis-cli"), "build"),
                       (os.path.join(os.environ["TRELLIS_GGUF"], "manifest.json"), "gguf")):
        if not os.path.isfile(path):
            raise Stop(f"no {path}. Run: bash mac/setup_mac.sh {step} (docs/MAC.md)")
    return python


def write_props(job, a):
    prop = dict(gold=os.path.basename(a.image), size_m=a.size_m)
    if a.lie_flat:
        prop.update(rest_up=[0, 0, 1], rest_up_why="--lie-flat: the image's front faces up")
    path = os.path.join(job, "props.json")
    with open(path, "w") as f:
        json.dump({"props": {a.id: prop}}, f, indent=1)
    return path


def tee(cmd, log_path, env=None):
    """Runs cmd; its output goes to the terminal and to log_path. Returns the exit status. If this command is stopped
    (Ctrl-C, SIGTERM from a guard), the child is stopped too."""
    print("$ " + shlex.join(cmd), flush=True)
    with open(log_path, "ab") as log:
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for chunk in iter(lambda: p.stdout.read1(65536), b""):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                log.write(chunk)
            return p.wait()
        finally:
            if p.poll() is None:
                p.terminate()
                p.wait()


def oom_file(od):
    names = sorted(n for n in os.listdir(od) if n.startswith("oom-") and n.endswith(".json")) if os.path.isdir(od) else []
    return os.path.join(od, names[0]) if names else None


def make(a, profile, python):
    """Runs run.py and the post step for one profile. Returns (asset.glb path or None, OOM file or None, job folder)."""
    mac = a.backend == "trellis-cpp"
    job = os.path.join(a.out, f"{a.id}-{profile}{'-chunk-norm' if a.chunk_norm else ''}{'-trellis-cpp' if mac else ''}-seed{a.seed}")
    for path in (os.path.dirname(a.image), job):
        if ":" in path:
            raise Stop(f"{path}: run.py's --job cannot read a ':' in a path; move or rename it")
    if os.path.exists(job):
        raise Stop(f"{job} already exists. Use another --out or --seed, or remove that folder yourself")
    os.makedirs(job)
    props = write_props(job, a)
    od = os.path.join(job, a.id)
    runner = os.path.join(ROOT, "mac", "run_mac.py") if mac else os.path.join(ROOT, "trellis2", "run.py")
    run_cmd = [python, "-I", runner, "--job", f"{props}:{os.path.dirname(a.image)}:{job}",
               "--only", a.id, "--seed", str(a.seed), "--profile", profile]
    if a.chunk_norm:
        run_cmd += ["--chunk-norm", str(CHUNK_NORM_ROWS)]
    t0 = time.time()
    code = tee(run_cmd, os.path.join(job, "run.log"), env=dict(os.environ, **(MAC_ENV if mac else RUN_ENV)))
    run_s = time.time() - t0
    if not os.path.isfile(os.path.join(od, "raw.glb")):
        oom = oom_file(od)
        if oom:
            return None, oom, job
        cut_error = os.path.join(od, "cutout-error.json")
        if os.path.isfile(cut_error):
            raise Stop(f"the cut-out failed ({json.load(open(cut_error))['error']}). Use another image: one whole object "
                       "on a plain background")
        raise Stop(f"run.py stopped (exit status {code}) and made no raw.glb. See {os.path.join(job, 'run.log')}")
    t1 = time.time()
    code = tee([python, "-I", os.path.join(ROOT, "post", "post_step.py"), "--props", props, "--dir", job, "--only", a.id],
               os.path.join(job, "post.log"))
    asset = os.path.join(od, "asset.glb")
    if code != 0 or not os.path.isfile(asset):
        raise Stop(f"the post step stopped (exit status {code}). See {os.path.join(job, 'post.log')}")
    report(a, profile, od, run_s, time.time() - t1)
    return asset, None, job


def report(a, profile, od, run_s, post_s):
    rec = json.load(open(os.path.join(od, "record.json")))
    info = json.load(open(os.path.join(od, "asset.json")))
    print(f"\nDONE  {os.path.join(od, 'asset.glb')}")
    print(f"      profile {profile}, seed {a.seed}, longest side {a.size_m} m, pose {info['pose']['chosen']}, "
          f"GPU {rec['gpu']}")
    if rec.get("backend") == "trellis-cpp":
        print(f"      time: 3D {run_s:.0f} s (trellis.cpp {rec['seconds']['trellis_cpp_wall']:.0f} s), post step {post_s:.0f} s")
        print("NOTE  Apple Silicon route (trellis.cpp): experimental, not production, not byte-identical to the CUDA route.")
    else:
        print(f"      time: 3D {run_s:.0f} s (model load {rec['load']['load_s']:.0f} s included), post step {post_s:.0f} s")
    if rec.get("export_note", "as configured") != "as configured":
        print("WARN  the GLB export ran out of GPU memory once and was made again from a simplified mesh. This file "
              "cannot be made again byte for byte from the image and seed (record.json, export_note).")
    if not info.get("checks", {}).get("passed"):
        print(f"WARN  the post step's checks did not pass. See {os.path.join(od, 'asset.json')}")
    print("LOOK  Open asset.glb in Blender (File > Import > glTF 2.0) or another glTF 2.0 viewer. Units are metres, "
          "Y is up.\n      Check cut.png in the same folder: it must show the whole object and no background.")


def oom_message(oom, profile):
    stage = "the GLB export" if oom.endswith("oom-glb.json") else "generation"
    return (f"OUT OF GPU MEMORY in {stage} ({oom}). With the {profile} profile this object needs more GPU memory than "
            "this GPU has free.")


def main(argv=None):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))   # stops the child in tee()
    a = parse_args(argv)
    try:
        if not os.path.isfile(a.image):
            raise Stop(f"no image at {a.image}")
        python = check_setup_mac() if a.backend == "trellis-cpp" else check_setup(a.profile)
        asset, oom, job = make(a, a.profile, python)
        if not oom:
            return EXIT_OK
        print("\n" + oom_message(oom, a.profile))
        if a.profile != "clean":
            print("Close other programs that use the GPU, or use a GPU with more memory (docs/LOW_VRAM.md).")
            return EXIT_OOM
        same = [sys.executable, "-I", os.path.abspath(__file__), a.image, str(a.size_m), "--seed", str(a.seed), "--out", a.out,
                "--id", a.id] + (["--lie-flat"] if a.lie_flat else []) + (["--backend", a.backend] if a.backend != "cuda" else [])
        draft = same + ["--profile", "draft"]
        if a.on_oom == "stop":
            if a.backend == "cuda" and not a.chunk_norm and not oom.endswith("oom-glb.json"):
                print("--chunk-norm lowers the GPU peak in generation and gives the same files. With a 15.70 GiB limit it "
                      "made 3 of 4 objects that ran out of memory without it (docs/LOW_VRAM.md). To try it first:\n  "
                      f"{shlex.join(same + ['--chunk-norm'])}")
            print("The draft profile needs less GPU memory. It gives a rougher surface and different files. To make a "
                  f"draft:\n  {shlex.join(draft)}")
            return EXIT_OOM
        print("You asked for a draft on out-of-memory (--on-oom draft). Making a draft now: rougher surface, different "
              f"files, in its own folder. The clean attempt stays in {job}.\n", flush=True)
        if a.backend == "cuda":
            check_setup("draft")
        a.chunk_norm = False   # the draft is the same command as the printed one
        asset, oom, _ = make(a, "draft", python)
        if oom:
            print("\n" + oom_message(oom, "draft"))
            return EXIT_OOM
        return EXIT_OK
    except Stop as e:
        print(f"STOP  {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
