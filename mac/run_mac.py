"""Apple Silicon route (EXPERIMENTAL, not production): image -> cut-out -> trellis.cpp (Metal) -> raw.glb + record.json.

  source config.env
  $MAC_VENV/bin/python -I mac/run_mac.py --job PROPS.json:GOLD_ROOT:OUT [--only a,b] [--seed 42] [--profile draft|clean]
                                         [--device mps|cpu]

Per object (the same files as trellis2/run.py, so post/post_step.py runs unchanged):
1. cut-out: run.py's BiRefNetCutout (the pinned BiRefNet code and weights, the largest-piece rule) on --device (mps by
   default, cpu if MPS is missing) -> OUT/<id>/cut.png (RGBA), mask-birefnet.png, mask.png.
2. trellis.cpp (a C++/GGML port of TRELLIS.2, MIT; TRELLIS_CPP_COMMIT, built by mac/setup_mac.sh with Metal) on cut.png
   (it keeps the alpha: its own background remover is not used) with the GGUF files that mac/convert_weights.py made
   from our pinned weights -> OUT/<id>/trellis/ (its GLB, base-colour PNG, raw-mesh PLY, log). Its GLB frame is
   TRELLIS.2's (glTF Y up, front +Z, inside [-0.5, 0.5]^3), so it is copied to OUT/<id>/raw.glb unchanged.
3. record.json: backend, pins (trellis.cpp commit and binary SHA-256, every GGUF file's SHA-256 and source), settings,
   seed, times, peak memory (the trellis.cpp process's maximum resident set size; Apple GPUs share this memory).
Profiles: draft = trellis.cpp --res 512, 1024 px textures; clean = --res 1024 (its 512 -> 1024 cascade), 4096 px
textures. trellis.cpp fixes the face count itself (150,000 at 512, 300,000 at 1024): not our 100k / 500k.
Not byte-identical to the CUDA route and not tested for production: other code (C++ port, f16 weights, its own remesh,
QEM decimation, xatlas and bake), other hardware. trellis.cpp names no nvdiffrast, cubvh or instant-ngp parts; its BVH
and UV raster are its own (docs/MAC.md).
"""
import argparse, hashlib, json, os, platform, re, shutil, subprocess, sys, time
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
TRELLIS_CPP_REPO = "https://github.com/pwilkin/trellis.cpp"
TRELLIS_CPP_COMMIT = "c0bed38c1578f7e36e3e50c8ff1e38fa0d47583f"   # tag v0.8.1, 2026-09-25 (MIT)
PROFILES = {"draft": dict(res=512, atlas=1024, faces=150000), "clean": dict(res=1024, atlas=4096, faces=300000)}
GGUF_NEEDED = {512: ["ss_flow", "ss_dec", "shape_flow_512", "tex_flow_512", "shape_dec", "tex_dec", "dinov3"],
               1024: ["ss_flow", "ss_dec", "shape_flow_512", "shape_flow_1024", "tex_flow_1024", "shape_dec", "tex_dec",
                      "dinov3"]}
EXIT_OOM = 3


def load_run():
    """trellis2/run.py as a module (its top level is the standard library only): BiRefNetCutout and the pins."""
    spec = importlib.util.spec_from_file_location("t2run", os.path.join(os.path.dirname(HERE), "trellis2", "run.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def machine():
    sysctl = lambda k: subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
    return dict(chip=sysctl("machdep.cpu.brand_string") or platform.processor(), memory_gb=round(int(sysctl("hw.memsize") or 0) / 1e9, 1),
                macos=platform.mac_ver()[0], memory_note="unified: CPU and GPU share this memory")


def trellis_cpp_info(home):
    """Stops unless the clone is the clean pinned commit and the binary exists; -> pins for record.json."""
    src, cli = os.path.join(home, "src"), os.path.join(home, "build", "trellis-cli")
    git = lambda *a: subprocess.run(["git", "-C", src, *a], capture_output=True, text=True).stdout.strip()
    if git("rev-parse", "HEAD") != TRELLIS_CPP_COMMIT or git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit(f"STOP: {src} is not the clean trellis.cpp commit {TRELLIS_CPP_COMMIT}; run mac/setup_mac.sh build")
    if not os.access(cli, os.X_OK):
        raise SystemExit(f"STOP: no {cli}; run mac/setup_mac.sh build")
    build = os.path.join(home, "build", "conjure3d-build.json")
    return dict(repo=TRELLIS_CPP_REPO, commit=TRELLIS_CPP_COMMIT, ggml_commit=git("-C", "thirdparty/ggml", "rev-parse", "HEAD"),
                cli_sha256=sha256(cli), build=json.load(open(build)) if os.path.isfile(build) else None), cli


def gguf_info(gdir, res):
    """Stops unless every GGUF file this resolution loads is listed in manifest.json with its SHA-256."""
    man = json.load(open(os.path.join(gdir, "manifest.json")))
    out = {}
    for c in GGUF_NEEDED[res]:
        p, want = os.path.join(gdir, f"{c}.gguf"), man.get(f"{c}.gguf", {}).get("sha256")
        if not want or not os.path.isfile(p) or sha256(p) != want:
            raise SystemExit(f"STOP: {p} is missing or not the file in manifest.json; run mac/setup_mac.sh gguf")
        out[f"{c}.gguf"] = dict(sha256=want, sources=man[f"{c}.gguf"]["sources"])
    if not os.path.isfile(os.path.join(gdir, "DINOv3-LICENSE.md")):
        raise SystemExit(f"STOP: {gdir}/DINOv3-LICENSE.md is missing (the DINOv3 License travels with its weights)")
    return out


def peak_rss_bytes(time_output):
    """/usr/bin/time -l on macOS: 'NNN  maximum resident set size' in bytes."""
    m = re.search(r"(\d+)\s+maximum resident set size", time_output)
    return int(m.group(1)) if m else None


def run_trellis(cli, cut, out_glb, gdir, seed, prof, log_path):
    cmd = ["/usr/bin/time", "-l", cli, cut, out_glb, "--models", gdir, "--seed", str(seed), "--res", str(prof["res"]),
           "--atlas", str(prof["atlas"]), "--webp", "off", "--require-gpu", "--verbose"]
    t = time.time()
    with open(log_path, "w") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    wall = time.time() - t
    text = open(log_path, errors="replace").read()
    return p.returncode, round(wall, 1), peak_rss_bytes(text), text, cmd[2:]


def glb_stats(path):
    import struct
    with open(path, "rb") as f:
        f.read(12)
        jlen, _ = struct.unpack("<II", f.read(8))
        j = json.loads(f.read(jlen))
    prim = j["meshes"][0]["primitives"][0]
    faces = j["accessors"][prim["indices"]]["count"] // 3
    mimes = sorted({im.get("mimeType") for im in j.get("images", [])})
    return dict(faces=faces, vertices=j["accessors"][prim["attributes"]["POSITION"]]["count"], image_types=mimes,
                bytes=os.path.getsize(path))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Apple Silicon route (experimental): cut-out + trellis.cpp -> raw.glb")
    ap.add_argument("--job", action="append", required=True, help="PROPS.json:GOLD_ROOT:OUT (run.py's format)")
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--profile", default="draft", choices=sorted(PROFILES))
    ap.add_argument("--device", default="", help="cut-out device: mps (default when available) or cpu")
    ap.add_argument("--cut-root", default="", help="use CUT_ROOT/<id>/cut.png (e.g. the CUDA route's) instead of making a cut-out")
    a = ap.parse_args(argv)
    if a.seed <= 0:   # trellis-cli: seed 0 = a random seed (trellis_cli.cpp); negative values wrap
        raise SystemExit("STOP: trellis.cpp treats --seed 0 as 'pick a random seed'; use a seed of 1 or more")
    home, gdir = os.environ.get("TRELLIS_CPP_HOME", ""), os.environ.get("TRELLIS_GGUF", "")
    if not home or not gdir or not os.environ.get("T2_SRC") or not os.environ.get("HF_HOME"):
        raise SystemExit("STOP: set TRELLIS_CPP_HOME, TRELLIS_GGUF, T2_SRC and HF_HOME (config.example.env, Apple block)")
    prof = PROFILES[a.profile]
    tc, cli = trellis_cpp_info(home)
    ggufs = gguf_info(gdir, prof["res"])
    run = load_run()
    mach = machine()
    items = list(run.job_items([j.split(":") for j in a.job], a.only))
    cuts, torch_version = {}, None
    if a.cut_root:   # the same cut-out bytes as another run: compares the 3D step alone
        cut_load = dict(source="--cut-root: cut-outs made elsewhere")
        for pid, p, gold_root, out in items:
            od = os.path.join(out, pid)
            os.makedirs(os.path.join(od, "trellis"), exist_ok=True)
            src = os.path.join(a.cut_root, pid, "cut.png")
            shutil.copyfile(src, os.path.join(od, "cut.png"))
            gold = os.path.join(gold_root, p["gold"])
            cuts[pid] = dict(cut_png_md5=run.md5_file(src), gold_md5=run.md5_file(gold) if os.path.isfile(gold) else None)
    else:
        import torch
        from PIL import Image
        torch_version = torch.__version__
        run.forbid_rmbg(os.environ["HF_HOME"])
        device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
        t = time.time()
        cutter = run.BiRefNetCutout(os.environ["T2_SRC"], torch, device=device)
        cut_load = dict(cutter.info, device=device, load_s=round(time.time() - t, 1))
        for pid, p, gold_root, out in items:   # all cut-outs first, then free the cut-out model before trellis.cpp runs
            od = os.path.join(out, pid)
            os.makedirs(os.path.join(od, "trellis"), exist_ok=True)
            gold = os.path.join(gold_root, p["gold"])
            cut, raw, alpha, info = cutter(Image.open(gold).convert("RGB"))
            cut.save(os.path.join(od, "cut.png")); raw.save(os.path.join(od, "mask-birefnet.png")); alpha.save(os.path.join(od, "mask.png"))
            cuts[pid] = dict(info, gold_md5=run.md5_file(gold))
            print(f"{pid}: cut-out {info}", flush=True)
        if device == "mps":
            cut_load["mps_driver_memory_gib"] = round(torch.mps.driver_allocated_memory() / 2 ** 30, 2)
        del cutter
        import gc
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()   # the MPS allocator keeps its cache otherwise: GBs that trellis.cpp needs
    status = 0
    for pid, p, gold_root, out in items:
        od = os.path.join(out, pid)
        tdir = os.path.join(od, "trellis")
        cut_png, info = os.path.join(od, "cut.png"), cuts[pid]
        tglb = os.path.join(tdir, f"{pid}.glb")
        code, wall, rss, text, args = run_trellis(cli, cut_png, tglb, gdir, a.seed, prof, os.path.join(tdir, "trellis-cli.log"))
        used = re.search(r"generating model with seed (\d+)", text)
        rec = dict(prop=pid, backend="trellis-cpp", experimental=True, gold=p["gold"], gold_md5=info.pop("gold_md5"),
                   seed=int(used.group(1)) if used else None, profile=dict(name=a.profile, **prof), pipeline_type=f"trellis.cpp --res {prof['res']}",
                   export=dict(texture_size=prof["atlas"], decimation_target=prof["faces"]), export_note="as configured",
                   model=f"microsoft/TRELLIS.2-4B@{run.HF_REV} (converted to GGUF f16/f32)", code=None,
                   trellis_cpp=dict(tc, args=args), gguf=ggufs, cutout=dict(cut_load, **info), gpu=f"{mach['chip']} (Metal)",
                   machine=mach, torch=torch_version, seconds=dict(trellis_cpp_wall=wall), peak_rss_bytes=rss,
                   exit_code=code, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        if code != 0 or not os.path.isfile(tglb):
            oom = bool(re.search(r"out of memory|failed to allocate|insufficient memory", text, re.I))
            with open(os.path.join(od, "oom-generate.json" if oom else "trellis-error.json"), "w") as f:
                json.dump(rec, f, indent=1)
            print(f"{pid}: trellis.cpp failed (exit {code}); see {tdir}/trellis-cli.log", flush=True)
            status = EXIT_OOM if oom else 1
            continue
        shutil.copyfile(tglb, os.path.join(od, "raw.glb"))
        rec["raw_glb"] = glb_stats(os.path.join(od, "raw.glb"))
        with open(os.path.join(od, "record.json"), "w") as f:
            json.dump(rec, f, indent=1)
        print(f"{pid}: raw.glb {rec['raw_glb']}, {wall} s, peak RSS {rss / 2**30 if rss else float('nan'):.2f} GiB", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
