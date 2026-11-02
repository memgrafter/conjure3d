"""Optional image stage: FLUX.2-klein-4B (Apache-2.0) worker. Loads the pipeline once, then processes job files dropped
into a queue folder (one job = one round of images, or part of one). Each finished image goes to
$IMAGEGEN_HOME/out/<round>/<id>.png with a line in $IMAGEGEN_HOME/out/<round>/log.jsonl; each job ends with a contact
sheet (contact.jpg).

  source config.env; HF_HUB_OFFLINE=1 $IMAGEGEN_VENV/bin/python -I imagegen/gen.py serve [--model klein4b]
  IMAGEGEN_QUEUE=<folder> picks the queue folder (default $IMAGEGEN_HOME/queue; one worker per queue). A file named
  STOP in the queue folder ends the worker once the queue is empty.
  Before each image the worker waits while MemAvailable is under IMAGEGEN_MIN_RAM_MIB (default 3072).
  GPU_GUARD_EXPECT_POWER_W (optional): stop before each job unless every GPU's power limit is that value (read only).
  job JSON: {"round": "r01-x", "items": [{"id", "prompt", "seed"|"seeds", "w", "h", "steps", "guidance", "refs": [...],
             "crop": [w, h], "meta": {...}}], "defaults": {...}}   refs are image paths relative to $IMAGEGEN_HOME;
  "meta" is copied into log.jsonl as it is.
Weights load from the pinned local snapshot in $HF_HOME (setup.sh downloads it and checks every file's SHA-256).
"""
import json, os, subprocess, sys, time, traceback

IG = os.environ["IMAGEGEN_HOME"]
MODELS = {
    "klein4b": dict(repo="black-forest-labs/FLUX.2-klein-4B", rev="e7b7dc27f91deacad38e78976d1f2b499d76a294",
                    licence="Apache-2.0", steps=4, guidance=1.0),
}


def power_ok():
    """Optional GPU guard: with GPU_GUARD_EXPECT_POWER_W set, every GPU's power limit must be that value."""
    want = os.environ.get("GPU_GUARD_EXPECT_POWER_W", "")
    if not want:
        return True, "no power check (GPU_GUARD_EXPECT_POWER_W not set)"
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,power.limit", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    bad = [l for l in out.strip().splitlines() if abs(float(l.split(",")[1]) - float(want)) > 0.5]
    return not bad, out.strip()


def wait_for_ram():
    floor = int(os.environ.get("IMAGEGEN_MIN_RAM_MIB", "3072"))
    while True:
        avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable:")).split()[1]) // 1024
        if avail >= floor:
            return
        print(f"RAM PAUSE: MemAvailable {avail} MiB < {floor} MiB; waiting", flush=True)
        time.sleep(30)


def load(name):
    import torch
    from diffusers import Flux2KleinPipeline
    m = MODELS[name]
    t = time.time()
    # load from the pinned local snapshot (only the diffusers folders were downloaded, so repo-id loading refuses offline)
    snap = os.path.join(os.environ["HF_HOME"], "hub", "models--" + m["repo"].replace("/", "--"), "snapshots", m["rev"])
    pipe = Flux2KleinPipeline.from_pretrained(snap, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe, time.time() - t


_ref_cache = {}


def ref_image(rel):
    from PIL import Image
    if rel not in _ref_cache:
        _ref_cache[rel] = Image.open(os.path.join(IG, rel)).convert("RGB")
    return _ref_cache[rel]


def expand(job):
    d = job.get("defaults", {})
    for it in job["items"]:
        seeds = it.get("seeds") or [it.get("seed", d.get("seed", 1))]
        for s in seeds:
            x = dict(d); x.update(it); x["seed"] = s
            x["id"] = f'{it["id"]}_s{s}' if len(seeds) > 1 or "seeds" in it else it["id"]
            yield x


def contact_sheet(rnd, ids, cols=8, tw=240):
    from PIL import Image, ImageDraw, ImageFont
    d = os.path.join(IG, "out", rnd)
    th = int(tw * 16 / 9)
    rows = (len(ids) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 22)), (40, 40, 40))
    font = ImageFont.load_default(size=15)
    dr = ImageDraw.Draw(sheet)
    for i, iid in enumerate(ids):
        p = os.path.join(d, iid + ".png")
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("RGB")
        im.thumbnail((tw, th))
        x, y = (i % cols) * tw, (i // cols) * (th + 22)
        sheet.paste(im, (x + (tw - im.width) // 2, y + (th - im.height) // 2))
        dr.text((x + 3, y + th + 3), iid[:30], fill=(255, 255, 255), font=font)
    out = os.path.join(d, "contact.jpg")
    sheet.save(out, quality=85)
    return out


def run_job(pipe, model, job):
    import torch
    m = MODELS[model]
    rnd = job["round"]
    od = os.path.join(IG, "out", rnd)
    os.makedirs(od, exist_ok=True)
    ids = []
    with open(os.path.join(od, "log.jsonl"), "a") as log:
        for it in expand(job):
            ids.append(it["id"])
            out = os.path.join(od, it["id"] + ".png")
            if os.path.exists(out):
                continue
            wait_for_ram()
            w, h = it.get("w", 1088), it.get("h", 1920)
            kw = dict(prompt=it["prompt"], width=w, height=h,
                      num_inference_steps=it.get("steps", m["steps"]), guidance_scale=it.get("guidance", m["guidance"]),
                      generator=torch.Generator("cuda").manual_seed(it["seed"]))
            refs = it.get("refs") or []
            if refs:
                ims = [ref_image(r) for r in refs]
                kw["image"] = ims if len(ims) > 1 else ims[0]
            torch.cuda.synchronize(); t0 = time.time()
            img = pipe(**kw).images[0]
            torch.cuda.synchronize(); gen_s = time.time() - t0
            if it.get("crop"):
                cw, ch = it["crop"]
                l, t = (img.width - cw) // 2, (img.height - ch) // 2
                img = img.crop((l, t, l + cw, t + ch))
            t1 = time.time(); img.save(out); save_s = time.time() - t1
            rec = dict(id=it["id"], file=it["id"] + ".png", model=m["repo"], revision=m["rev"], licence=m["licence"],
                       prompt=it["prompt"], negative=None, seed=it["seed"], steps=kw["num_inference_steps"],
                       guidance=kw["guidance_scale"], gen_size=[w, h], out_size=list(img.size), refs=refs,
                       meta=it.get("meta"),
                       gen_s=round(gen_s, 2), save_s=round(save_s, 2),
                       bytes=os.path.getsize(out), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f'{rnd}/{it["id"]} gen {gen_s:.1f}s save {save_s:.2f}s', flush=True)
    return contact_sheet(rnd, ids)


def serve(model):
    q = os.environ.get("IMAGEGEN_QUEUE") or os.path.join(IG, "queue"); os.makedirs(os.path.join(q, "done"), exist_ok=True)
    ok, txt = power_ok()
    if not ok:
        print("STOP: a GPU power limit is not GPU_GUARD_EXPECT_POWER_W\n" + txt, flush=True); sys.exit(2)
    pipe, load_s = load(model)
    print(f"loaded {model} in {load_s:.1f}s", flush=True)
    while True:
        jobs = sorted(f for f in os.listdir(q) if f.endswith(".json"))
        if not jobs:
            if os.path.exists(os.path.join(q, "STOP")):
                print("STOP file, exiting", flush=True); return
            time.sleep(3); continue
        f = os.path.join(q, jobs[0])
        ok, txt = power_ok()
        if not ok:
            print("STOP: a GPU power limit is not GPU_GUARD_EXPECT_POWER_W\n" + txt, flush=True); sys.exit(2)
        try:
            job = json.load(open(f))
            t = time.time()
            sheet = run_job(pipe, model, job)
            print(f"JOBDONE {jobs[0]} {time.time() - t:.0f}s sheet {sheet}", flush=True)
        except Exception:
            traceback.print_exc(); print(f"JOBFAIL {jobs[0]}", flush=True)
        os.replace(f, os.path.join(q, "done", jobs[0]))


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        serve(sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--model" else "klein4b")
    elif sys.argv[1] == "sheet":
        print(contact_sheet(sys.argv[2], sys.argv[3].split(",")))
