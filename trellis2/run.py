"""TRELLIS.2 image to 3D: many objects in one warm process.

  source config.env
  HF_HUB_OFFLINE=1 ATTN_BACKEND=xformers PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 \\
    $T2_VENV/bin/python -I trellis2/run.py --job PROPS.json:GOLD_ROOT:OUT [--job PROPS:GOLD_ROOT:OUT ...] \\
    [--offload upstream] [--save-mesh DIR] [--only a,b] [--cutout-only]
  or, several runners on one shared queue (one per GPU): ... run.py --queue QDIR --stream gpuN --log DIR

Per object: GOLD_ROOT/<props "gold" path> (the reference image, RGB) -> cut-out (BiRefNet general + keep the largest
piece, below) -> OUT/<id>/cut.png (RGBA) -> TRELLIS.2's own preprocess (alpha present, so its background remover is
never called) -> pipeline.run -> o_voxel.postprocess.to_glb (remesh, UV, texture bake) -> OUT/<id>/raw.glb (glTF Y up;
TRELLIS.2's native frame is Z up), input-trellis.png, record.json. --cutout-only: the cut-out stage only (no TRELLIS.2
load); OUT/<id>/cutout.json.

Cut-out: BiRefNet general, MIT. Code = github.com/ZhengPeng7/BiRefNet at BIREFNET_SHA, cloned by setup.sh into
<sources>/ext/BiRefNet (next to the pinned TRELLIS.2); weights = HF ZhengPeng7/BiRefNet model.safetensors at
BIREFNET_REV (SHA-256 checked), loaded with strict=True. No trust_remote_code: the model repo's own Python files are
never downloaded or run. The run stops if the clone is not the clean pinned commit or the weights file differs. Input
1024x1024 (resize, ImageNet normalisation), fp32 with TF32 off, sigmoid, 8-bit mask resized to the image (bilinear);
then "keep the largest piece" (LARGEST_PIECE_RULE).

Bake: uvrast (github.com/memgrafter/uvrast, MIT) is put in sys.modules["nvdiffrast.torch"] before o_voxel is
imported. uvrast must be the INSTALLED package at the commit pinned in requirements.txt; its version and commit go
into every record.json. The run stops if nvdiffrast can be imported (NVIDIA's licence allows research or evaluation
use only).
TRELLIS.2's own code comes from the venv's trellis2-src.pth (setup.sh); the run stops unless it is at T2_SHA.
TRELLIS.2's own background remover: the class is replaced by a stub before loading, so briaai/RMBG-2.0 (CC-BY-NC) is
never downloaded or run; any Hugging Face load of an RMBG repo stops the run, and so does an RMBG-2.0 folder in
HF_HOME. Only the models the chosen pipeline type uses are loaded.

Warm runs (RAM): upstream's low_vram mode moves each model to the GPU for its stage and back with module.cpu(),
which allocates new host memory for every weight each time. --offload persistent (default) keeps ONE host copy of
every weight for the whole process: moving to the GPU copies from it, moving back only drops the GPU copy. After each
object, glibc's malloc_trim gives free heap pages back to the OS.
--offload resident: every model (and BiRefNet) is copied to the GPU once at load and stays there; moving back does
nothing and the host copy (memory map or private copy) is freed, so the process keeps no host weights (draft: 11.2 GB
of GPU memory). TRELLIS.2 still runs its low_vram code path, so the computation is the same. --inflight N (needs
--offload resident): N objects at a time in one process, on N worker threads that share the GPU weights; each thread
claims its own objects. Same bytes as one at a time: in a worker thread torch.manual_seed seeds a per-thread
generator and torch.randn (TRELLIS.2's noise, on the CPU) draws from it; get_cond, decode_latent and the cascade's
shape sampling (they set shared model attributes) run one thread at a time, and so do the cut-out, get_cond and the
sparse-structure decoder, because the cut-out turns cuDNN's TF32 off for the whole process while it runs and the
other two use cuDNN in fp32. FlexGEMM's autotuners keep per-call state on objects all threads share, so N > 1 stops
unless FlexGEMM was built with FLEXGEMM_PATCH (one lock around every autotuner call; flexgemm_provenance, docs/PATCHES.md;
record.json "flexgemm" says which build ran). All threads use the default CUDA stream. GPU peaks in record.json are then device-wide
for all objects in flight.

Weights: TRELLIS.2 and DINOv3 load from a local copy (WEIGHT_FILES, made by setup.sh weights with --install-weights;
--weights, default $T2_WEIGHTS); every file's SHA-256 is checked at start.
--host-weights mapped (default): after a model loads, each weight that is byte-identical to its tensor in the
.safetensors file (same dtype and shape) is re-pointed into a read-only memory map of that file, so the host copy is
the kernel's page cache, shared by every process on the machine, not a private heap copy. A weight whose model dtype
differs from the file's (the decoders keep some layers in fp32; BiRefNet: fp16 file, fp32 model) keeps a private copy.
--host-weights private: a private heap copy per process (same bytes).

CuMesh: the run stops unless the imported CuMesh is the one setup.sh built from the pinned clone with the
determinism patch (patches/cumesh-deterministic.patch; cumesh_provenance below).
BVH: TRELLIS.2's GLB export calls CuMesh's cuBVH (the remesh's distance queries and the bake's attribute lookup).
CuMesh's own cuBVH comes from cubvh, which carries a non-commercial NVIDIA licence, so setup.sh builds CuMesh without
it (CUMESH_NO_CUBVH). trianglebvh (clean-room, MIT; the INSTALLED package at TRIANGLEBVH_SHA, built by setup.sh)
takes its place before the export; its version, commit and library SHA-256 go into every record.json. Its results
and the GLB bytes are the same as with cubvh (docs/WORK_IDENTICAL.md); any call into cubvh stops the run.
Remesh (--remesh lean, the default): cumesh_lean_remesh.py replaces CuMesh's remesh_narrow_band_dc with the same
function plus `del` of each intermediate (same GLB bytes, about 7 GiB lower export peak on large meshes). The run
stops if CuMesh's remeshing.py is not the file the copy was made from. --remesh upstream: CuMesh's own.
Export: after an out-of-memory error, one retry on the decoded mesh simplified to PRESIMPLIFY faces (export_glb); its
GLB differs from a first-try export and record.json says so (export_note).
--vram-cap-gb: simulate a smaller card (vram_ballast); off by default.
--chunk-norm ROWS: the decoders' LayerNorm32 runs on ROWS rows at a time (chunk_norm); lower decode peak, same bytes.
Off by default (0).
--export-from MESH.pt ... --export-out DIR: only the GLB export of decoded meshes saved with --save-mesh (export_from);
no cut-out, no generation, no TRELLIS.2 weights.
--profile draft|clean|best (PROFILES): pipeline type, texture size and GLB face target in one name; clean is the
default and gives the same bytes as runs before profiles existed. --pipeline-type, --texture-size and
--decimation-target override a profile's value (record.json "profile" lists them).
Measured per object: seconds and process RSS after every stage (stages.jsonl), the bake's rasterize/interpolate time,
GPU memory peaks (device-wide, sampled every 0.2 s), torch allocator peaks.
"""
import argparse, contextlib, ctypes, gc, hashlib, importlib.util, json, math, os, resource, shutil, struct, subprocess, sys, threading, time
from importlib import metadata

T2_SHA = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
HF_REV = "af44b45f2e35a493886929c6d786e563ec68364d"
TIL_REV = "25e0d31ffbebe4b5a97464dd851910efc3002d96"     # microsoft/TRELLIS-image-large (its sparse-structure decoder)
DINO_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_REV = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
CUMESH_SHA = "12289e1062f0603f2f0d0771b02e1395d247f26f"
CUMESH_PATCH = "patches/cumesh-deterministic.patch"
CUMESH_PATCH_SHA256 = "875efddae208bde1351c8061c7f7fb51c90898a89be02f9dacd700cc9d2c80d4"
# CuMesh is built without its cubvh submodule (NVIDIA non-commercial licence); this patch drops it from setup.py
CUMESH_NO_CUBVH = "patches/cumesh-no-cubvh.patch"
CUMESH_NO_CUBVH_SHA256 = "365bd3713e0a438926dc2693bd50d8387fbe6e0921b818d6993d8f179fa8437a"
TRIANGLEBVH_SHA = "b8ef3415f38839f96969417e6a7e311ccfd73bc8"   # trianglebvh (MIT): the BVH of the GLB export
FLEXGEMM_SHA = "6dd94a859c26ee8246888502eada3dd8ad85532e"      # JeffreyXiang/FlexGEMM (MIT)
# FlexGEMM's autotuners keep per-call state on objects all threads share; this patch makes every autotuner call hold
# one process-wide lock (docs/PATCHES.md). --inflight N > 1 stops without it.
FLEXGEMM_PATCH = "patches/flexgemm-autotuner-threadsafe.patch"
FLEXGEMM_PATCH_SHA256 = "dc182f4610002a3761df9ada8ef398a25ae2ce94489df83f19b8a6868ba4aa65"
FLEXGEMM_AUTOTUNER_BLOB = "ca95fc5843ec71ca064e0aa162c327f31b49f822"   # git blob id of flex_gemm/utils/autotuner.py at FLEXGEMM_SHA
# The patch is `git diff -- src` of the patched clone; this command prints it byte for byte whatever the git config
# (for example diff.algorithm=histogram). setup.sh uses the same flags.
CUMESH_DIFF = ["-c", "core.quotepath=on", "diff", "--no-color", "--no-ext-diff", "--no-renames", "--diff-algorithm=histogram",
               "--abbrev=7", "--src-prefix=a/", "--dst-prefix=b/", "--", "src"]
FLEXGEMM_DIFF = CUMESH_DIFF[:-1] + ["flex_gemm"]   # the FlexGEMM patch is `git diff -- flex_gemm` of the patched clone
T2_DIR = f"TRELLIS.2-4B@{HF_REV[:8]}"
DINO_DIR = f"dinov3-vitl16@{DINO_REV[:8]}"
# The files the 1024_cascade, 1536_cascade and 512 pipelines load, copied to $T2_WEIGHTS by --install-weights: local
# path -> (HF repo, revision, file, SHA-256). For the .safetensors files the SHA-256 is the HF blob's own name.
# TRELLIS-image-large's decoder sits under the TRELLIS.2 folder at the path pipeline.json names, so TRELLIS.2's loader
# finds it locally.
T2, TIL, DINO = ("microsoft/TRELLIS.2-4B", HF_REV), ("microsoft/TRELLIS-image-large", TIL_REV), (DINO_REPO, DINO_REV)
WEIGHT_FILES = {
    f"{T2_DIR}/pipeline.json": (*T2, "pipeline.json", "222c359ab1ed9bc6735a640a34f95d47f8681b9bc4aaa101bfb80274676253c6"),
    f"{T2_DIR}/ckpts/ss_flow_img_dit_1_3B_64_bf16.json": (*T2, "ckpts/ss_flow_img_dit_1_3B_64_bf16.json", "6cdfb636854f60ffcadd8c49244169cd449f34c5ccdb55b5ff7c04678f9c399e"),
    f"{T2_DIR}/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors": (*T2, "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors", "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6"),
    f"{T2_DIR}/ckpts/shape_dec_next_dc_f16c32_fp16.json": (*T2, "ckpts/shape_dec_next_dc_f16c32_fp16.json", "5f1b856dffce79466fb18f5f0eefc08f08c673b24b3c05b20359c0b6d318a209"),
    f"{T2_DIR}/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors": (*T2, "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors", "e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581"),
    f"{T2_DIR}/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json": (*T2, "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json", "ddbd7a1d34ce9a8e7af2f6ef9ae686d9ab49f69e62dacb98d8409b511cdbd5e2"),
    f"{T2_DIR}/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors": (*T2, "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f"),
    f"{T2_DIR}/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json": (*T2, "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json", "7ebd6d367393f52fee9e92c2b3d727a9026a338549c6a6fa84c7ea122140ffdd"),
    f"{T2_DIR}/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors": (*T2, "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors", "07cd0596f634c5adc1890023d16023afc5eed02fb84b22bb23aff5bf0030fbbd"),
    # the 512 pipeline's texture flow model (the draft profile)
    f"{T2_DIR}/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json": (*T2, "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json", "3093796257a62f838d2e5578ac4580ca3b7d88d10b87d348065d3fdfcc4c2077"),
    f"{T2_DIR}/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors": (*T2, "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors", "8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d"),
    f"{T2_DIR}/ckpts/tex_dec_next_dc_f16c32_fp16.json": (*T2, "ckpts/tex_dec_next_dc_f16c32_fp16.json", "22074e4ab5b28e2b72d33a3fe61bf416134b89e12d0fc5c729f37da75da0c4d1"),
    f"{T2_DIR}/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors": (*T2, "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors", "97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c"),
    f"{T2_DIR}/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json": (*T2, "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json", "31f4fdad9974930e5e9324ac9dfb6ef7de6b30cea51854ba515ef827a8d353b2"),
    f"{T2_DIR}/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors": (*T2, "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors", "580401269059a339b8318ab9ced459a13ba63391721c83a6c383198c29e77686"),
    f"{T2_DIR}/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.json": (*TIL, "ckpts/ss_dec_conv3d_16l8_fp16.json", "646781293f1cda74720de85d1cef50a957fb4aebd9a4bd014e454e32f2330ac5"),
    f"{T2_DIR}/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors": (*TIL, "ckpts/ss_dec_conv3d_16l8_fp16.safetensors", "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a"),
    f"{DINO_DIR}/config.json": (*DINO, "config.json", "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5"),
    f"{DINO_DIR}/model.safetensors": (*DINO, "model.safetensors", "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"),
    f"{DINO_DIR}/LICENSE.md": (*DINO, "LICENSE.md", "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e"),
}
WEIGHTS_MIN_FREE_GB = float(os.environ.get("T2_WEIGHTS_MIN_FREE_GB", "20"))   # the weights disk keeps this much free after the copy
SAFETENSORS_DTYPES = {"F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16", "I64": "int64", "I32": "int32",
                      "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool"}
NEEDED = {
    "512": ["sparse_structure_flow_model", "sparse_structure_decoder", "shape_slat_flow_model_512", "shape_slat_decoder",
            "tex_slat_flow_model_512", "tex_slat_decoder"],
    "1024_cascade": ["sparse_structure_flow_model", "sparse_structure_decoder", "shape_slat_flow_model_512",
                     "shape_slat_flow_model_1024", "shape_slat_decoder", "tex_slat_flow_model_1024", "tex_slat_decoder"],
}
# 1536_cascade: the same models as 1024_cascade (TRELLIS.2's run(): the 1024 shape flow model upsamples to 1536, or
# lower if the token count passes max_num_tokens; texture with the 1024 model). No new weights.
NEEDED["1536_cascade"] = NEEDED["1024_cascade"]
# Generation profiles: the whole recipe in one name; an explicit flag overrides its value.
# clean = the default (same bytes as runs before profiles existed). draft = fast and rough. best = 1536_cascade.
PROFILES = {
    "draft": dict(pipeline_type="512", texture_size=1024, decimation_target=100000),
    "clean": dict(pipeline_type="1024_cascade", texture_size=4096, decimation_target=500000),
    "best": dict(pipeline_type="1536_cascade", texture_size=4096, decimation_target=500000),
}
BIREFNET_SHA = "f7079632d681ecf12161fcea8905682934e59e51"   # github.com/ZhengPeng7/BiRefNet main, 2026-10-25 (MIT)
BIREFNET_REPO = "ZhengPeng7/BiRefNet"                          # HF: the "general" model (MIT)
BIREFNET_REV = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"     # HF main, 2026-02-04
BIREFNET_SHA256 = "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154"  # its model.safetensors
BIREFNET_SIZE = (1024, 1024)  # (w, h): the size the general model was trained at, and TRELLIS.2's own wrapper's size
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
LARGEST_DILATE = 17
LARGEST_PIECE_RULE = (f"keep the largest piece: 8-connected parts of alpha > 0.5; if there are 2 or more, alpha is "
                      f"multiplied by the largest part dilated with a {LARGEST_DILATE}x{LARGEST_DILATE} square (keeps its soft edge)")
BIREFNET_MODULES = ("config", "dataset", "image_proc", "utils", "models")  # its top-level names (repo root on sys.path)
PRESIMPLIFY = 4000000
CUDART = "libcudart.so.13"   # the CUDA runtime torch (cu130) and CuMesh share; same soname -> the loaded copy
EXPORT = dict(decimation_target=500000, remesh=True, remesh_band=1, remesh_project=0)
LIBC = ctypes.CDLL("libc.so.6") if sys.platform.startswith("linux") else None   # glibc: malloc_trim, mallinfo2
QUEUE_IDLE_S = 10


def power_check():
    """Optional GPU guard: with GPU_GUARD_EXPECT_POWER_W set, stop unless every GPU's power limit is that value (read
    only; limits are never changed here). Unset (the default): no check."""
    want = os.environ.get("GPU_GUARD_EXPECT_POWER_W", "")
    if not want:
        return
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,power.limit", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.strip()
    if any(abs(float(l.split(",")[1]) - float(want)) > 0.5 for l in out.splitlines()):
        raise SystemExit(f"STOP: a GPU power limit is not {float(want):.0f} W\n{out}")


def job_items(jobs, only):
    for props_path, gold_root, out in jobs:
        for pid, p in json.load(open(props_path))["props"].items():
            if not only or pid in only.split(","):
                yield pid, p, gold_root, out


def queue_items(qdir, stream):
    """--queue: several warm runners (one per GPU) share one queue. A job file (id, gold, gold_root, out) is claimed
    by an atomic rename todo/F -> claimed/F.<stream>; it moves to done/ when the runner asks for the next one. The
    optional power check (power_check) runs before every claim. Ends when todo/ is empty and QDIR/CLOSED exists, or when QDIR/STOP-<stream>
    exists. While it waits for work it prints a line every minute (the batch's hang guard watches the log)."""
    todo, claimed, done = (os.path.join(qdir, d) for d in ("todo", "claimed", "done"))
    prev, idle = None, 0
    while True:
        if prev:
            os.replace(prev, os.path.join(done, os.path.basename(prev)))
            prev = None
        if os.path.exists(os.path.join(qdir, f"STOP-{stream}")):
            print(f"queue: STOP-{stream} exists; no more claims", flush=True)
            return
        names = sorted(f for f in os.listdir(todo) if f.endswith(".json"))
        if not names:
            if os.path.exists(os.path.join(qdir, "CLOSED")):
                print("queue: empty and closed", flush=True)
                return
            if idle % max(1, 60 // QUEUE_IDLE_S) == 0:
                print(f"queue: waiting for work {time.strftime('%T')}", flush=True)
            idle += 1
            time.sleep(QUEUE_IDLE_S)
            continue
        idle = 0
        power_check()
        dst = os.path.join(claimed, f"{names[0]}.{stream}")
        try:
            os.rename(os.path.join(todo, names[0]), dst)
        except FileNotFoundError:   # another runner claimed it first
            continue
        prev = dst
        job = json.load(open(dst))
        print(f"queue: claimed {names[0]} ({job['id']}) {time.strftime('%T')}", flush=True)
        yield job["id"], job, job["gold_root"], job["out"]


class NoRembg:
    """Stands in for BiRefNet(briaai/RMBG-2.0): construction loads nothing; a call means the input had no alpha."""
    def __init__(self, *a, **k): pass
    def to(self, *a, **k): return self
    def cuda(self): return self
    def cpu(self): return self
    def __call__(self, image):
        raise RuntimeError("input has no alpha: background removal is disabled (RMBG-2.0 is CC-BY-NC)")


def forbid_rmbg(hf_home):
    """briaai/RMBG-2.0 is CC-BY-NC: stop if it is in HF_HOME, and make every Hugging Face load of an RMBG repo stop the run."""
    if os.path.exists(os.path.join(hf_home, "hub", "models--briaai--RMBG-2.0")):
        raise SystemExit(f"STOP: briaai/RMBG-2.0 (CC-BY-NC) is in {hf_home}; it must never be downloaded")
    import huggingface_hub, huggingface_hub.file_download, transformers.modeling_utils, transformers.utils.hub
    def guard(fn):
        def checked(*args, **kwargs):
            names = [str(v) for v in (*args, kwargs.get("repo_id"), kwargs.get("pretrained_model_name_or_path")) if isinstance(v, (str, os.PathLike))]
            if any("rmbg" in n.lower() for n in names):
                raise SystemExit(f"STOP: something tried to load {names} (briaai/RMBG-2.0 is CC-BY-NC)")
            return fn(*args, **kwargs)
        return checked
    for mod in (huggingface_hub, huggingface_hub.file_download, transformers.utils.hub):  # the last has its own references
        for n in ("hf_hub_download", "snapshot_download"):
            if hasattr(mod, n):
                setattr(mod, n, guard(getattr(mod, n)))
    pm = transformers.modeling_utils.PreTrainedModel
    pm.from_pretrained = classmethod(guard(pm.from_pretrained.__func__))


def largest_part(a):
    """LARGEST_PIECE_RULE; = cutout_eval/sheet.py largest_part (the research rule, 8/8 correct objects)"""
    import cv2, numpy as np
    n, lab, st, _ = cv2.connectedComponentsWithStats((a > 0.5).astype("uint8"), connectivity=8)
    if n <= 2:
        return a, n - 1
    keep = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype("uint8")
    return a * cv2.dilate(keep, np.ones((LARGEST_DILATE, LARGEST_DILATE), "uint8")).astype("float32"), n - 1


def md5_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class BiRefNetCutout:
    """The pinned BiRefNet general model and the largest-piece rule. Weights stay on the host between props (one copy,
    persistent_offload); each cut-out moves them to the GPU and back. resident: they stay on the GPU."""
    def __init__(self, src_root, torch, resident=False, device="cuda"):
        self.torch, self.device = torch, device   # device: "cuda"; the Apple route (mac/run_mac.py) passes "mps" or "cpu"
        root = os.path.join(src_root, "ext", "BiRefNet")
        git = lambda *a: subprocess.run(["git", "-C", root, *a], capture_output=True, text=True, check=True).stdout.strip()
        if git("rev-parse", "HEAD") != BIREFNET_SHA or git("status", "--porcelain"):
            raise SystemExit(f"STOP: BiRefNet code at {root} is not the clean pinned commit {BIREFNET_SHA}")
        snap = os.path.join(os.environ["HF_HOME"], "hub", "models--" + BIREFNET_REPO.replace("/", "--"), "snapshots", BIREFNET_REV)
        weights = os.path.realpath(os.path.join(snap, "model.safetensors"))
        if not os.path.exists(weights) or sha256(weights) != BIREFNET_SHA256:
            raise SystemExit(f"STOP: {snap}/model.safetensors is missing or not the pinned weights (sha256 {BIREFNET_SHA256})")
        clash = [m for m in sys.modules if m.split(".")[0] in BIREFNET_MODULES]
        if clash:
            raise SystemExit(f"STOP: modules {clash} are already imported; BiRefNet's top-level names would clash")
        # Its Config() reads train.sh from the working directory, so import and build it in the clone's root. Then take
        # its top-level names out of sys.modules again (the model keeps its own references), so nothing else sees them.
        # Its dataset.py also sets PIL's MAX_IMAGE_PIXELS to None (no decompression-bomb check); that is put back.
        from PIL import Image
        cwd, max_pixels = os.getcwd(), Image.MAX_IMAGE_PIXELS
        os.chdir(root); sys.path.insert(0, root)
        try:
            from models.birefnet import BiRefNet
            model = BiRefNet(bb_pretrained=False)  # no backbone download; every weight comes from the safetensors file
        finally:
            sys.path.remove(root); os.chdir(cwd); Image.MAX_IMAGE_PIXELS = max_pixels
            for m in [m for m in sys.modules if m.split(".")[0] in BIREFNET_MODULES]:
                del sys.modules[m]
        from safetensors.torch import load_file
        model.load_state_dict(load_file(weights), strict=True)
        self.model = model.eval()
        self.host_bytes = persistent_offload(self.model, torch, resident)
        self.info = dict(model="BiRefNet general (swin_v1_l backbone)", licence="MIT (code and weights)",
                         code=f"github.com/ZhengPeng7/BiRefNet@{BIREFNET_SHA}", code_path=root,
                         weights=f"{BIREFNET_REPO}@{BIREFNET_REV}/model.safetensors", weights_sha256=BIREFNET_SHA256,
                         input_size=list(BIREFNET_SIZE), precision="fp32, TF32 off", mask_resize="8-bit, bilinear to the image size",
                         post_rule=LARGEST_PIECE_RULE, trust_remote_code=False)

    def __call__(self, img):
        """img: PIL RGB -> (RGBA cut-out, 8-bit BiRefNet mask, 8-bit final alpha, per-image info)"""
        import numpy as np
        from PIL import Image
        from torchvision import transforms
        torch = self.torch
        t = time.time()
        tf = transforms.Compose([transforms.Resize((BIREFNET_SIZE[1], BIREFNET_SIZE[0])), transforms.ToTensor(),
                                 transforms.Normalize(*IMAGENET)])
        flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        try:
            self.model.to(torch.device(self.device))
            with torch.no_grad():
                pred = self.model(tf(img).unsqueeze(0).to(self.device))[-1].sigmoid().float().cpu()[0, 0]
        finally:
            self.model.cpu()
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
            torch.cuda.empty_cache()
        raw = Image.fromarray((pred.numpy() * 255).round().astype("uint8")).resize(img.size, Image.BILINEAR)
        a, parts = largest_part(np.asarray(raw).astype("float32") / 255)
        a8 = (a * 255).round().clip(0, 255).astype("uint8")
        cut = Image.fromarray(np.dstack([np.asarray(img), a8]))
        info = dict(parts=int(parts), fg=round(float((a > 0.5).mean()), 4), fg_birefnet=round(float((np.asarray(raw) > 127).mean()), 4),
                    seconds=round(time.time() - t, 3))
        if not (a8 < 255).any() or not (a8 > 0).any():
            raise RuntimeError(f"cut-out is empty or full (fg {info['fg']})")
        return cut, raw, Image.fromarray(a8), info


class GpuPeak(threading.Thread):
    def __init__(self, torch):
        super().__init__(daemon=True)
        self.torch, self.peak, self.run_flag = torch, 0, True
    def run(self):
        while self.run_flag:
            free, total = self.torch.cuda.mem_get_info()
            self.peak = max(self.peak, total - free)
            time.sleep(0.2)
    def reset(self):
        self.peak = 0


def rss_gb():
    with open("/proc/self/statm") as f:
        return round(int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3, 3)


def rss_peak_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2)


class Mallinfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks",
                                                "uordblks", "fordblks", "keepcost")]


def heap_gb():
    """glibc heap: in use, free inside the heap (fragmentation), and mmapped blocks"""
    LIBC.mallinfo2.restype = Mallinfo2
    m = LIBC.mallinfo2()
    g = lambda x: round(x / 1024 ** 3, 3)
    return dict(heap_used=g(m.uordblks), heap_free=g(m.fordblks), mmapped=g(m.hblkhd))


class Stages:
    """Seconds and RSS per stage; stage = a wrapped function. Lines go to stages.jsonl (ts = unix time at the stage
    end). failures: the innermost stage (with its parents) each exception came out of, e.g. an out-of-memory error.
    prop, acc, stack and failures are per thread (--inflight); lines are written whole under a lock."""
    def __init__(self, torch, path, digest=False):
        self.torch, self.f, self.digest = torch, open(path, "a"), digest
        self.tl, self.write_lock = threading.local(), threading.Lock()

    def _mine(self):
        if not hasattr(self.tl, "acc"):
            self.tl.prop, self.tl.acc, self.tl.stack, self.tl.failures = None, {}, [], []
        return self.tl

    prop = property(lambda self: self._mine().prop, lambda self, v: setattr(self._mine(), "prop", v))
    acc = property(lambda self: self._mine().acc, lambda self, v: setattr(self._mine(), "acc", v))
    stack = property(lambda self: self._mine().stack, lambda self, v: setattr(self._mine(), "stack", v))
    failures = property(lambda self: self._mine().failures, lambda self, v: setattr(self._mine(), "failures", v))

    def log(self, stage, dt, **extra):
        free, total = self.torch.cuda.mem_get_info()
        rec = dict(prop=self.prop, stage=stage, s=round(dt, 3), rss_gb=rss_gb(), **mem_status(), **heap_gb(), gpu_used_gb=round((total - free) / 1024 ** 3, 2),
                   torch_reserved_gb=round(self.torch.cuda.memory_reserved() / 1024 ** 3, 2), t=time.strftime("%H:%M:%S"), ts=round(time.time(), 3), **extra)
        with self.write_lock:
            self.f.write(json.dumps(rec) + "\n"); self.f.flush()
        a = self.acc.setdefault(stage, [0.0, 0]); a[0] += dt; a[1] += 1

    def wrap(self, owner, name, stage=None, sync=True, digest=False):
        fn, stage = getattr(owner, name), stage or name
        def timed(*args, **kwargs):
            if sync: self.torch.cuda.synchronize()
            rng = rng_digest(self.torch) if digest and self.digest else None   # RNG state when the stage starts
            t = time.time()
            self.stack.append(stage)
            try:
                out = fn(*args, **kwargs)
            except Exception as e:
                if not any(f["exc"] == id(e) for f in self.failures):   # only where it starts, not in every parent
                    self.failures.append(dict(exc=id(e), stage="/".join(self.stack), error=(str(e).splitlines() or [type(e).__name__])[0][:160],
                                              s=round(time.time() - t, 3), ts=round(time.time(), 3)))
                raise
            finally:
                self.stack.pop()
            if sync: self.torch.cuda.synchronize()
            dt = time.time() - t
            extra = dict(sha256=output_digest(out, self.torch), rng_before=rng) if digest and self.digest else {}
            if stage.startswith("gen.") and token_count(out) is not None:
                extra["tokens"] = token_count(out)   # sparse-structure / latent size: a measure of the object's difficulty
            self.log(stage, dt, **extra)
            return out
        setattr(owner, name, timed)

    def take_failures(self):
        out, self.failures = [{k: v for k, v in f.items() if k != "exc"} for f in self.failures], []
        return out

    def take(self):
        acc, self.acc = self.acc, {}
        return {k: dict(s=round(v[0], 3), calls=v[1]) for k, v in acc.items()}


def rng_digest(torch):
    """SHA-256 of torch's CPU and CUDA RNG states (--hash-stages): equal before a stage = the same random draws follow.
    In an --inflight worker thread the CPU state is that thread's generator (thread_rng)."""
    gen = getattr(_tl, "gen", None)
    cpu = gen.get_state() if gen is not None else torch.get_rng_state()
    return hashlib.sha256(cpu.numpy().tobytes() + torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()


def token_count(out):
    """Rows of a generation stage's output: voxel coords (sparse structure) or a sparse latent's coords; None otherwise."""
    if isinstance(out, (tuple, list)) and out:
        return token_count(out[0])
    for o in (getattr(out, "coords", None), out):   # a SparseTensor also has a .shape (batch, channels): coords first
        if hasattr(o, "shape") and hasattr(o, "dtype") and len(o.shape) == 2:
            return int(o.shape[0])
    return None


def output_digest(obj, torch):
    """SHA-256 over the tensors of a stage's output (tensors, lists, dicts, sparse tensors and meshes by their tensor
    fields), for --hash-stages: the first stage whose hash differs between two runs is where they part."""
    h = hashlib.sha256()
    def walk(o):
        if torch.is_tensor(o):
            t = o.detach().contiguous().cpu()
            h.update(f"{tuple(t.shape)}{t.dtype}".encode()); h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(o, dict):
            for k in sorted(o, key=str): walk(o[k])
        elif isinstance(o, (list, tuple)):
            for x in o: walk(x)
        else:
            for k in ("feats", "coords", "vertices", "faces", "attrs"):
                if torch.is_tensor(getattr(o, k, None)): walk(getattr(o, k))
    walk(obj)
    return h.hexdigest()[:16]


def persistent_offload(module, torch, resident=False):
    """Replace module.to(device) / module.cpu() with moves that keep one host copy of every weight: to the GPU =
    copy from the host copy; back = drop the GPU copy (weights are never changed at inference). No host memory is
    allocated after the first call, so the process RSS cannot grow from the low_vram moves.
    resident (--offload resident): the weights go to the GPU now and stay; back does nothing; the host copy is freed."""
    tensors = [t for _, t in module.named_parameters()] + [t for _, t in module.named_buffers()]
    host = [t.data for t in tensors]
    assert all(h.device.type == "cpu" for h in host), "persistent_offload must start from host weights"
    nbytes = sum(h.numel() * h.element_size() for h in host)
    state = {"dev": "cpu"}
    def to(device=None, *a, **k):
        dev = torch.device(device) if device is not None else torch.device("cpu")
        if dev.type == "cpu":
            return cpu()
        if state["dev"] != str(dev):
            for t, h in zip(tensors, host):
                t.data = h.to(dev)
            state["dev"] = str(dev)
        return module
    def cpu():
        if resident:
            return module
        for t, h in zip(tensors, host):
            t.data = h
        state["dev"] = "cpu"
        return module
    module.to, module.cpu = to, cpu
    module.cuda = lambda *a, **k: to(torch.device("cuda"))
    if resident:
        to(torch.device("cuda"))
        host.clear()
    return nbytes


_tl = threading.local()   # --inflight: per worker thread: rng (bool), gen (torch.Generator)


def thread_rng(torch):
    """--inflight: in a worker thread (_tl.rng set) torch.manual_seed seeds that thread's own CPU generator and
    torch.randn without a generator draws from it, so objects in flight do not share one random stream. The same seed
    gives the same values as the global generator. Elsewhere both behave as before."""
    seed0, randn0 = torch.manual_seed, torch.randn
    def manual_seed(seed):
        if getattr(_tl, "rng", False):
            _tl.gen = torch.Generator().manual_seed(int(seed))
            return _tl.gen
        return seed0(seed)
    def randn(*args, **kwargs):
        gen = getattr(_tl, "gen", None)
        if gen is not None and kwargs.get("generator") is None and torch.device(kwargs.get("device") or "cpu").type == "cpu":
            kwargs["generator"] = gen
        return randn0(*args, **kwargs)
    torch.manual_seed, torch.randn = manual_seed, randn


def run_inflight(a, jobs, one):
    """--inflight N: N worker threads call one(item). With --queue each thread claims for itself (its own queue_items:
    a job moves to done/ when that thread asks for its next one); with --job they share one list. An error in a
    thread stops the others after their current object; then it is raised here."""
    shared, items_lock = job_items(jobs, a.only), threading.Lock()
    stop, errors = threading.Event(), []
    def worker():
        _tl.rng = True
        mine = queue_items(a.queue, a.stream) if a.queue else None
        try:
            while not stop.is_set():
                if mine is not None:
                    item = next(mine, None)
                else:
                    with items_lock:
                        item = next(shared, None)
                if item is None:
                    return
                one(*item)
        except BaseException as e:
            errors.append(e)
            stop.set()
    threads = [threading.Thread(target=worker, name=f"inflight{k}") for k in range(a.inflight)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def one_at_a_time(owner, name, lock):
    """owner.name runs under lock (outermost, so stage timers do not count the wait)"""
    fn = getattr(owner, name)
    def locked(*args, **kwargs):
        with lock:
            return fn(*args, **kwargs)
    setattr(owner, name, locked)


def install_weights(dest, hf_home):
    """setup.sh: copy WEIGHT_FILES from HF_HOME's pinned snapshots to dest (a local disk), read-only, each SHA-256
    checked. Stops before copying if the disk would keep less than WEIGHTS_MIN_FREE_GB free."""
    todo = []
    for rel, (repo, rev, name, want) in WEIGHT_FILES.items():
        src = os.path.join(hf_home, "hub", "models--" + repo.replace("/", "--"), "snapshots", rev, name)
        blob = os.path.realpath(src)
        if not os.path.isfile(blob):
            raise SystemExit(f"STOP: {src} is missing (setup.sh downloads it first)")
        if len(os.path.basename(blob)) == 64 and os.path.basename(blob) != want:
            raise SystemExit(f"STOP: {src} is HF blob {os.path.basename(blob)}, not sha256 {want}")
        dst = os.path.join(dest, rel)
        if not (os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(blob) and sha256(dst) == want):
            todo.append((blob, dst, want))
    os.makedirs(dest, exist_ok=True)
    need, st = sum(os.path.getsize(b) for b, _, _ in todo), os.statvfs(dest)
    left = (st.f_bavail * st.f_frsize - need) / 1024 ** 3
    print(f"weights: {len(todo)} of {len(WEIGHT_FILES)} files to copy ({need / 1e9:.2f} GB); free after: {left:.1f} GiB", flush=True)
    if left < WEIGHTS_MIN_FREE_GB:
        raise SystemExit(f"STOP: {dest} would keep {left:.1f} GiB free, under {WEIGHTS_MIN_FREE_GB} GiB")
    for blob, dst, want in todo:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        part = dst + ".part"
        shutil.copyfile(blob, part)
        if sha256(part) != want:
            raise SystemExit(f"STOP: the copy {part} of {blob} is not sha256 {want}")
        os.chmod(part, 0o444)
        os.replace(part, dst)
    manifest = {rel: dict(source=f"{repo}@{rev}/{name}", sha256=want, bytes=os.path.getsize(os.path.join(dest, rel)))
                for rel, (repo, rev, name, want) in WEIGHT_FILES.items()}
    json.dump(manifest, open(os.path.join(dest, "manifest.json"), "w"), indent=1)
    print(f"weights OK in {dest}: {len(manifest)} files, SHA-256 checked", flush=True)


def check_weights(dest):
    """Every local weight file is present with its pinned SHA-256 (also reads it into the page cache)."""
    t = time.time()
    for rel, (_, _, _, want) in WEIGHT_FILES.items():
        p = os.path.join(dest, rel)
        if not os.path.isfile(p) or sha256(p) != want:
            raise SystemExit(f"STOP: {p} is missing or not sha256 {want}; run setup.sh (--install-weights)")
    return dict(dir=dest, files={rel: v[3] for rel, v in WEIGHT_FILES.items()},
                sources={rel: f"{v[0]}@{v[1]}/{v[2]}" for rel, v in WEIGHT_FILES.items()}, sha256_check_s=round(time.time() - t, 1))


def map_safetensors(path, torch):
    """{name: CPU tensor} pointing into a read-only private memory map of a .safetensors file (MAP_PRIVATE, file opened
    read-only): unwritten pages are the page cache's own pages, shared with every other process mapping the file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    storage = torch.UntypedStorage.from_file(path, shared=False, nbytes=os.path.getsize(path))
    out = {}
    for name, h in header.items():
        dtype = getattr(torch, SAFETENSORS_DTYPES[h["dtype"]])
        size = torch.empty((), dtype=dtype).element_size()
        b0, b1 = (8 + n + o for o in h["data_offsets"])
        if b0 % size or b1 - b0 != math.prod(h["shape"]) * size:
            raise SystemExit(f"STOP: {path}: tensor {name} is not aligned or has the wrong length")
        out[name] = torch.empty(0, dtype=dtype).set_(storage, b0 // size, h["shape"])
    return out


def map_weights(module, path, torch):
    """Point every weight of `module` that is byte-identical to its tensor in `path` (same name, dtype, shape, bytes)
    into the file's memory map; the private copy is then freed. Others (dtype converted on load, or not in the file)
    stay private. Returns the counts."""
    mapped = map_safetensors(path, torch)
    as_bytes = lambda t: t.reshape(-1).view(torch.uint8)
    rec = dict(file=path, mapped_tensors=0, mapped_gb=0.0, private_tensors=0, private_gb=0.0, private_examples=[])
    for name, t in module.state_dict(keep_vars=True).items():
        m, gb = mapped.get(name), t.numel() * t.element_size() / 1024 ** 3
        if m is not None and m.dtype == t.dtype and m.shape == t.shape and t.device.type == "cpu" and torch.equal(as_bytes(m), as_bytes(t.data)):
            t.data = m
            rec["mapped_tensors"] += 1; rec["mapped_gb"] += gb
        else:
            rec["private_tensors"] += 1; rec["private_gb"] += gb
            if len(rec["private_examples"]) < 4:
                rec["private_examples"].append(f"{name} {str(t.dtype)[6:]} (file: {str(m.dtype)[6:] if m is not None else 'absent'})")
    rec["mapped_gb"], rec["private_gb"] = round(rec["mapped_gb"], 3), round(rec["private_gb"], 3)
    return rec


def mem_status():
    """This process's resident memory split: anonymous (private heap etc.) vs file-backed (memory maps, shareable)."""
    with open("/proc/self/status") as f:
        kv = dict(line.split(":", 1) for line in f if line.startswith(("RssAnon", "RssFile", "RssShmem", "VmHWM")))
    return {k.lower().replace("vmhwm", "rss_peak") + "_gb": round(int(v.split()[0]) / 1024 ** 2, 3) for k, v in kv.items()}


def cumesh_provenance(cumesh):
    """Stops unless the imported CuMesh is the one setup.sh built from the pinned clone with the determinism patch:
    1. <venv>/cumesh-build.json (setup.sh writes it right after the build) names CUMESH_SHA, CUMESH_PATCH_SHA256 and
       CUMESH_NO_CUBVH_SHA256, and no compiled cubvh module is in the package;
    2. every compiled module of the imported package has the SHA-256 recorded there at build time (a rebuild or a
       reinstall from any other tree gives other binaries);
    3. the clone it was installed from (direct_url.json) is at CUMESH_SHA, and its source diff is exactly the patch."""
    pkg = os.path.dirname(os.path.realpath(cumesh.__file__))
    rec_path = os.path.join(sys.prefix, "cumesh-build.json")
    if not os.path.isfile(rec_path):
        raise SystemExit(f"STOP: no {rec_path}: this venv's CuMesh was not built by setup.sh with the determinism patch")
    rec = json.load(open(rec_path))
    if rec.get("commit") != CUMESH_SHA or rec.get("patch_sha256") != CUMESH_PATCH_SHA256:
        raise SystemExit(f"STOP: {rec_path} names commit {rec.get('commit')} / patch {rec.get('patch_sha256')}, not the pins")
    if rec.get("no_cubvh_patch_sha256") != CUMESH_NO_CUBVH_SHA256:
        raise SystemExit(f"STOP: {rec_path} has no {CUMESH_NO_CUBVH} record: this CuMesh may include cubvh (rebuild with setup.sh env)")
    sos = {f: sha256(os.path.join(pkg, f)) for f in sorted(os.listdir(pkg)) if f.endswith(".so")}
    if any(f.startswith("_cubvh") for f in sos):
        raise SystemExit(f"STOP: {pkg} has a compiled cubvh module (NVIDIA non-commercial licence)")
    if not sos or sos != rec.get("so_sha256"):
        raise SystemExit(f"STOP: the CuMesh binaries in {pkg} are not the ones setup.sh built from the patched clone")
    url = json.loads(metadata.distribution("cumesh").read_text("direct_url.json") or "{}").get("url", "")
    src = url.removeprefix("file://")
    git = lambda *a: subprocess.run(["git", "-C", src, *a], capture_output=True, check=True).stdout
    if not url.startswith("file://") or src != rec.get("source"):
        raise SystemExit(f"STOP: CuMesh was installed from {url}, not from the clone in {rec_path}")
    if git("rev-parse", "HEAD").decode().strip() != CUMESH_SHA or hashlib.sha256(git(*CUMESH_DIFF)).hexdigest() != CUMESH_PATCH_SHA256 \
            or git("ls-files", "--others", "--exclude-standard", "--", "src").strip():
        raise SystemExit(f"STOP: the CuMesh clone {src} is not {CUMESH_SHA} plus exactly {CUMESH_PATCH}")
    return dict(cumesh_patched=True, commit=CUMESH_SHA, patch=CUMESH_PATCH, patch_sha256=CUMESH_PATCH_SHA256, source=src,
                cubvh="not built", no_cubvh_patch=CUMESH_NO_CUBVH, no_cubvh_patch_sha256=CUMESH_NO_CUBVH_SHA256,
                so_sha256=sos, built=rec.get("built"))


def flexgemm_provenance(autotuner):
    """FlexGEMM's thread-safety patch (FLEXGEMM_PATCH), for the imported flex_gemm.utils.autotuner module. Patched only if
    1. the module has the patch's lock (_AUTOTUNE_LOCK);
    2. <venv>/flexgemm-build.json (setup.sh writes it right after the build) names FLEXGEMM_SHA and FLEXGEMM_PATCH_SHA256
       and the SHA-256 of the installed autotuner file, and the file still has it;
    3. the clone it was installed from (direct_url.json) is at FLEXGEMM_SHA, and its flex_gemm/ diff is exactly the patch.
    Otherwise flexgemm_patched is False, with the reason. One object at a time does not need the patch (same code path);
    --inflight N > 1 stops without it (main)."""
    path = os.path.realpath(autotuner.__file__)
    info = dict(flexgemm_patched=False, autotuner=path, autotuner_sha256=sha256(path))
    if not hasattr(autotuner, "_AUTOTUNE_LOCK"):
        return dict(info, reason="the imported FlexGEMM autotuner has no thread-safety lock (not built with the patch)")
    rec_path = os.path.join(sys.prefix, "flexgemm-build.json")
    if not os.path.isfile(rec_path):
        return dict(info, reason=f"no {rec_path}: FlexGEMM was not built by setup.sh with the patch")
    rec = json.load(open(rec_path))
    if rec.get("commit") != FLEXGEMM_SHA or rec.get("patch_sha256") != FLEXGEMM_PATCH_SHA256:
        return dict(info, reason=f"{rec_path} names commit {rec.get('commit')} / patch {rec.get('patch_sha256')}, not the pins")
    if rec.get("autotuner_sha256") != info["autotuner_sha256"]:
        return dict(info, reason=f"the installed {path} is not the file setup.sh built ({rec_path})")
    url = json.loads(metadata.distribution("flex_gemm").read_text("direct_url.json") or "{}").get("url", "")
    src = url.removeprefix("file://")
    if not url.startswith("file://") or src != rec.get("source"):
        return dict(info, reason=f"FlexGEMM was installed from {url}, not from the clone in {rec_path}")
    git = lambda *a: subprocess.run(["git", "-C", src, *a], capture_output=True, check=True).stdout
    if git("rev-parse", "HEAD").decode().strip() != FLEXGEMM_SHA or hashlib.sha256(git(*FLEXGEMM_DIFF)).hexdigest() != FLEXGEMM_PATCH_SHA256 \
            or git("ls-files", "--others", "--exclude-standard", "--", "flex_gemm").strip():
        return dict(info, reason=f"the FlexGEMM clone {src} is not {FLEXGEMM_SHA} plus exactly {FLEXGEMM_PATCH}")
    return dict(info, flexgemm_patched=True, commit=FLEXGEMM_SHA, patch=FLEXGEMM_PATCH, patch_sha256=FLEXGEMM_PATCH_SHA256,
                source=src, built=rec.get("built"))


def trianglebvh_provenance(trianglebvh):
    """The installed trianglebvh distribution: version, git commit (PEP 610 direct_url.json), library SHA-256. Stops on a
    path copy, a library that is not the installed package's, or a commit that is not TRIANGLEBVH_SHA."""
    dist = metadata.distribution("trianglebvh")
    pkg_dir = os.path.realpath(str(dist.locate_file("trianglebvh")))
    if os.path.dirname(os.path.realpath(trianglebvh.__file__)) != pkg_dir:
        raise SystemExit(f"STOP: trianglebvh imported from {trianglebvh.__file__}, not from the installed package {pkg_dir}")
    url = json.loads(dist.read_text("direct_url.json") or "{}")
    commit = url.get("vcs_info", {}).get("commit_id")
    if commit != TRIANGLEBVH_SHA:
        raise SystemExit(f"STOP: trianglebvh is not installed from commit {TRIANGLEBVH_SHA} (direct_url.json: {url})")
    if os.path.dirname(os.path.realpath(trianglebvh._LIB_PATH)) != pkg_dir:
        raise SystemExit(f"STOP: trianglebvh's library {trianglebvh._LIB_PATH} is not in the installed package {pkg_dir}")
    return dict(version=dist.version, commit=commit, url=url.get("url"), path=pkg_dir)


def uvrast_provenance(uvrast):
    """The installed uvrast distribution: version and git commit (PEP 610 direct_url.json). Stops on a path copy."""
    dist = metadata.distribution("uvrast")
    pkg_dir = os.path.realpath(str(dist.locate_file("uvrast")))
    if os.path.dirname(os.path.realpath(uvrast.__file__)) != pkg_dir:
        raise SystemExit(f"STOP: uvrast imported from {uvrast.__file__}, not from the installed package {pkg_dir}")
    url = json.loads(dist.read_text("direct_url.json") or "{}")
    commit = url.get("vcs_info", {}).get("commit_id")
    if not commit:
        raise SystemExit(f"STOP: uvrast is not installed from a git commit (direct_url.json: {url})")
    return dict(version=dist.version, commit=commit, url=url.get("url"), path=pkg_dir)


def trellis2_source(trellis2):
    """Path and git commit of the TRELLIS.2 code the venv imports; stops unless it is the pinned commit."""
    root = os.path.dirname(os.path.dirname(os.path.realpath(trellis2.__file__)))
    sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if sha != T2_SHA:
        raise SystemExit(f"STOP: TRELLIS.2 code at {root} is {sha}, not {T2_SHA}")
    return root


def vram_ballast(cap_gib, torch):
    """--vram-cap-gb: simulate a card whose CUDA-visible total memory is cap_gib (cudaMemGetInfo's total; e.g. an RTX
    4080 SUPER under Linux reports about 15.70 GiB). Before anything loads, one raw cudaMalloc takes (this card's total
    - cap) and keeps it for the whole process. It really occupies device memory, so torch's allocator, CuMesh and
    cuBVH (their own cudaMalloc) all see only the capped amount. Only an approximation of the smaller card: same
    kernels and allocator, but that card's own driver reserve, display use and speed are not simulated."""
    torch.cuda.init(); torch.zeros(1, device="cuda"); torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    uuid = f"GPU-{torch.cuda.get_device_properties(0).uuid}"
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout.splitlines()
    others = [l for l in apps if l.startswith(uuid) and int(l.split(",")[1]) != os.getpid()]
    if others:
        raise SystemExit(f"STOP: --vram-cap-gb needs the GPU to itself; other processes on it: {others}")
    size = total - int(cap_gib * 1024 ** 3)
    if size <= 0:
        raise SystemExit(f"STOP: --vram-cap-gb {cap_gib} is not below this card's {total / 1024 ** 3:.2f} GiB")
    ptr = ctypes.c_void_p()
    rc = ctypes.CDLL(CUDART).cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    free_after, _ = torch.cuda.mem_get_info()
    if rc != 0 or free - free_after < size:
        raise SystemExit(f"STOP: ballast of {size} bytes not resident (cudaMalloc {rc}; free {free} -> {free_after})")
    g = lambda x: round(x / 1024 ** 3, 3)
    return ptr, dict(cap_gib=cap_gib, card_total_gib=g(total), ballast_gib=g(size), context_gib=g(total - free),
                     free_after_gib=g(free_after), method="one raw cudaMalloc (libcudart, shared with torch and CuMesh) held for the process")


def load_sibling(name):
    """A module next to this script (python -I keeps the script folder off sys.path). A copy deployed under a prefixed
    name loads the module with the same prefix (p-run.py -> p-<name>.py)."""
    here, me = os.path.split(os.path.abspath(__file__))
    path = os.path.join(here, me[:-len("run.py")] + name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_glb(o_voxel, mesh, texture_size, export=EXPORT):
    return o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                                      attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                                      verbose=False, texture_size=texture_size, **export)


def export_glb(o_voxel, mesh, texture_size, torch, label="", export=EXPORT):
    """to_glb; after an out-of-memory error, one retry on the decoded mesh simplified to PRESIMPLIFY faces.
    Returns (glb or None, export_note, oom message or None)."""
    export_note, glb, oom = "as configured", None, None
    try:
        glb = to_glb(o_voxel, mesh, texture_size, export)
    except RuntimeError as e:
        if "out of memory" not in str(e):
            raise
        oom = str(e).splitlines()[0][:80]
    if glb is not None:
        return glb, export_note, None
    # Retry OUTSIDE the except block: inside it, the traceback keeps the failed to_glb frame and its CuMesh
    # objects alive, so their GPU memory is not freed (seen once: the retry ran out too).
    print(f"{label}: GLB export OOM ({oom}); retry after simplify to {PRESIMPLIFY} faces", flush=True)
    gc.collect(); torch.cuda.empty_cache()
    # CuMesh's failed cudaMalloc leaves the CUDA runtime's last error set (CuMesh and torch share one
    # libcudart), and CuMesh's first cudaGetLastError check in the retry reported it at once.
    # Clear it here; a safety net only: the retry's GLB differs from a first-try export, and is recorded.
    cleared = ctypes.CDLL(CUDART).cudaGetLastError()
    try:
        mesh.simplify(PRESIMPLIFY)
        gc.collect(); torch.cuda.empty_cache()
        export_note = (f"first export OOM ({oom}); CUDA last error {cleared} cleared; decoded mesh simplified to "
                       f"{PRESIMPLIFY} faces, then exported")
        glb = to_glb(o_voxel, mesh, texture_size, export)
    except RuntimeError as e:
        if "out of memory" not in str(e):
            raise
        oom = str(e).splitlines()[0][:80]
    return glb, export_note, oom


def trace_decode(pipe, torch, path):
    """--trace-decode FILE (diagnostic): GPU memory around every block of the two decoders and a list of the
    CUDA tensors alive when decode_latent starts. Resets torch's peak counter per block, so record.json's torch peaks
    are per block too in this mode. Results are not changed."""
    f = open(path, "a")
    G = 1024 ** 3
    def snap(tag, **kw):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        f.write(json.dumps(dict(tag=tag, ts=round(time.time(), 3), alloc=round(torch.cuda.memory_allocated() / G, 3),
                                peak=round(torch.cuda.max_memory_allocated() / G, 3), reserved=round(torch.cuda.memory_reserved() / G, 3),
                                used=round((total - free) / G, 3), **kw)) + "\n"); f.flush()
    def wrap_block(block, tag):
        fwd = block.forward
        def traced(*args, **kwargs):
            x = args[0]
            snap(tag + ":in", tokens=int(x.feats.shape[0]), channels=int(x.feats.shape[1]))
            torch.cuda.reset_peak_memory_stats()
            out = fwd(*args, **kwargs)
            h = out[0] if isinstance(out, tuple) else out
            snap(tag + ":out", tokens=int(h.feats.shape[0]), channels=int(h.feats.shape[1]))
            return out
        block.forward = traced
    for name in ("shape_slat_decoder", "tex_slat_decoder"):
        dec = pipe.models[name]
        for i, res in enumerate(dec.blocks):
            for j, block in enumerate(res):
                wrap_block(block, f"{name}.{i}.{j}")
        for n in ("from_latent", "output_layer"):
            wrap_block(getattr(dec, n), f"{name}.{n}")
    decode = pipe.decode_latent
    def traced_decode(*args, **kwargs):
        live = {}
        for o in gc.get_objects():
            try:
                if torch.is_tensor(o) and o.is_cuda:
                    k = f"{'param' if isinstance(o, torch.nn.Parameter) else 'tensor'} {tuple(o.shape)} {o.dtype}"
                    live[k] = live.get(k, 0) + o.untyped_storage().nbytes()
            except Exception:
                pass
        top = sorted(live.items(), key=lambda kv: -kv[1])[:25]
        snap("decode_latent:start", live_total_gib=round(sum(live.values()) / G, 3), live_top=[(k, round(v / G, 3)) for k, v in top])
        torch.cuda.reset_peak_memory_stats()
        out = decode(*args, **kwargs)
        snap("decode_latent:end")
        return out
    pipe.decode_latent = traced_decode


def chunk_norm(torch, rows):
    """--chunk-norm ROWS: LayerNorm32 (fp32 math on fp16 features) ROWS rows at a time on inputs with more rows, so its
    full-size fp32 copies are not made at once. The decoders' last up block works on every voxel at once (14.5M voxels
    x 64 channels on one object). The norm is per row, so the bytes do not change: with 1048576 rows the decode output
    and the GLB were identical to an unchunked run (one object, RTX 3090), and the GLBs of 3 objects under a 15.70 GiB
    cap too. Measured: the last up block's torch peak 0.96 GiB lower, enough for 3 of 4 objects that ran out of
    memory under that cap (docs/LOW_VRAM.md)."""
    from trellis2.modules import norm
    base = norm.LayerNorm32.forward
    def forward(self, x):
        if x.shape[0] <= rows:
            return base(self, x)
        out = torch.empty_like(x)
        for s in range(0, x.shape[0], rows):
            out[s:s + rows] = base(self, x[s:s + rows])
        return out
    norm.LayerNorm32.forward = forward


def export_from(a, export, provenance, st, torch, o_voxel):
    """--export-from: only the GLB export (remesh, simplify, UV, bake, file) of saved decoded meshes (--save-mesh), with
    this run's texture size and face target. The same mesh and settings give the same GLB bytes as the full run."""
    from trellis2.representations.mesh import base as mesh_base
    for path in a.export_from:
        d = torch.load(path, weights_only=False, map_location="cpu")
        pid = d["prop"]
        mesh = mesh_base.Mesh(d["vertices"].cuda(), d["faces"].cuda())
        mesh.attrs, mesh.coords, mesh.layout, mesh.voxel_size = d["attrs"].cuda(), d["coords"].cuda(), d["layout"], d["voxel_size"]
        od = os.path.join(a.export_out, pid)
        os.makedirs(od, exist_ok=True)
        st.prop = pid
        st.take_failures()
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); t = time.time()
        glb, note, oom = export_glb(o_voxel, mesh, a.texture_size, torch, pid, export)
        if glb is None:
            json.dump(dict(prop=pid, source=path, error="GLB export out of GPU memory twice", detail=oom,
                           stage_failures=st.take_failures()), open(os.path.join(od, "oom-glb.json"), "w"), indent=1)
            print(f"{pid}: SKIPPED, export out of memory", flush=True)
            continue
        torch.cuda.synchronize(); t1 = time.time()
        glb.export(os.path.join(od, "raw.glb"))
        t2 = time.time()
        st.log("glb.export_file", t2 - t1)
        rec = dict(prop=pid, source_mesh=path, source_bytes=os.path.getsize(path), seed=d.get("seed"), export_from=True,
                   export=dict(export, texture_size=a.texture_size), export_note=note, mesh_faces=int(mesh.faces.shape[0]),
                   glb_faces=int(len(glb.faces)), glb_s=round(t1 - t, 2), glb_file_s=round(t2 - t1, 2), stages=st.take(),
                   torch_alloc_peak_gb=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2), gpu=torch.cuda.get_device_name(0),
                   time=time.strftime("%Y-%m-%dT%H:%M:%S"), **provenance)
        failures = st.take_failures()
        if failures:
            rec["stage_failures"] = failures
        json.dump(rec, open(os.path.join(od, "record.json"), "w"), indent=1)
        print(f"{pid}: export {rec['glb_s']}s file {rec['glb_file_s']}s faces {rec['mesh_faces']}->{rec['glb_faces']}", flush=True)
        del mesh, glb, d
        gc.collect(); torch.cuda.empty_cache()


def build_parser():
    ap = argparse.ArgumentParser(description="TRELLIS.2 image to 3D: reference image -> cut-out -> raw.glb, many objects per process")
    ap.add_argument("--job", action="append", help="PROPS.json:GOLD_ROOT:OUT_DIR (repeatable)")
    ap.add_argument("--profile", default="clean", choices=sorted(PROFILES), help="generation profile (PROFILES); the three flags it sets override it")
    ap.add_argument("--pipeline-type", default=None, choices=sorted(NEEDED))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default="", help="comma list of prop ids (any job)")
    ap.add_argument("--texture-size", type=int, default=None)
    ap.add_argument("--bake", default="uvrast", choices=["uvrast"], help="uvrast (MIT); nvdiffrast is not supported")
    ap.add_argument("--offload", default="persistent", choices=["persistent", "resident", "upstream"],
                    help="persistent: one host copy, each model on the GPU for its stage; resident: all weights stay on the GPU, no host copy")
    ap.add_argument("--inflight", type=int, default=1, metavar="N",
                    help="objects at a time in this process (worker threads sharing the GPU weights); N > 1 needs --offload resident")
    ap.add_argument("--bvh", default="ours", choices=["ours"],
                    help="ours: trianglebvh (clean-room, MIT); CuMesh's cubvh is not offered (NVIDIA non-commercial licence)")
    ap.add_argument("--trim", default="on", choices=["on", "off"], help="malloc_trim(0) after each prop")
    ap.add_argument("--save-mesh", default="", help="dir: save each decoded mesh (.pt) for the bake comparison")
    ap.add_argument("--log", default="", help="dir for stages.jsonl and run.json (default: the first job's OUT)")
    ap.add_argument("--cutout-only", action="store_true", help="only the cut-out stage (no TRELLIS.2 load)")
    ap.add_argument("--weights", default=os.environ.get("T2_WEIGHTS", ""),
                    help="local weight copy (WEIGHT_FILES); default $T2_WEIGHTS")
    ap.add_argument("--host-weights", default="mapped", choices=["mapped", "private"],
                    help="mapped: host weights point into the files' page cache (shared); private: a heap copy per process")
    ap.add_argument("--hash-stages", action="store_true", help="log a SHA-256 of each generation stage's output (stages.jsonl)")
    ap.add_argument("--install-weights", action="store_true", help="setup.sh: copy WEIGHT_FILES from HF_HOME to --weights, then exit")
    ap.add_argument("--queue", default="", help="QDIR: claim job files from a shared queue (queue_items) instead of --job")
    ap.add_argument("--stream", default="", help="this runner's name in the queue (e.g. gpu0)")
    ap.add_argument("--vram-cap-gb", type=float, default=0,
                    help="simulate a smaller card: CUDA-visible total in GiB (e.g. 15.70 = RTX 4080 SUPER under Linux); off by default")
    ap.add_argument("--decimation-target", type=int, default=None, help="GLB faces after the remesh")
    ap.add_argument("--trace-decode", default="", help="diagnostic: JSONL of GPU memory per decoder block (trace_decode)")
    ap.add_argument("--chunk-norm", type=int, default=0, metavar="ROWS",
                    help="decoders' LayerNorm32 in blocks of ROWS rows (chunk_norm; same bytes, lower decode peak); 0 = off")
    ap.add_argument("--export-from", nargs="+", default=[], metavar="MESH.pt",
                    help="export only: decoded meshes saved with --save-mesh; no cut-out, no generation")
    ap.add_argument("--export-out", default="", help="with --export-from: OUT_DIR/<prop>/raw.glb + record.json")
    ap.add_argument("--remesh", default="lean", choices=["lean", "upstream"],
                    help="lean: cumesh_lean_remesh.py (same bytes, lower GPU peak); upstream: CuMesh's own function")
    return ap


def parse_args(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    overridden = sorted(k for k in PROFILES[a.profile] if getattr(a, k) is not None)
    for k, v in PROFILES[a.profile].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    a.profile_record = dict(name=a.profile, values={k: getattr(a, k) for k in PROFILES[a.profile]}, overridden=overridden)
    if a.chunk_norm < 0:
        ap.error("--chunk-norm needs 0 (off) or a positive row count")
    if a.inflight < 1 or (a.inflight > 1 and a.offload != "resident"):
        ap.error("--inflight needs N >= 1, and N > 1 needs --offload resident")
    if a.inflight > 1 and (a.cutout_only or a.trace_decode):
        ap.error("--inflight N > 1 does not combine with --cutout-only or --trace-decode")
    if not a.weights and not a.cutout_only and not a.export_from:
        ap.error("--weights or $T2_WEIGHTS is required (see config.example.env)")
    if a.install_weights:
        return a
    if a.export_out and not a.export_from:
        ap.error("--export-out needs --export-from")
    if a.export_from:
        if not a.export_out:
            ap.error("--export-from needs --export-out")
        if a.job or a.queue or a.cutout_only:
            ap.error("--export-from runs alone: no --job, --queue or --cutout-only")
        a.job = []
        return a
    if a.queue and (a.job or a.cutout_only or not a.stream or not a.log):
        ap.error("--queue needs --stream and --log, and no --job or --cutout-only")
    if not a.job and not a.queue:
        ap.error("--job or --queue is required")
    a.job = a.job or []
    return a


def main():
    a = parse_args()
    if a.install_weights:
        return install_weights(a.weights, os.environ["HF_HOME"])
    if a.host_weights == "mapped" and a.offload == "upstream":
        raise SystemExit("STOP: --host-weights mapped needs --offload persistent or resident (upstream's offload makes new host copies)")
    jobs = [j.split(":") for j in a.job]
    nvdiffrast_present = importlib.util.find_spec("nvdiffrast") is not None
    if nvdiffrast_present:
        raise SystemExit(f"STOP: nvdiffrast can be imported in {sys.prefix}; this runner needs a venv without it "
                         "(NVIDIA's licence allows research or evaluation use only)")
    import uvrast
    uvrast_info = uvrast_provenance(uvrast)
    uvrast.install()
    import trellis2
    t2_src = trellis2_source(trellis2)
    import numpy as np
    import torch
    from PIL import Image
    forbid_rmbg(os.environ["HF_HOME"])
    ballast, vram_cap = vram_ballast(a.vram_cap_gb, torch) if a.vram_cap_gb else (None, None)   # before any model loads
    if vram_cap:
        print(f"vram cap: {vram_cap}", flush=True)
    export = dict(EXPORT, decimation_target=a.decimation_target)
    import trellis2.pipelines.rembg as rembg_mod
    rembg_mod.BiRefNet = NoRembg
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel
    import cumesh
    assert o_voxel.postprocess.dr.__name__ == "uvrast", "bake module mismatch"
    cumesh_info = cumesh_provenance(cumesh)   # stops unless CuMesh is the patched build
    import flex_gemm.utils.autotuner
    flexgemm_info = flexgemm_provenance(flex_gemm.utils.autotuner)
    if a.inflight > 1 and not flexgemm_info["flexgemm_patched"]:
        raise SystemExit(f"STOP: --inflight {a.inflight} needs FlexGEMM built with {FLEXGEMM_PATCH}: {flexgemm_info['reason']} "
                         "(setup.sh env for a new venv, setup.sh flexgemm for an existing one)")
    remesh_info = load_sibling("cumesh_lean_remesh").install(cumesh, swap=a.remesh == "lean")  # before the stage timers wrap it
    import trianglebvh   # before the stage timers wrap cuBVH
    bvh_info = dict(trianglebvh_provenance(trianglebvh), **trianglebvh.install(cumesh))
    Trellis2ImageTo3DPipeline.model_names_to_load = NEEDED[a.pipeline_type]
    logdir = a.log or (jobs[0][2] if jobs else a.export_out)
    os.makedirs(logdir, exist_ok=True)
    if not a.export_from:
        t = time.time()
        cutter = BiRefNetCutout(os.path.dirname(t2_src), torch, resident=a.offload == "resident")
        cutout_load = dict(cutter.info, load_s=round(time.time() - t, 1), host_weights_gb=round(cutter.host_bytes / 1024 ** 3, 3))
        print(f"cut-out model loaded: {cutout_load}", flush=True)
    if a.cutout_only:
        for props_path, gold_root, out in jobs:
            for pid, p in json.load(open(props_path))["props"].items():
                if a.only and pid not in a.only.split(","):
                    continue
                od = os.path.join(out, pid)
                os.makedirs(od, exist_ok=True)
                cut, raw, alpha, info = cutter(Image.open(os.path.join(gold_root, p["gold"])).convert("RGB"))
                cut.save(os.path.join(od, "cut.png")); raw.save(os.path.join(od, "mask-birefnet.png")); alpha.save(os.path.join(od, "mask.png"))
                json.dump(dict(prop=pid, gold=p["gold"], cutout=dict(cutout_load, **info), venv=sys.prefix, gpu=torch.cuda.get_device_name(0),
                               torch=torch.__version__, time=time.strftime("%Y-%m-%dT%H:%M:%S")), open(os.path.join(od, "cutout.json"), "w"), indent=1)
                print(pid, info, flush=True)
        return
    st = Stages(torch, os.path.join(logdir, "stages.jsonl"), digest=a.hash_stages)
    # CuMesh allocates outside torch's cache. On a 35M-face decoded mesh its hole filling inside decode_latent ran out
    # of GPU memory while torch still held cached blocks, so the cache is emptied before each CuMesh hole fill.
    from trellis2.representations.mesh import base as mesh_base
    fill_holes = mesh_base.Mesh.fill_holes
    def fill_holes_empty_cache(self, *args, **kwargs):
        torch.cuda.empty_cache()
        return fill_holes(self, *args, **kwargs)
    mesh_base.Mesh.fill_holes = fill_holes_empty_cache
    # Same for every CuMesh init: a 25.0M-face decoded mesh ran out of GPU memory in the export's simplify, twice,
    # while torch kept ~17.9 GB cached from the remesh; with the cache emptied first it exported. Emptying
    # the cache frees only unused cached blocks, so results do not change.
    cumesh_init = cumesh.CuMesh.init
    def cumesh_init_empty_cache(self, *args, **kwargs):
        torch.cuda.empty_cache()
        return cumesh_init(self, *args, **kwargs)
    cumesh.CuMesh.init = cumesh_init_empty_cache
    # stage timers: the pipeline's stages, and inside to_glb the CuMesh steps, the two rasterizer calls, the voxel
    # attribute sampling and the inpaint
    pp = o_voxel.postprocess
    for n in ("fill_holes", "simplify", "uv_unwrap", "init"):
        st.wrap(cumesh.CuMesh, n, f"cumesh.{n}")
    st.wrap(cumesh.remeshing, "remesh_narrow_band_dc", "glb.cumesh.remesh")
    st.wrap(cumesh.cuBVH, "unsigned_distance", "glb.bvh.unsigned_distance")
    st.wrap(pp.dr, "rasterize", "glb.rasterize")
    st.wrap(pp.dr, "interpolate", "glb.interpolate")
    st.wrap(pp, "grid_sample_3d", "glb.grid_sample_3d")
    st.wrap(pp.cv2, "inpaint", "glb.inpaint", sync=False)
    if a.export_from:
        provenance = dict(venv=sys.prefix, trellis2_src=t2_src, uvrast=uvrast_info, bvh=bvh_info, cumesh=cumesh_info, flexgemm=flexgemm_info, remesh=remesh_info,
                          nvdiffrast_present="yes" if nvdiffrast_present else "no", profile=a.profile_record,
                          **(dict(vram_cap=vram_cap) if vram_cap else {}))
        return export_from(a, export, provenance, st, torch, o_voxel)
    props_info = torch.cuda.get_device_properties(0)
    mon = GpuPeak(torch)
    mon.start()
    t = time.time()
    # weights from the local copy only: TRELLIS.2's loader would fall back to the Hub for a missing file; this stops instead
    weights = check_weights(os.path.realpath(a.weights))
    snap, dino_dir = os.path.join(weights["dir"], T2_DIR), os.path.join(weights["dir"], DINO_DIR)
    from trellis2 import models as t2_models
    from trellis2.modules import image_feature_extractor as ife
    load_model, load_dino, mapping = t2_models.from_pretrained, ife.DINOv3ViTModel.from_pretrained, []
    def model_from_local(path, **kwargs):
        if not os.path.realpath(path).startswith(weights["dir"] + os.sep) or not os.path.isfile(f"{path}.safetensors"):
            raise SystemExit(f"STOP: model {path} is not in the local weights {weights['dir']}")
        model = load_model(path, **kwargs)
        if a.host_weights == "mapped":
            mapping.append(map_weights(model, f"{path}.safetensors", torch))
            gc.collect(); LIBC.malloc_trim(0)
        return model
    def dino_from_local(name, *args, **kwargs):
        if name != DINO_REPO:
            raise SystemExit(f"STOP: image condition model {name} is not {DINO_REPO}")
        return load_dino(dino_dir, *args, **kwargs)
    t2_models.from_pretrained, ife.DINOv3ViTModel.from_pretrained = model_from_local, staticmethod(dino_from_local)
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(snap)
    if a.host_weights == "mapped":
        mapping.append(map_weights(pipe.image_cond_model.model, os.path.join(dino_dir, "model.safetensors"), torch))
        gc.collect(); LIBC.malloc_trim(0)
    weights.update(host=a.host_weights, mapping=mapping,
                   mapped_gb=round(sum(m["mapped_gb"] for m in mapping), 3), private_gb=round(sum(m["private_gb"] for m in mapping), 3))
    host_bytes = 0
    if a.offload in ("persistent", "resident"):
        assert pipe.low_vram, "persistent and resident offload are for low_vram mode"
        for m in pipe.models.values():
            host_bytes += persistent_offload(m, torch, a.offload == "resident")
        host_bytes += persistent_offload(pipe.image_cond_model.model, torch, a.offload == "resident")
    pipe.cuda()
    if a.offload == "resident":   # the host copies (memory maps, private fp32 copies) are no longer referenced
        gc.collect(); LIBC.malloc_trim(0)
    for n in ("get_cond", "sample_sparse_structure", "sample_shape_slat_cascade", "sample_shape_slat", "sample_tex_slat", "decode_latent"):
        st.wrap(pipe, n, f"gen.{n}", digest=True)
    if a.trace_decode:
        trace_decode(pipe, torch, a.trace_decode)
    if a.chunk_norm:
        chunk_norm(torch, a.chunk_norm)
    if a.hash_stages:   # every shape sampling: the 512 pipeline's latent and the cascade's first (512) latent are comparable
        st.wrap(pipe.shape_slat_sampler, "sample", "gen.shape_slat_sampler", digest=True)
    if a.inflight > 1:   # shared state between objects in flight (module docstring, --inflight)
        thread_rng(torch)
        if os.environ.get("ATTN_BACKEND") == "xformers":
            # TRELLIS.2 imports it at the first attention call; done here once, so two threads never load it at the same time
            import xformers.ops   # noqa: F401
        cudnn_lock, decode_lock = threading.RLock(), threading.Lock()
        one_at_a_time(pipe, "get_cond", cudnn_lock)
        one_at_a_time(pipe.models["sparse_structure_decoder"], "forward", cudnn_lock)
        one_at_a_time(pipe, "decode_latent", decode_lock)
        one_at_a_time(pipe, "sample_shape_slat_cascade", decode_lock)   # clean/best: it switches the shape decoder's low_vram too
    else:
        cudnn_lock = contextlib.nullcontext()
    LIBC.malloc_trim(0)
    load_s = time.time() - t
    load = dict(venv=sys.prefix, trellis2_src=t2_src, uvrast=uvrast_info, bvh=bvh_info, cumesh=cumesh_info, cumesh_patched=cumesh_info["cumesh_patched"],
                flexgemm=flexgemm_info, remesh=remesh_info,
                cutout=cutout_load, nvdiffrast_present="yes" if nvdiffrast_present else "no",
                load_s=round(load_s, 1), gpu_used_peak_gb=round(mon.peak / 1024 ** 3, 2), rss_gb=rss_gb(), rss_peak_gb=rss_peak_gb(),
                mem=mem_status(), host_weights_gb=round(host_bytes / 1024 ** 3, 2), weights=weights, low_vram=bool(pipe.low_vram),
                models=sorted(pipe.models), profile=a.profile_record, **(dict(vram_cap=vram_cap) if vram_cap else {}))
    if a.offload == "resident":   # host_weights_gb above = what moved to the GPU; nothing stays on the host
        load.update(offload="resident", weights_on_gpu_gb=round((host_bytes + cutter.host_bytes) / 1024 ** 3, 2), inflight=a.inflight)
    st.prop = "_load"; st.log("load", load_s); st.take()
    print(f"loaded in {load_s:.1f}s on {props_info.name} {props_info.uuid}; {load}", flush=True)
    run = dict(args=vars(a), load=load, props=[], started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(run, open(os.path.join(logdir, "run.json"), "w"), indent=1)
    n_done, run_lock = 0, threading.Lock()

    def one(pid, p, gold_root, out):
        nonlocal n_done
        src = os.path.join(gold_root, p["gold"])
        if not os.path.exists(src):
            print(f"{pid}: no gold image {src}; skipped", flush=True)
            return
        od = os.path.join(out, pid)
        os.makedirs(od, exist_ok=True)
        st.prop = pid
        st.take_failures()
        tp = time.time()
        try:
            with cudnn_lock:
                img, raw, alpha, cut_info = cutter(Image.open(src).convert("RGB"))
        except RuntimeError as e:
            json.dump(dict(prop=pid, gold=p["gold"], error=str(e)), open(os.path.join(od, "cutout-error.json"), "w"), indent=1)
            print(f"{pid}: SKIPPED, cut-out failed: {e}", flush=True)
            return
        img.save(os.path.join(od, "cut.png")); raw.save(os.path.join(od, "mask-birefnet.png")); alpha.save(os.path.join(od, "mask.png"))
        st.log("cutout", cut_info["seconds"])
        assert not np.all(np.asarray(img)[..., 3] == 255), f"{pid}: cut-out has no alpha"
        pre = pipe.preprocess_image(img)
        pre.save(os.path.join(od, "input-trellis.png"))
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); mon.reset(); t0 = time.time()
        try:
            mesh = pipe.run(pre, seed=a.seed, preprocess_image=False, pipeline_type=a.pipeline_type)[0]
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not isinstance(e, torch.cuda.OutOfMemoryError) and "out of memory" not in str(e):
                raise
            err = dict(prop=pid, pipeline_type=a.pipeline_type, error="CUDA out of memory", detail=str(e)[:400],
                       seconds_before_oom=round(time.time() - t0, 1), gpu_used_peak_gb=round(mon.peak / 1024 ** 3, 2),
                       stage_failures=st.take_failures(), **(dict(vram_cap=vram_cap) if vram_cap else {}))
            json.dump(err, open(os.path.join(od, f"oom-{a.pipeline_type}.json"), "w"), indent=1)
            print(f"{pid}: OOM {err}", flush=True)
            torch.cuda.empty_cache()
            return
        torch.cuda.synchronize(); t1 = time.time()
        run_peak = dict(gpu_used_peak_gb=round(mon.peak / 1024 ** 3, 2), torch_alloc_peak_gb=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
                        torch_reserved_peak_gb=round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2))
        n_v, n_f = int(mesh.vertices.shape[0]), int(mesh.faces.shape[0])
        mesh.simplify(16777216)  # 2**24 faces at most (a rasterizer limit kept from the original bake): same mesh for every bake
        print(f"{pid}: decoded mesh {n_v} vertices, {n_f} faces", flush=True)
        if a.save_mesh:
            os.makedirs(a.save_mesh, exist_ok=True)
            torch.save(dict(vertices=mesh.vertices.cpu(), faces=mesh.faces.cpu(), attrs=mesh.attrs.cpu(), coords=mesh.coords.cpu(),
                            layout=mesh.layout, voxel_size=mesh.voxel_size, prop=pid, seed=a.seed),
                       os.path.join(a.save_mesh, f"{pid}.pt"))
            st.log("save_mesh", time.time() - t1)
        # CuMesh allocates outside torch's cache: give the cached blocks back first (a large mesh hit a CuMesh OOM)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(); mon.reset(); t2 = time.time()
        glb, export_note, oom = export_glb(o_voxel, mesh, a.texture_size, torch, pid, export)
        if glb is None:
            # A warm run must not stop on one object (seen once: CuMesh ran out of GPU memory in both tries)
            err = dict(prop=pid, error="GLB export out of GPU memory twice (CuMesh)", detail=oom, mesh_faces=n_f,
                       model_s=round(t1 - t0, 2), seconds_in_export=round(time.time() - t2, 1), stage_failures=st.take_failures(),
                       **(dict(vram_cap=vram_cap) if vram_cap else {}))
            json.dump(err, open(os.path.join(od, "oom-glb.json"), "w"), indent=1)
            print(f"{pid}: SKIPPED {err}", flush=True)
            del mesh
            gc.collect(); torch.cuda.empty_cache()
            return
        torch.cuda.synchronize(); t3 = time.time()
        glb.export(os.path.join(od, "raw.glb"))  # PNG textures (no WebP extension), so every importer reads it
        t4 = time.time()
        st.log("glb.export_file", t4 - t3)
        glb_peak = dict(gpu_used_peak_gb=round(mon.peak / 1024 ** 3, 2), torch_alloc_peak_gb=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2))
        bake = "uvrast (clean-room, MIT)"
        rec = dict(prop=pid, gold=p["gold"], gold_md5=md5_file(src), cutout=dict(cutout_load, file="cut.png", **cut_info), preprocess="TRELLIS.2 preprocess_image (alpha bbox crop, premultiplied on black)",
                   model=f"microsoft/TRELLIS.2-4B@{HF_REV}", code=f"microsoft/TRELLIS.2@{T2_SHA}", licence="MIT (code and weights)",
                   image_cond="facebook/dinov3-vitl16-pretrain-lvd1689m (DINOv3 License)", rembg="TRELLIS.2's own remover: stub, never loaded (briaai/RMBG-2.0 is CC-BY-NC); alpha from our cut-out",
                   bake=f"o_voxel.postprocess.to_glb with {bake}", uvrast=uvrast_info, bvh=bvh_info, cumesh=cumesh_info, cumesh_patched=cumesh_info["cumesh_patched"], remesh=remesh_info,
                   flexgemm=flexgemm_info, nvdiffrast_present="yes" if nvdiffrast_present else "no", venv=sys.prefix, offload=a.offload, attention=os.environ.get("ATTN_BACKEND", "flash_attn"),
                   profile=a.profile_record, pipeline_type=a.pipeline_type, seed=a.seed, export=dict(export, texture_size=a.texture_size), export_note=export_note,
                   mesh_vertices=n_v, mesh_faces=n_f, glb_vertices=int(len(glb.vertices)), glb_faces=int(len(glb.faces)),
                   frame="raw.glb: glTF Y up (to_glb swaps TRELLIS.2's Z-up frame: y_glb = z, z_glb = -y); object in the [-0.5, 0.5]^3 box",
                   gpu=props_info.name, gpu_uuid=str(props_info.uuid), torch=torch.__version__, cuda=torch.version.cuda,
                   load=load, warm_index=n_done, model_s=round(t1 - t0, 2), glb_s=round(t3 - t2, 2), glb_file_s=round(t4 - t3, 2),
                   prop_s=round(t4 - tp, 2), stages=st.take(), run_mem=run_peak, glb_mem=glb_peak, rss_peak_gb=rss_peak_gb(),
                   time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        failures = st.take_failures()
        if failures:   # e.g. the first export's OOM before a retry that worked
            rec["stage_failures"] = failures
        if vram_cap:
            rec["vram_cap"] = vram_cap
        if a.inflight > 1:   # GPU peaks above are device-wide for every object in flight
            rec["inflight"] = a.inflight
        del mesh, glb
        torch.cuda.empty_cache()
        rec["rss_end_before_trim_gb"] = rss_gb()
        if a.trim == "on":
            LIBC.malloc_trim(0)
        rec["rss_end_gb"] = rss_gb()
        rec["heap_end"] = heap_gb()
        rec["mem_end"] = mem_status()
        json.dump(rec, open(os.path.join(od, "record.json"), "w"), indent=1)
        with run_lock:
            run["props"].append(dict(prop=pid, out=od, prop_s=rec["prop_s"], model_s=rec["model_s"], glb_s=rec["glb_s"],
                                     rss_end_gb=rec["rss_end_gb"], rss_peak_gb=rec["rss_peak_gb"]))
            json.dump(run, open(os.path.join(logdir, "run.json"), "w"), indent=1)
        print(f"{pid}: prop {rec['prop_s']}s model {t1 - t0:.1f}s glb {t3 - t2:.1f}s faces {n_f}->{rec['glb_faces']} "
              f"rss end {rec['rss_end_gb']} GB (before trim {rec['rss_end_before_trim_gb']}) peak {rec['rss_peak_gb']} GB", flush=True)
        with run_lock:
            n_done += 1

    if a.inflight == 1:
        for item in (queue_items(a.queue, a.stream) if a.queue else job_items(jobs, a.only)):
            one(*item)
    else:
        run_inflight(a, jobs, one)
    mon.run_flag = False


if __name__ == "__main__":
    main()
