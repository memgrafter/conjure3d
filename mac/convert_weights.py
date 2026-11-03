"""Apple Silicon route (experimental): our pinned safetensors -> the GGUF files trellis.cpp loads.

  $MAC_VENV/bin/python -I mac/convert_weights.py --weights $T2_WEIGHTS --birefnet BIREFNET_DIR --trellis-cpp SRC --out GGUF_DIR

1. Every source file's SHA-256 must equal its pin: trellis2/run.py WEIGHT_FILES (TRELLIS.2-4B, the TRELLIS-image-large
   decoder, DINOv3 and its LICENSE.md) and BIREFNET_SHA256 (BIREFNET_DIR/model.safetensors).
2. A staging folder (GGUF_DIR/staging) gets the layout trellis.cpp's tools/convert.py reads: symlinks to the TRELLIS.2
   files, and DINOv3 renamed from its transformers names to the timm names trellis.cpp uses (q, k and v fused into
   qkv; DINOv3 has no key bias, so that third of qkv.bias is zero: the same numbers). BiRefNet is linked as it is.
3. trellis.cpp's own converter runs with python -I from the staging folder (never from inside the clone). It stores
   matrices as f16 (bf16 weights lose range there) and 1-D tensors as f32.
4. GGUF_DIR/manifest.json: the SHA-256 of each GGUF file, its source files and their pins. DINOv3's LICENSE.md is
   copied next to the GGUF files: its licence travels with every copy.
"""
import argparse, hashlib, json, os, shutil, struct, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMPONENTS = ("ss_flow", "shape_flow_512", "tex_flow_512", "shape_flow_1024", "tex_flow_1024", "shape_dec", "tex_dec",
              "ss_dec", "dinov3", "birefnet")


def run_pins():
    import runpy
    return runpy.run_path(os.path.join(os.path.dirname(HERE), "trellis2", "run.py"), run_name="convert_weights")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_tensor(path, header, base, name):
    import numpy as np
    v = header[name]
    assert v["dtype"] == "F32", (name, v["dtype"])
    o0, o1 = v["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + o0)
        return np.frombuffer(f.read(o1 - o0), dtype="<f4").reshape(v["shape"])


def dinov3_timm_names(src, dst):
    """transformers DINOv3ViTModel names -> the timm names trellis.cpp's dinov3.cpp reads. Same values, f32."""
    import numpy as np
    h, base = read_header(src)
    names = {k for k in h if k != "__metadata__"}
    get = lambda k: read_tensor(src, h, base, k)
    out = {"patch_embed.proj.weight": get("embeddings.patch_embeddings.weight"),
           "patch_embed.proj.bias": get("embeddings.patch_embeddings.bias"),
           "cls_token": get("embeddings.cls_token"),
           "reg_token": get("embeddings.register_tokens")}
    used = {"embeddings.patch_embeddings.weight", "embeddings.patch_embeddings.bias", "embeddings.cls_token",
            "embeddings.register_tokens", "embeddings.mask_token", "norm.weight", "norm.bias"}   # final norm: TRELLIS.2 uses a plain layer norm
    layers = sorted({int(k.split(".")[1]) for k in names if k.startswith("layer.")})
    for i in layers:
        s, d = f"layer.{i}.", f"blocks.{i}."
        q, k, v = (get(s + f"attention.{x}_proj.weight") for x in "qkv")
        out[d + "attn.qkv.weight"] = np.concatenate([q, k, v], 0)
        zero = np.zeros(q.shape[0], np.float32)
        bias = [get(s + f"attention.{x}_proj.bias") if s + f"attention.{x}_proj.bias" in names else zero for x in "qkv"]
        out[d + "attn.qkv.bias"] = np.concatenate(bias, 0)
        pairs = {"attention.o_proj": "attn.proj", "mlp.up_proj": "mlp.fc1", "mlp.down_proj": "mlp.fc2", "norm1": "norm1",
                 "norm2": "norm2"}
        for a, b in pairs.items():
            for w in ("weight", "bias"):
                out[d + f"{b}.{w}"] = get(s + f"{a}.{w}")
        out[d + "gamma_1"] = get(s + "layer_scale1.lambda1")
        out[d + "gamma_2"] = get(s + "layer_scale2.lambda1")
        used |= {s + f"attention.{x}_proj.{w}" for x in "qkvo" for w in ("weight", "bias")}
        used |= {s + f"{a}.{w}" for a in ("mlp.up_proj", "mlp.down_proj", "norm1", "norm2") for w in ("weight", "bias")}
        used |= {s + "layer_scale1.lambda1", s + "layer_scale2.lambda1"}
    left = sorted(names - used)
    if left:
        raise SystemExit(f"STOP: DINOv3 tensors not mapped: {left[:5]}")
    write_safetensors_f32(out, dst)
    return len(out)


def write_safetensors_f32(tensors, dst):
    """A safetensors file (all F32), written by hand like trellis.cpp's converter reads it."""
    import numpy as np
    header, offset, blobs = {}, 0, []
    for name in sorted(tensors):
        b = np.ascontiguousarray(tensors[name], dtype="<f4").tobytes()
        header[name] = dict(dtype="F32", shape=list(np.shape(tensors[name])), data_offsets=[offset, offset + len(b)])
        offset += len(b)
        blobs.append(b)
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb)
        for b in blobs:
            f.write(b)


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.unlink(dst)
    os.symlink(os.path.abspath(src), dst)


def stage(weights, birefnet, staging, pins):
    """-> {component: [source files]} after checking every source's SHA-256."""
    t2, dino = pins["T2_DIR"], pins["DINO_DIR"]
    for rel, v in sorted(pins["WEIGHT_FILES"].items()):
        p = os.path.join(weights, rel)
        if not os.path.isfile(p) or sha256(p) != v[3]:
            raise SystemExit(f"STOP: {p} is missing or not the pinned file (sha256 {v[3]})")
    bw = os.path.join(birefnet, "model.safetensors")
    if not os.path.isfile(bw) or sha256(bw) != pins["BIREFNET_SHA256"]:
        raise SystemExit(f"STOP: {bw} is missing or not the pinned BiRefNet weights (sha256 {pins['BIREFNET_SHA256']})")
    sources = {}
    for name in os.listdir(os.path.join(weights, t2, "ckpts")):
        link(os.path.join(weights, t2, "ckpts", name), os.path.join(staging, "ckpts", name))
    til = os.path.join(weights, t2, "microsoft", "TRELLIS-image-large", "ckpts")
    for name in os.listdir(til):
        link(os.path.join(til, name), os.path.join(staging, "tilarge", "ckpts", name))
    os.makedirs(os.path.join(staging, "dinov3"), exist_ok=True)
    dsrc = os.path.join(weights, dino, "model.safetensors")
    n = dinov3_timm_names(dsrc, os.path.join(staging, "dinov3", "model.safetensors"))
    print(f"dinov3: {n} tensors renamed to timm names", flush=True)
    link(os.path.join(weights, dino, "config.json"), os.path.join(staging, "dinov3", "config.json"))
    link(bw, os.path.join(staging, "birefnet", "model.safetensors"))
    ck = lambda f: f"{t2}/ckpts/{f}"
    sources.update(ss_flow=[ck("ss_flow_img_dit_1_3B_64_bf16.safetensors")],
                   shape_flow_512=[ck("slat_flow_img2shape_dit_1_3B_512_bf16.safetensors")],
                   tex_flow_512=[ck("slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors")],
                   shape_flow_1024=[ck("slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors")],
                   tex_flow_1024=[ck("slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors")],
                   shape_dec=[ck("shape_dec_next_dc_f16c32_fp16.safetensors")],
                   tex_dec=[ck("tex_dec_next_dc_f16c32_fp16.safetensors")],
                   ss_dec=[f"{t2}/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"],
                   dinov3=[f"{dino}/model.safetensors"],
                   birefnet=[f"{pins['BIREFNET_REPO']}@{pins['BIREFNET_REV']}/model.safetensors"])
    return sources


def main():
    ap = argparse.ArgumentParser(description="Convert our pinned weights to trellis.cpp GGUF files.")
    ap.add_argument("--weights", required=True, help="$T2_WEIGHTS (run.py's pinned copy)")
    ap.add_argument("--birefnet", required=True, help="folder with the pinned BiRefNet model.safetensors")
    ap.add_argument("--trellis-cpp", required=True, help="the trellis.cpp clone at its pinned commit")
    ap.add_argument("--out", required=True, help="output folder for the .gguf files")
    ap.add_argument("--only", default="", help="comma-separated components (default: all)")
    a = ap.parse_args()
    pins = run_pins()
    out = os.path.abspath(a.out)
    staging = os.path.join(out, "staging")
    sources = stage(a.weights, a.birefnet, staging, pins)
    comps = [c for c in a.only.split(",") if c] or list(COMPONENTS)
    env = dict(os.environ, TRELLIS_MODELS=staging, TRELLIS_GGUF_OUT=out, TRELLIS_FAMILY="trellis")
    conv = os.path.join(os.path.abspath(a.trellis_cpp), "tools", "convert.py")
    subprocess.run([sys.executable, "-I", conv, *comps], cwd=staging, env=env, check=True)
    shutil.copyfile(os.path.join(a.weights, pins["DINO_DIR"], "LICENSE.md"), os.path.join(out, "DINOv3-LICENSE.md"))
    mpath = os.path.join(out, "manifest.json")
    man = json.load(open(mpath)) if os.path.isfile(mpath) else {}
    for c in comps:
        man[f"{c}.gguf"] = dict(sha256=sha256(os.path.join(out, f"{c}.gguf")), sources=sources[c],
                                converter="trellis.cpp tools/convert.py" + (" (DINOv3 renamed by mac/convert_weights.py)" if c == "dinov3" else ""))
    with open(mpath, "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    print(f"wrote {len(comps)} GGUF files and {mpath}")


if __name__ == "__main__":
    main()
