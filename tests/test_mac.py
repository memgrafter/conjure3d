"""The Apple Silicon route (mac/, conjure3d.py --backend trellis-cpp): the parts that need no GPU, no torch, no weights."""
import json, os, stat, struct, sys
import numpy as np
import pytest
from conftest import load

conv = load("mac/convert_weights.py", "mac_convert_weights")
run_mac = load("mac/run_mac.py", "mac_run_mac")
post = load("post/post_step.py", "post_step_mac")
cli_module = load("conjure3d.py", "conjure3d_cli_mac")


def hf_dinov3(path, layers=2, d=4, extra=()):
    """A tiny DINOv3 file with the transformers names (no key bias, as DINOv3)."""
    rng = np.random.default_rng(0)
    t = {"embeddings.cls_token": (1, 1, d), "embeddings.mask_token": (1, 1, d), "embeddings.register_tokens": (1, 4, d),
         "embeddings.patch_embeddings.weight": (d, 3, 2, 2), "embeddings.patch_embeddings.bias": (d,),
         "norm.weight": (d,), "norm.bias": (d,)}
    for i in range(layers):
        s = f"layer.{i}."
        t.update({s + "attention.q_proj.weight": (d, d), s + "attention.q_proj.bias": (d,), s + "attention.k_proj.weight": (d, d),
                  s + "attention.v_proj.weight": (d, d), s + "attention.v_proj.bias": (d,), s + "attention.o_proj.weight": (d, d),
                  s + "attention.o_proj.bias": (d,), s + "mlp.up_proj.weight": (4 * d, d), s + "mlp.up_proj.bias": (4 * d,),
                  s + "mlp.down_proj.weight": (d, 4 * d), s + "mlp.down_proj.bias": (d,), s + "norm1.weight": (d,),
                  s + "norm1.bias": (d,), s + "norm2.weight": (d,), s + "norm2.bias": (d,), s + "layer_scale1.lambda1": (d,),
                  s + "layer_scale2.lambda1": (d,)})
    t.update({k: (d,) for k in extra})
    arrays = {k: rng.standard_normal(v).astype(np.float32) for k, v in t.items()}
    conv.write_safetensors_f32(arrays, path)
    return arrays


def read_all(path):
    h, base = conv.read_header(path)
    return {k: conv.read_tensor(path, h, base, k) for k in h if k != "__metadata__"}


def test_dinov3_names_and_fused_qkv(tmp_path):
    src = hf_dinov3(str(tmp_path / "hf.safetensors"))
    n = conv.dinov3_timm_names(str(tmp_path / "hf.safetensors"), str(tmp_path / "timm.safetensors"))
    out = read_all(str(tmp_path / "timm.safetensors"))
    assert n == len(out) == 4 + 2 * 14   # 340 for the real ViT-L (24 layers)
    assert np.array_equal(out["blocks.1.attn.qkv.weight"], np.concatenate(
        [src[f"layer.1.attention.{x}_proj.weight"] for x in "qkv"], 0))
    q, k, v = np.split(out["blocks.1.attn.qkv.bias"], 3)
    assert np.array_equal(q, src["layer.1.attention.q_proj.bias"]) and not k.any() and np.array_equal(v, src["layer.1.attention.v_proj.bias"])
    assert np.array_equal(out["blocks.0.gamma_2"], src["layer.0.layer_scale2.lambda1"])
    assert np.array_equal(out["reg_token"], src["embeddings.register_tokens"])
    assert np.array_equal(out["blocks.0.mlp.fc2.weight"], src["layer.0.mlp.down_proj.weight"])
    assert "norm.weight" not in out and "mask_token" not in str(sorted(out))


def test_dinov3_unknown_tensor_stops(tmp_path):
    hf_dinov3(str(tmp_path / "hf.safetensors"), extra=("layer.0.attention.k_proj.bias_new",))
    with pytest.raises(SystemExit, match="not mapped"):
        conv.dinov3_timm_names(str(tmp_path / "hf.safetensors"), str(tmp_path / "timm.safetensors"))


def test_peak_rss_from_time_l():
    text = "done in 500.0s\n      500.12 real  10.0 user\n          5634564096  maximum resident set size\n"
    assert run_mac.peak_rss_bytes(text) == 5634564096
    assert run_mac.peak_rss_bytes("no time output") is None


def test_glb_stats(tmp_path):
    j = {"asset": {"version": "2.0"}, "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
         "accessors": [{"count": 4}, {"count": 6}], "images": [{"mimeType": "image/png"}, {"mimeType": "image/png"}]}
    jb = json.dumps(j).encode()
    jb += b" " * (-len(jb) % 4)
    p = tmp_path / "a.glb"
    p.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(jb)) + struct.pack("<II", len(jb), 0x4E4F534A) + jb)
    assert run_mac.glb_stats(str(p)) == dict(faces=2, vertices=4, image_types=["image/png"], bytes=p.stat().st_size)


def test_gguf_manifest_mismatch_stops(tmp_path):
    for c in run_mac.GGUF_NEEDED[512]:
        (tmp_path / f"{c}.gguf").write_bytes(c.encode())
    man = {f"{c}.gguf": dict(sha256=run_mac.sha256(str(tmp_path / f"{c}.gguf")), sources=[c]) for c in run_mac.GGUF_NEEDED[512]}
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    (tmp_path / "DINOv3-LICENSE.md").write_text("licence")
    assert set(run_mac.gguf_info(str(tmp_path), 512)) == {f"{c}.gguf" for c in run_mac.GGUF_NEEDED[512]}
    with pytest.raises(SystemExit, match="shape_flow_1024.gguf is missing"):
        run_mac.gguf_info(str(tmp_path), 1024)
    (tmp_path / "ss_flow.gguf").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="ss_flow.gguf is missing or not the file"):
        run_mac.gguf_info(str(tmp_path), 512)
    (tmp_path / "ss_flow.gguf").write_bytes(b"ss_flow")
    (tmp_path / "DINOv3-LICENSE.md").unlink()
    with pytest.raises(SystemExit, match="DINOv3-LICENSE.md"):
        run_mac.gguf_info(str(tmp_path), 512)


def test_object_json_names_the_trellis_cpp_route():
    asset = dict(pose=dict(chosen="trellis-up", tilt_deg=0.0, yaw_deg=0.0), size_m=0.4, bbox_m_gltf_xyz=[0.4, 0.3, 0.2],
                 height_m=0.3, output=dict(md5="a" * 32), input=dict(md5="b" * 32))
    rec = dict(backend="trellis-cpp", seed=0, gpu="Apple M6 (Metal)", model="microsoft/TRELLIS.2-4B@x (converted to GGUF f16/f32)",
               trellis_cpp=dict(repo=run_mac.TRELLIS_CPP_REPO, commit=run_mac.TRELLIS_CPP_COMMIT, ggml_commit="g", cli_sha256="c"),
               gguf={"ss_flow.gguf": dict(sha256="f" * 64, sources=["x"])}, cutout=dict(code="bc", weights="bw", seconds=1.0),
               profile=dict(name="draft"), export=dict(texture_size=1024, decimation_target=150000))
    obj = post.object_record("box", dict(size_m=0.4), asset, rec, post.run_pins())
    assert obj["backend"].startswith("trellis-cpp (Apple Silicon, experimental")
    assert obj["licences"] == post.LICENCES_TRELLIS_CPP and "uvrast (texture bake), trianglebvh (BVH)" not in obj["licences"]
    assert obj["pins"] == dict(trellis2_weights=rec["model"], trellis_cpp=dict(repo=run_mac.TRELLIS_CPP_REPO, commit=run_mac.TRELLIS_CPP_COMMIT,
                               ggml_commit="g"), gguf_sha256_12={"ss_flow.gguf": "f" * 12}, birefnet=dict(code="bc", weights="bw"))
    assert obj["settings"] == dict(profile="draft", pipeline_type=None, texture_size=1024, face_target=150000, export_note=None)
    assert not any(isinstance(v, str) and v.startswith("/") for v in json.dumps(obj).split('"'))


# Stands in for $MAC_VENV/bin/python: logs its arguments, then acts like mac/run_mac.py or post_step.py.
FAKE_MAC_PYTHON = r'''#!{python}
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps(dict(args=args, env={{k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "ATTN_BACKEND")}})) + "\n")
def opt(name):
    return args[args.index(name) + 1]
if args[1].endswith("run_mac.py"):
    props, gold, out = opt("--job").split(":")
    od = os.path.join(out, opt("--only"))
    os.makedirs(od, exist_ok=True)
    if os.environ.get("FAKE_OOM") == opt("--profile"):
        json.dump(dict(error="out of memory"), open(os.path.join(od, "oom-generate.json"), "w"))
        sys.exit(3)
    open(os.path.join(od, "raw.glb"), "wb").write(b"glTF")
    json.dump(dict(backend="trellis-cpp", gpu="Apple M6 (Metal)", seconds=dict(trellis_cpp_wall=9.0), export_note="as configured"),
              open(os.path.join(od, "record.json"), "w"))
else:
    od = os.path.join(opt("--dir"), opt("--only"))
    open(os.path.join(od, "asset.glb"), "wb").write(b"glTF")
    json.dump(dict(pose=dict(chosen="trellis-up"), checks=dict(passed=True)), open(os.path.join(od, "asset.json"), "w"))
'''


@pytest.fixture
def mac(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    fake = venv / "bin" / "python"
    fake.write_text(FAKE_MAC_PYTHON.format(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "tc" / "build").mkdir(parents=True)
    (tmp_path / "tc" / "build" / "trellis-cli").write_bytes(b"")
    (tmp_path / "gguf").mkdir()
    (tmp_path / "gguf" / "manifest.json").write_text("{}")
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "Brass Lantern.png").write_bytes(b"png")
    log = tmp_path / "calls.jsonl"
    env = dict(MAC_VENV=str(venv), TRELLIS_CPP_HOME=str(tmp_path / "tc"), TRELLIS_GGUF=str(tmp_path / "gguf"),
               T2_SRC=str(tmp_path / "src"), HF_HOME=str(tmp_path / "hf"), FAKE_LOG=str(log))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    for k in ("FAKE_OOM", "T2_VENV", "T2_WEIGHTS"):
        monkeypatch.delenv(k, raising=False)
    calls = lambda: [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return tmp_path, calls


def mac_cli(tmp_path, *extra):
    return cli_module.main([str(tmp_path / "images" / "Brass Lantern.png"), "0.35", "--out", str(tmp_path / "out"),
                            "--backend", "trellis-cpp", *extra])


def test_backend_trellis_cpp_runs_run_mac_then_the_post_step(mac, capsys):
    tmp_path, calls = mac
    assert mac_cli(tmp_path, "--profile", "draft", "--seed", "0") == cli_module.EXIT_OK
    job = tmp_path / "out" / "brass-lantern-draft-trellis-cpp-seed0"
    run_call, post_call = calls()
    assert run_call["args"][1].endswith(os.path.join("mac", "run_mac.py"))
    assert run_call["args"][2:] == ["--job", f"{job}/props.json:{tmp_path}/images:{job}", "--only", "brass-lantern", "--seed", "0", "--profile", "draft"]
    assert run_call["env"] == dict(HF_HUB_OFFLINE="1", ATTN_BACKEND=None)
    assert post_call["args"][1].endswith(os.path.join("post", "post_step.py"))
    assert (job / "brass-lantern" / "asset.glb").exists()
    assert "experimental, not production, not byte-identical" in capsys.readouterr().out


def test_backend_trellis_cpp_refuses_best_and_chunk_norm(mac):
    tmp_path, _ = mac
    for extra in (["--profile", "best"], ["--chunk-norm"]):
        with pytest.raises(SystemExit):
            mac_cli(tmp_path, *extra)


def test_backend_trellis_cpp_names_the_missing_setup_step(mac, capsys, monkeypatch):
    tmp_path, calls = mac
    (tmp_path / "gguf" / "manifest.json").unlink()
    assert mac_cli(tmp_path) == cli_module.EXIT_ERROR
    assert "mac/setup_mac.sh gguf" in capsys.readouterr().err and not calls()
    monkeypatch.delenv("TRELLIS_GGUF")
    assert mac_cli(tmp_path) == cli_module.EXIT_ERROR
    assert "not set: TRELLIS_GGUF" in capsys.readouterr().err


def test_backend_trellis_cpp_oom_offers_a_draft_on_the_same_route(mac, capsys, monkeypatch):
    tmp_path, _ = mac
    monkeypatch.setenv("FAKE_OOM", "clean")
    assert mac_cli(tmp_path) == cli_module.EXIT_OOM
    out = capsys.readouterr().out
    assert "--backend trellis-cpp --profile draft" in out and "--chunk-norm" not in out


def test_seed_0_stops_because_trellis_cpp_makes_it_random():
    with pytest.raises(SystemExit, match="random seed"):
        run_mac.main(["--job", "p.json:g:o", "--seed", "0"])
