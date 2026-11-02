"""run.py's command line, defaults and optional power check (no GPU, no torch: only the parts that run before them)."""
import json, subprocess, sys
import pytest
from conftest import load

run = load("trellis2/run.py", "t2_run")


def test_defaults_are_the_pipeline_settings():
    a = run.parse_args(["--job", "p.json:gold:out", "--weights", "/w"])
    assert (a.pipeline_type, a.seed, a.texture_size) == ("1024_cascade", 42, 4096)
    assert (a.bake, a.offload, a.host_weights, a.remesh, a.trim, a.bvh) == ("uvrast", "persistent", "mapped", "lean", "on", "ours")
    assert a.job == ["p.json:gold:out"] and not a.save_mesh and not a.hash_stages
    assert a.vram_cap_gb == 0 and a.decimation_target == 500000
    assert a.chunk_norm == 0 and a.export_from == [] and a.export_out == ""


def test_export_settings_are_pinned():
    assert run.EXPORT == dict(decimation_target=500000, remesh=True, remesh_band=1, remesh_project=0)
    assert run.PRESIMPLIFY == 4000000 and run.LARGEST_DILATE == 17 and run.BIREFNET_SIZE == (1024, 1024)
    assert sorted(run.NEEDED) == ["1024_cascade", "1536_cascade", "512"]
    assert len(run.WEIGHT_FILES) == 20
    assert all(len(v[3]) == 64 and int(v[3], 16) >= 0 for v in run.WEIGHT_FILES.values())


def test_clean_profile_is_the_old_default():
    a = run.parse_args(["--job", "p.json:g:o", "--weights", "/w"])
    assert a.profile == "clean" and a.profile_record == dict(name="clean", overridden=[], values=dict(
        pipeline_type="1024_cascade", texture_size=4096, decimation_target=run.EXPORT["decimation_target"]))


@pytest.mark.parametrize("profile,want", [("draft", ("512", 1024, 100000)), ("best", ("1536_cascade", 4096, 500000))])
def test_profiles_set_three_values(profile, want):
    a = run.parse_args(["--job", "p.json:g:o", "--weights", "/w", "--profile", profile])
    assert (a.pipeline_type, a.texture_size, a.decimation_target) == want and a.profile_record["overridden"] == []


def test_a_flag_overrides_its_profile_value():
    a = run.parse_args(["--job", "p.json:g:o", "--weights", "/w", "--profile", "draft", "--texture-size", "4096"])
    assert (a.pipeline_type, a.texture_size, a.decimation_target) == ("512", 4096, 100000)
    assert a.profile_record["overridden"] == ["texture_size"]


def test_every_profile_has_its_weights():
    files = " ".join(run.WEIGHT_FILES)
    assert "slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors" in files and set(run.NEEDED) == {p["pipeline_type"] for p in run.PROFILES.values()}


def test_weights_default_from_env(monkeypatch):
    monkeypatch.setenv("T2_WEIGHTS", "/from/env")
    assert run.parse_args(["--job", "p.json:g:o"]).weights == "/from/env"


def test_weights_required(monkeypatch):
    monkeypatch.delenv("T2_WEIGHTS", raising=False)
    with pytest.raises(SystemExit):
        run.parse_args(["--job", "p.json:g:o"])
    assert run.parse_args(["--job", "p.json:g:o", "--cutout-only"]).cutout_only


@pytest.mark.parametrize("argv", [
    ["--weights", "/w"],                                               # no --job and no --queue
    ["--weights", "/w", "--queue", "q", "--log", "l"],                 # --queue without --stream
    ["--weights", "/w", "--queue", "q", "--stream", "s"],              # --queue without --log
    ["--weights", "/w", "--queue", "q", "--stream", "s", "--log", "l", "--job", "p:g:o"],
    ["--weights", "/w", "--job", "p:g:o", "--bake", "nvdiffrast"],     # not offered
    ["--weights", "/w", "--job", "p:g:o", "--bvh", "cubvh"],           # not offered
    ["--weights", "/w", "--job", "p:g:o", "--pipeline-type", "1536"],
    ["--weights", "/w", "--job", "p:g:o", "--chunk-norm", "-1"],
    ["--weights", "/w", "--job", "p:g:o", "--chunk-norm", "1.5"],
    ["--export-from", "m.pt"],                                         # no --export-out
    ["--weights", "/w", "--job", "p:g:o", "--export-out", "x"],        # --export-out without --export-from
    ["--export-from", "m.pt", "--export-out", "x", "--job", "p:g:o"],  # export runs alone
    ["--export-from", "m.pt", "--export-out", "x", "--queue", "q", "--stream", "s", "--log", "l"],
    ["--export-from", "m.pt", "--export-out", "x", "--cutout-only"],
])
def test_bad_command_lines_stop(argv):
    with pytest.raises(SystemExit):
        run.parse_args(argv)


def test_power_check_is_off_by_default(monkeypatch):
    monkeypatch.delenv("GPU_GUARD_EXPECT_POWER_W", raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("nvidia-smi must not run"))
    assert run.power_check() is None


def test_power_check_stops_on_other_limit(monkeypatch):
    class Out:
        stdout = "0, 275.00\n1, 300.00\n"
    monkeypatch.setenv("GPU_GUARD_EXPECT_POWER_W", "275")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Out())
    with pytest.raises(SystemExit, match="not 275 W"):
        run.power_check()
    Out.stdout = "0, 275.00\n1, 275.20\n"
    assert run.power_check() is None


def test_job_items_only_filter(tmp_path):
    props = tmp_path / "props.json"
    props.write_text(json.dumps({"props": {"a": {"gold": "a.png"}, "b": {"gold": "b.png"}, "c": {"gold": "c.png"}}}))
    jobs = [(str(props), "gold", "out")]
    assert [x[0] for x in run.job_items(jobs, "")] == ["a", "b", "c"]
    assert [x[0] for x in run.job_items(jobs, "c,a")] == ["a", "c"]


def test_largest_piece_rule():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    a = np.zeros((64, 64), "float32")
    a[5:30, 5:30] = 1.0      # large piece
    a[50:55, 50:55] = 1.0    # small piece, far away: removed
    out, parts = run.largest_part(a)
    assert parts == 2 and out[10, 10] == 1.0 and out[52, 52] == 0.0
    one, parts = run.largest_part(a[:40, :40])
    assert parts == 1 and (one == a[:40, :40]).all()


def test_chunk_norm_flag():
    a = run.parse_args(["--job", "p.json:g:o", "--weights", "/w", "--chunk-norm", "1048576"])
    assert a.chunk_norm == 1048576 and a.job == ["p.json:g:o"]


def test_export_from_needs_no_job_or_weights(monkeypatch):
    monkeypatch.delenv("T2_WEIGHTS", raising=False)
    a = run.parse_args(["--export-from", "m/a.pt", "m/b.pt", "--export-out", "x", "--profile", "draft"])
    assert a.export_from == ["m/a.pt", "m/b.pt"] and a.export_out == "x" and a.job == []
    assert (a.texture_size, a.decimation_target) == (1024, 100000)   # the export uses the profile's settings


def test_chunk_norm_gives_the_unchunked_rows(monkeypatch):
    np = pytest.importorskip("numpy")
    import types

    class LayerNorm32:
        calls = []
        def forward(self, x):
            LayerNorm32.calls.append(x.shape[0])
            mu, sd = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True) + 1e-6
            return (x - mu) / sd

    norm = types.SimpleNamespace(LayerNorm32=LayerNorm32)
    modules = types.SimpleNamespace(norm=norm)
    monkeypatch.setitem(sys.modules, "trellis2", types.SimpleNamespace(modules=modules))
    monkeypatch.setitem(sys.modules, "trellis2.modules", modules)
    monkeypatch.setitem(sys.modules, "trellis2.modules.norm", norm)
    x = np.random.default_rng(0).standard_normal((10, 4)).astype("float32")
    want = LayerNorm32().forward(x)
    run.chunk_norm(types.SimpleNamespace(empty_like=np.empty_like), 4)
    LayerNorm32.calls.clear()
    assert (LayerNorm32().forward(x) == want).all() and LayerNorm32.calls == [4, 4, 2]
    LayerNorm32.calls.clear()
    assert (LayerNorm32().forward(x[:4]) == want[:4]).all() and LayerNorm32.calls == [4]
