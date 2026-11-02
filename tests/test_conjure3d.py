"""conjure3d.py (one command for one object) with a fake venv: no GPU, no torch."""
import json, os, stat, sys
import pytest
from conftest import install_weights, load

cli_module = load("conjure3d.py", "conjure3d_cli")

# Stands in for $T2_VENV/bin/python: logs its arguments, then acts like run.py or post_step.py.
FAKE_PYTHON = r'''#!{python}
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps(dict(args=args, env={{k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "ATTN_BACKEND", "PYTORCH_CUDA_ALLOC_CONF")}})) + "\n")
def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default
if args[1].endswith("run.py"):
    props, gold, out = opt("--job").split(":")
    od = os.path.join(out, opt("--only"))
    os.makedirs(od, exist_ok=True)
    ptype = dict(clean="1024_cascade", draft="512", best="1536_cascade")[opt("--profile", "clean")]
    if ptype in os.environ.get("FAKE_OOM", "").split(","):
        json.dump(dict(error="CUDA out of memory"), open(os.path.join(od, f"oom-{{ptype}}.json"), "w"))
        sys.exit(0)
    open(os.path.join(od, "raw.glb"), "wb").write(b"glTF")
    json.dump(dict(gpu="Fake GPU", load=dict(load_s=1.0), export_note="as configured"), open(os.path.join(od, "record.json"), "w"))
else:
    od = os.path.join(opt("--dir"), opt("--only"))
    open(os.path.join(od, "asset.glb"), "wb").write(b"glTF")
    json.dump(dict(pose=dict(chosen="trellis-up"), checks=dict(passed=True)), open(os.path.join(od, "asset.json"), "w"))
'''


@pytest.fixture
def setup(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    fake = venv / "bin" / "python"
    fake.write_text(FAKE_PYTHON.format(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    install_weights(str(tmp_path / "weights"), with_512_texture=True)
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "Brass Lantern.png").write_bytes(b"png")
    log = tmp_path / "calls.jsonl"
    for k, v in dict(T2_VENV=str(venv), T2_WEIGHTS=str(tmp_path / "weights"), HF_HOME=str(tmp_path / "hf"), FAKE_LOG=str(log)).items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("FAKE_OOM", raising=False)
    calls = lambda: [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return tmp_path, calls


def cli(tmp_path, *extra):
    return cli_module.main([str(tmp_path / "images" / "Brass Lantern.png"), "0.35", "--out", str(tmp_path / "out"), *extra])


def test_clean_is_the_runner_default(setup):
    tmp_path, calls = setup
    assert cli(tmp_path) == cli_module.EXIT_OK
    job = tmp_path / "out" / "brass-lantern-clean-seed42"
    run_call, post_call = calls()
    assert run_call["args"][2:] == ["--job", f"{job}/props.json:{tmp_path}/images:{job}", "--only", "brass-lantern", "--seed", "42", "--profile", "clean"]
    assert run_call["env"] == cli_module.RUN_ENV
    assert post_call["args"][2:] == ["--props", f"{job}/props.json", "--dir", str(job), "--only", "brass-lantern"]
    props = json.load(open(job / "props.json"))
    assert props == {"props": {"brass-lantern": {"gold": "Brass Lantern.png", "size_m": 0.35}}}
    assert (job / "brass-lantern" / "asset.glb").is_file() and (job / "run.log").is_file()


def test_draft_passes_the_draft_profile(setup):
    tmp_path, calls = setup
    assert cli(tmp_path, "--profile", "draft", "--seed", "7", "--lie-flat") == cli_module.EXIT_OK
    assert calls()[0]["args"][-4:] == ["--seed", "7", "--profile", "draft"]
    props = json.load(open(tmp_path / "out" / "brass-lantern-draft-seed7" / "props.json"))
    assert props["props"]["brass-lantern"]["rest_up"] == [0, 0, 1]


def test_oom_stops_and_offers_draft(setup, monkeypatch, capsys):
    tmp_path, calls = setup
    monkeypatch.setenv("FAKE_OOM", "1024_cascade")
    assert cli(tmp_path) == cli_module.EXIT_OOM
    assert len(calls()) == 1
    out = capsys.readouterr().out
    assert "OUT OF GPU MEMORY in generation" in out and "--profile draft" in out
    assert out.index("--chunk-norm") < out.index("--profile draft")   # same bytes first, then the draft


def test_chunk_norm_passes_the_proven_rows(setup, monkeypatch, capsys):
    tmp_path, calls = setup
    assert cli(tmp_path, "--chunk-norm") == cli_module.EXIT_OK
    assert calls()[0]["args"][-4:] == ["--profile", "clean", "--chunk-norm", "1048576"]
    assert (tmp_path / "out" / "brass-lantern-clean-chunk-norm-seed42" / "brass-lantern" / "asset.glb").is_file()
    monkeypatch.setenv("FAKE_OOM", "1024_cascade")
    assert cli(tmp_path, "--chunk-norm", "--seed", "7") == cli_module.EXIT_OOM
    out = capsys.readouterr().out
    assert "To try it first" not in out and "--profile draft" in out   # chunk-norm was used: only the draft is left


def test_export_oom_offers_only_draft(setup, monkeypatch, capsys):
    tmp_path, calls = setup
    monkeypatch.setenv("FAKE_OOM", "1024_cascade")
    monkeypatch.setattr(cli_module, "oom_file", lambda od: os.path.join(od, "oom-glb.json"))
    assert cli(tmp_path) == cli_module.EXIT_OOM
    out = capsys.readouterr().out
    assert "in the GLB export" in out and "--chunk-norm" not in out and "--profile draft" in out


def test_oom_makes_draft_when_asked(setup, monkeypatch, capsys):
    tmp_path, calls = setup
    monkeypatch.setenv("FAKE_OOM", "1024_cascade")
    assert cli(tmp_path, "--on-oom", "draft") == cli_module.EXIT_OK
    clean_run, draft_run, post = calls()
    assert clean_run["args"][-2:] == ["--profile", "clean"] and draft_run["args"][-2:] == ["--profile", "draft"]
    assert (tmp_path / "out" / "brass-lantern-draft-seed42" / "brass-lantern" / "asset.glb").is_file()
    assert "Making a draft now" in capsys.readouterr().out


def test_draft_needs_its_model_files(setup, tmp_path, monkeypatch):
    install_weights(str(tmp_path / "weights-no512"), with_512_texture=False)
    monkeypatch.setenv("T2_WEIGHTS", str(tmp_path / "weights-no512"))
    _, calls = setup
    assert cli(tmp_path, "--profile", "draft") == cli_module.EXIT_ERROR
    assert calls() == []


def test_existing_output_is_never_overwritten(setup):
    tmp_path, calls = setup
    (tmp_path / "out" / "brass-lantern-clean-seed42").mkdir(parents=True)
    assert cli(tmp_path) == cli_module.EXIT_ERROR
    assert calls() == []


def test_object_id_from_file_name():
    assert cli_module.object_id("/x/Brass Lantern (v2).PNG") == "brass-lantern-v2"
    assert cli_module.object_id("/x/___.png") == "object"


def test_on_oom_draft_runs_without_chunk_norm(setup, monkeypatch):
    tmp_path, calls = setup
    monkeypatch.setenv("FAKE_OOM", "1024_cascade")
    assert cli(tmp_path, "--chunk-norm", "--on-oom", "draft") == cli_module.EXIT_OK
    clean_run, draft_run, post = calls()
    assert "--chunk-norm" in clean_run["args"] and draft_run["args"][-2:] == ["--profile", "draft"]
