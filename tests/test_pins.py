"""The pins in setup.sh, run.py, the lean remesh copy and the patch file agree with each other."""
import hashlib, os, re
import pytest
from conftest import ROOT, load

run = load("trellis2/run.py", "t2_run_pins")


def read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def shell_var(text, name):
    m = re.search(rf"^{name}=([0-9a-f]+)", text, re.M)
    assert m, name
    return m.group(1)


def test_setup_and_runner_pins_agree():
    setup = read("trellis2/setup.sh")
    for var, value in [("T2_SHA", run.T2_SHA), ("HF_REV", run.HF_REV), ("TIL_REV", run.TIL_REV), ("DINO_REV", run.DINO_REV),
                       ("CUMESH_SHA", run.CUMESH_SHA), ("CUMESH_PATCH_SHA256", run.CUMESH_PATCH_SHA256),
                       ("BIREFNET_SHA", run.BIREFNET_SHA), ("BIREFNET_REV", run.BIREFNET_REV), ("BIREFNET_SHA256", run.BIREFNET_SHA256),
                       ("TRIANGLEBVH_SHA", run.TRIANGLEBVH_SHA), ("CUMESH_NO_CUBVH_SHA256", run.CUMESH_NO_CUBVH_SHA256)]:
        assert shell_var(setup, var) == value, var


def test_patch_files_are_the_pinned_patches():
    for rel, want in ((run.CUMESH_PATCH, run.CUMESH_PATCH_SHA256), (run.CUMESH_NO_CUBVH, run.CUMESH_NO_CUBVH_SHA256)):
        with open(os.path.join(ROOT, "trellis2", rel), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == want, rel


def test_cubvh_is_never_fetched_or_built():
    setup, patch = read("trellis2/setup.sh"), read("trellis2/" + run.CUMESH_NO_CUBVH)
    assert re.search(r'^  clone https://github.com/JeffreyXiang/CuMesh.git "\$SRC/ext/CuMesh" \$CUMESH_SHA nosub$', setup, re.M)
    assert '-        name="cumesh._cubvh",' in patch and "+++ b/cumesh/_cubvh.py" in patch
    assert not [l for l in patch.splitlines() if l.startswith("+") and "third_party/cubvh" in l]


def test_lean_remesh_is_pinned_to_the_same_cumesh():
    assert re.search(rf'^UPSTREAM_COMMIT = "{run.CUMESH_SHA}"', read("trellis2/cumesh_lean_remesh.py"), re.M)


def test_requirements_are_all_pinned():
    for rel in ("trellis2/requirements.txt", "imagegen/requirements.txt"):
        lines = [l for l in read(rel).splitlines() if l and not l.startswith("#")]
        assert lines and all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^ ]+|[A-Za-z0-9_.-]+ @ git\+https://\S+@[0-9a-f]{40}", l) for l in lines), rel
    assert "uvrast @ git+https://github.com/memgrafter/uvrast@" in read("trellis2/requirements.txt")


def test_no_excluded_parts_are_installed():
    text = read("trellis2/requirements.txt") + read("imagegen/requirements.txt") + read("trellis2/setup.sh") + read("imagegen/setup.sh")
    for name in ("NVlabs/", "nvdiffrast @", "nvdiffrast==", "download briaai", "FLUX.2-klein-9B", "Qwen/Qwen-Image"):
        assert name not in text, name


def test_cubvh_stand_in_stops_on_use():
    """The stand-in module the no-cubvh patch adds: importable, and any use of cubvh stops with a clear error."""
    import pytest, types
    patch = read("trellis2/" + run.CUMESH_NO_CUBVH)
    body = patch.split("+++ b/cumesh/_cubvh.py\n", 1)[1].split("\ndiff --git", 1)[0]
    code = "\n".join(l[1:] for l in body.splitlines() if l.startswith("+"))
    stub = types.ModuleType("cumesh._cubvh")
    exec(code, stub.__dict__)
    with pytest.raises(RuntimeError, match="trianglebvh"):
        stub.cuBVH
    assert not hasattr(stub, "__path__")


def lock_pins(rel):
    """name==version -> SHA-256 list, from a lock file of tools/lock_requirements.sh."""
    pins, name = {}, None
    for line in read(rel).splitlines():
        m = re.match(r"^(\S+==\S+) \\$", line)
        if m:
            name = m.group(1); pins[name] = []
        m = re.match(r"^    --hash=sha256:([0-9a-f]{64})", line)
        if m:
            pins[name].append(m.group(1))
    return pins


@pytest.mark.parametrize("venv,pytorch", [("trellis2", {"torch", "torchvision", "triton", "xformers"}),
                                          ("imagegen", {"torch", "torchvision", "triton"})])
def test_hashed_locks_are_the_pinned_versions(venv, pytorch):
    reqs = {l for l in read(f"{venv}/requirements.txt").splitlines() if l and not l.startswith("#") and " @ " not in l}
    pt, pypi = lock_pins(f"{venv}/requirements-pytorch.lock"), lock_pins(f"{venv}/requirements-pypi.lock")
    assert set(pt) | set(pypi) == reqs and not set(pt) & set(pypi)
    assert {p.split("==")[0] for p in pt} == pytorch
    assert all(pt.values()) and all(pypi.values())   # every package has file hashes


def test_xformers_and_triton_are_the_pytorch_index_files():
    # same file names on PyPI hold other bytes; the venvs were built with these (installed RECORD checked)
    pt = lock_pins("trellis2/requirements-pytorch.lock")
    assert "07a88b13e4f7e8b1769094238ab75109a5db2a8160a24ddc49c24d2efa44cb58" in pt["triton==3.8.0"]
    assert "962eb73f7243fb6a6b68ed85ed8f97780070ee35c1be464eefe3299b0382391d" in pt["xformers==0.0.35"]
    assert "e91ffa46d095b252248297292dd22bcbacd53a125a0c2eefbbbf74925a320bc3" not in pt["triton==3.8.0"]


@pytest.mark.parametrize("rel", ["trellis2/setup.sh", "imagegen/setup.sh"])
def test_setup_installs_with_hashes_from_one_index(rel):
    setup = read(rel)
    assert "unsafe-best-match" not in setup and "--extra-index-url" not in setup
    locks = [l for l in setup.splitlines() if '"$UV" pip install' in l and "--require-hashes" in l]
    assert len(locks) == 2 and all(l.count("--index-url") == 1 for l in locks)
    assert '-r "$HERE/requirements-pypi.lock"' in setup and '-r "$HERE/requirements-pytorch.lock"' in setup
