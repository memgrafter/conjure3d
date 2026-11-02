"""FlexGEMM thread-safety patch (trellis2/patches/flexgemm-autotuner-threadsafe.patch, docs/PATCHES.md).

The fixture is FlexGEMM's flex_gemm/utils/autotuner.py at the pinned commit, byte for byte (MIT, THIRD_PARTY_NOTICES.md).
The patch must apply to it; two threads on one patched autotuner must never use each other's arguments or tune one
key twice; the same race test on the unpatched file must fail (so the test can see the bug). torch, triton and filelock
are small stand-ins: no GPU needed.
"""
import hashlib, importlib.util, os, re, subprocess, sys, threading, time, types
import pytest
from conftest import ROOT, load

run = load("trellis2/run.py", "t2_run_flexgemm")
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "flexgemm_autotuner_6dd94a85.py")
PATCH = os.path.join(ROOT, "trellis2", run.FLEXGEMM_PATCH)
# The FlexGEMM commit the patch and this race test were last checked against. On a FlexGEMM upgrade follow
# docs/PATCHES.md, then change these values; until then test_patch_was_verified_at_the_pinned_commit fails.
VERIFIED = dict(commit="6dd94a859c26ee8246888502eada3dd8ad85532e",
                autotuner_blob="ca95fc5843ec71ca064e0aa162c327f31b49f822",   # git blob id of the upstream file
                patch_sha256="dc182f4610002a3761df9ada8ef398a25ae2ce94489df83f19b8a6868ba4aa65")
THREADS, CALLS, BLOCKS = 2, 150, (1, 2, 3, 4)


def git_blob_id(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def test_patch_was_verified_at_the_pinned_commit():
    setup = open(os.path.join(ROOT, "trellis2", "setup.sh")).read()
    pinned = re.search(r"^FLEXGEMM_SHA=([0-9a-f]+)", setup, re.M).group(1)
    assert pinned == run.FLEXGEMM_SHA == VERIFIED["commit"], "FlexGEMM pin changed: re-verify the patch (docs/PATCHES.md)"
    assert git_blob_id(FIXTURE) == run.FLEXGEMM_AUTOTUNER_BLOB == VERIFIED["autotuner_blob"]
    patch_sha = hashlib.sha256(open(PATCH, "rb").read()).hexdigest()
    assert patch_sha == run.FLEXGEMM_PATCH_SHA256 == VERIFIED["patch_sha256"]
    assert re.search(rf"^FLEXGEMM_PATCH_SHA256={patch_sha}", setup, re.M)


@pytest.fixture
def patched_file(tmp_path):
    d = tmp_path / "flex_gemm" / "utils"
    d.mkdir(parents=True)
    (d / "autotuner.py").write_bytes(open(FIXTURE, "rb").read())
    for check in (["--check"], []):
        subprocess.run(["git", "apply", *check, PATCH], cwd=tmp_path, check=True)
    return str(d / "autotuner.py")


def load_autotuner(path, monkeypatch, tag):
    """The autotuner module from path, with stand-ins for its imports (as triton 3.8: _bench reads self.nargs)."""
    class Config:
        def __init__(self, kwargs, pre_hook=None):
            self.kwargs, self.pre_hook = dict(kwargs or {}), pre_hook
        def all_kwargs(self):
            return dict(self.kwargs)

    class Autotuner:
        def __init__(self, fn, arg_names, configs, key, reset_to_zero, restore_value, pre_hook=None, post_hook=None,
                     prune_configs_by=None, warmup=None, rep=None, use_cuda_graph=False, do_bench=None):
            self.fn = self.base_fn = fn
            self.arg_names, self.configs, self.keys, self.cache = arg_names, configs, key, {}
            self.pre_hook = lambda kwargs, reset_only=False: 0
            self.early_config_prune = self.perf_model = None
        def _bench(self, *args, config, **meta):
            full = {**self.nargs, **meta, **config.all_kwargs()}
            return abs(config.kwargs["BLOCK"] - (full["n"] % len(BLOCKS) + 1))   # best BLOCK = n % 4 + 1

    triton = types.ModuleType("triton")
    triton.runtime = types.SimpleNamespace(Autotuner=Autotuner, Config=Config)
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(synchronize=lambda: None, get_device_name=lambda: "stand-in")
    filelock = types.ModuleType("filelock")
    filelock.FileLock = lambda path: None
    pkg = types.ModuleType(tag)
    pkg.__path__, pkg.AUTOSAVE_AUTOTUNE_CACHE, pkg.AUTOTUNE_CACHE_PATH = [], False, "unused"
    utils = types.ModuleType(f"{tag}.utils")
    utils.__path__ = []
    for name, mod in (("triton", triton), ("torch", torch), ("filelock", filelock), (tag, pkg), (f"{tag}.utils", utils)):
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location(f"{tag}.utils.autotuner", path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod, Config


class Arg:
    """A tensor stand-in that belongs to one thread; reading .dtype gives the other thread time to run."""
    def __init__(self):
        self.owner = threading.get_ident()
    @property
    def dtype(self):
        time.sleep(0.0005)
        return "float16"


def race(mod, Config):
    """THREADS threads call one Triton autotuner (inside one split-K autotuner) CALLS times each, over the same shapes
    in a different order, each with its own tensors. Returns (errors, wrong-arguments count, wrong-config count, keys tuned more than once)."""
    errors, wrong_args, wrong_config, tuned = [], [], [], {}

    def pre_hook(full_nargs):
        if full_nargs["x"].owner != threading.get_ident():
            wrong_args.append(1)

    class Kernel:
        arg_names = ["x", "n"]
        def run(self, x, n, BLOCK):
            if BLOCK != n % len(BLOCKS) + 1:
                wrong_config.append(1)
            time.sleep(0.0005)

    inner = mod.triton_autotune(configs=[Config({"BLOCK": b}, pre_hook=pre_hook) for b in BLOCKS], key=["n"])(Kernel())

    def configs(x, n, SPLITK=1):
        tuned[n] = tuned.get(n, 0) + 1
        return [{"SPLITK": 1}, {"SPLITK": 2}]

    @mod.autotune(config_fn=configs, key=["n"], warmup=1, runs=2)
    def splitk(x, n, SPLITK=1):
        return inner.run(x, n)   # as FlexGEMM's split-K wrappers launch the Triton kernels

    start = threading.Barrier(THREADS)

    def worker(k):
        start.wait()
        try:
            for i in range(CALLS):
                splitk(Arg(), (i + 3 * k) % 8)   # the same 8 shapes in both threads, not in step
        except Exception as e:
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads), "deadlock"
    return errors, len(wrong_args), len(wrong_config), {n: c for n, c in tuned.items() if c > 1}


def test_patched_autotuner_is_thread_safe(patched_file, monkeypatch):
    mod, Config = load_autotuner(patched_file, monkeypatch, "fg_patched")
    assert isinstance(mod._AUTOTUNE_LOCK, type(threading.RLock()))
    assert race(mod, Config) == ([], 0, 0, {})


def test_unpatched_autotuner_races(monkeypatch):
    """The race test sees the bug in the upstream file (TypeError on a cleared nargs, another thread's arguments, or a
    key tuned twice); if this ever passes cleanly, the race test above proves nothing."""
    mod, Config = load_autotuner(FIXTURE, monkeypatch, "fg_upstream")
    errors, wrong_args, wrong_config, tuned_twice = race(mod, Config)
    assert errors or wrong_args or wrong_config or tuned_twice


def test_patched_autotuner_single_thread_is_unchanged(patched_file, monkeypatch):
    """One thread: the patched and upstream autotuners pick the same configs and call the kernel the same way."""
    seen = {}
    for tag, path in (("fg_one_patched", patched_file), ("fg_one_upstream", FIXTURE)):
        mod, Config = load_autotuner(path, monkeypatch, tag)
        calls = []
        class Kernel:
            arg_names = ["x", "n"]
            def run(self, x, n, BLOCK):
                calls.append((n, BLOCK))
        inner = mod.triton_autotune(configs=[Config({"BLOCK": b}) for b in BLOCKS], key=["n"])(Kernel())
        for n in (3, 5, 3, 8, 5):
            inner.run(Arg(), n)
        seen[tag] = (calls, sorted(inner.cache))
    assert seen["fg_one_patched"] == seen["fg_one_upstream"]


def test_provenance_says_unpatched_without_the_lock(tmp_path):
    f = tmp_path / "autotuner.py"
    f.write_text("# upstream\n")
    info = run.flexgemm_provenance(types.SimpleNamespace(__file__=str(f)))
    assert info["flexgemm_patched"] is False and "lock" in info["reason"]
    assert info["autotuner_sha256"] == hashlib.sha256(b"# upstream\n").hexdigest()
