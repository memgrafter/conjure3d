# Local patches

`trellis2/setup.sh` applies these patches to pinned upstream clones before it builds them. It stops if a patch does
not apply cleanly, and it records each patch's SHA-256 in the venv. `trellis2/run.py` checks the records, and every
`record.json` names the patch.

| Patch (`trellis2/patches/`) | Upstream, pinned commit | What it changes | Why |
|---|---|---|---|
| `cumesh-deterministic.patch` | CuMesh `12289e10` | sorts each CSR adjacency segment after its atomic fill; sums chart perimeters in a fixed order | the same mesh gives the same GLB bytes every run |
| `cumesh-no-cubvh.patch` | CuMesh `12289e10` | removes the cubvh extension from `setup.py`; adds a stand-in `cumesh/_cubvh.py` | cubvh has a non-commercial licence; trianglebvh replaces it |
| `flexgemm-autotuner-threadsafe.patch` | FlexGEMM `6dd94a85` | every autotuner call holds one process-wide lock | `run.py --inflight N > 1` (below) |

## FlexGEMM: autotuner thread safety

**The bug.** FlexGEMM's autotuners are module-level objects, one per kernel, shared by all threads.
`TritonPersistentCacheAutotuner.run` (`flex_gemm/utils/autotuner.py`) keeps the call's arguments in `self.nargs`: it
sets them at the start, reads them for the tuning key, the pre-hook and the benchmark (triton's `Autotuner._bench`
reads `self.nargs` too), and sets them to `None` at the end. With two threads, one thread can read the other thread's
arguments, or `None`:

- `TypeError: 'NoneType' object is not a mapping` (seen 3 times in 8 minutes on RTX 3090s, `--inflight 2`);
- no error, but a kernel config chosen for the other thread's shape; the config can change the sum order, so the
  output bytes are not certain;
- on a tuning-cache miss both threads tune the same key, and the second result replaces the first.
  `PersistentCacheAutoTuner` (the split-K tuners) has this cache race too.

**The patch.** `_AUTOTUNE_LOCK = threading.RLock()` in `autotuner.py`. `TritonPersistentCacheAutotuner.run` and
`.warmup`, `PersistentCacheAutoTuner.__call__`, `save_autotune_cache` and `load_autotune_cache` hold it. It is
reentrant because the split-K tuners call the Triton tuners inside.

**Effect.** One thread: the same code runs in the same order, so the bytes do not change. Two threads: the threads take
turns only for the CPU part of an autotuned kernel launch. The GPU work goes into the queue as before. When a tuning-cache
miss runs its benchmark, the other thread's FlexGEMM calls wait until the benchmark ends.

**How it is applied.** `setup.sh env` (new venv) or `setup.sh flexgemm` (an existing venv built by `setup.sh env`):
it clones FlexGEMM at `FLEXGEMM_SHA`, runs `git apply --check` and then applies the patch, and checks that the clone's
`flex_gemm/` diff is the patch exactly. Then it builds FlexGEMM and writes `$T2_VENV/flexgemm-build.json` (commit,
patch SHA-256, the SHA-256 of the installed `autotuner.py`). `run.py` (`flexgemm_provenance`) checks the lock in the
imported module, the record, and the clone. It writes the result to `record.json` as `"flexgemm"`.
`--inflight N > 1` stops if FlexGEMM is not the patched build. One object at a time runs with either build.

**Tests.** `tests/test_flexgemm_patch.py` (CPU only) applies the patch to
`tests/fixtures/flexgemm_autotuner_6dd94a85.py`, which is the upstream file without changes. Two threads drive the
patched autotuner with stand-in kernels: no errors, no wrong arguments or configs, each key tuned once. The same test
must find the race in the upstream file. A third test checks that one thread gives the same results with and without
the patch. `test_patch_was_verified_at_the_pinned_commit` fails if the FlexGEMM pin, the fixture or the patch changes
and the `VERIFIED` values in the test do not change too.

**On each FlexGEMM upgrade (or a triton upgrade, since `_bench` reads `self.nargs`):**

1. In a clone, read `git log -p <old>..<new> -- flex_gemm/utils/autotuner.py`. Do the autotuners still keep per-call
   state on the shared object (`self.nargs`, the caches) without a lock?
2. If they do: run `git apply --check` with the patch on the new commit. If it fails, make the same change again by
   hand and write the patch again with the flags in `run.py` `FLEXGEMM_DIFF` (`git diff ... -- flex_gemm`). Copy the
   new upstream `autotuner.py` to `tests/fixtures/` (short commit in the name). Then update `FLEXGEMM_SHA`,
   `FLEXGEMM_PATCH_SHA256` and `FLEXGEMM_AUTOTUNER_BLOB` (`git rev-parse <commit>:flex_gemm/utils/autotuner.py`) in
   `setup.sh` and `run.py`, and `VERIFIED` and `FIXTURE` in the test. Run the CPU tests: the race test must fail on
   the new upstream file and pass on the patched file.
3. If upstream fixed it: remove the patch file, the fixture, the patch steps in `setup.sh` (`patch_flexgemm`,
   `flexgemm_record`, `flexgemm_check`, mode `flexgemm`) and `FLEXGEMM_PATCH*` in `run.py`. Then change
   `flexgemm_provenance` and the `--inflight` stop to accept the fixed upstream commit, and keep the race test,
   pointed at the new upstream file.
4. Then, on a GPU, do the md5 check with `--inflight 2` against one-at-a-time outputs ([WORK_IDENTICAL.md](WORK_IDENTICAL.md))
   before production uses the new build.
