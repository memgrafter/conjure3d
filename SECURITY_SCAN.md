# Security and privacy scans

Scans of this repository before its first release, 2026-11-01, on the commit that adds this file; checks 1-4 again on
2026-11-01 evening after the trianglebvh, cubvh and object.json changes (same results). Run them again with
`bash tools/scan.sh` (checks 1-4) before every release.

| Check | Tool and version | Scope | Result |
|---|---|---|---|
| Secrets | gitleaks 8.30.1 (`gitleaks dir` and `gitleaks git`) | working tree; every commit | no leaks found (working tree; all commits) |
| Personal and machine data | `tools/scan.sh` (grep): e-mail addresses, home folders, IPv4 addresses, token formats, private keys, `key = "..."` assignments, plus a private word list kept outside the repository (names, user names, host names, account names, internal ticket prefixes, project words) | tracked files; every commit's message and diff | nothing found in the files or in any commit message or diff |
| Commit metadata | `git log` | authors and committers | one identity, the owner's GitHub name and e-mail address (the same identity as in the owner's published uvrast repository); no other names or addresses |
| Binary files | `tools/scan.sh` | tracked files | none: the repository has no images, GLBs, weights or other binaries, so there is no EXIF or GLB metadata to check |
| Shell safety | `tools/scan.sh` | `*.sh` | no `rm` with globs or variables; no download piped into a shell |
| Python isolation | `tools/scan.sh` | `*.sh`, `*.md`, `*.py` | every documented or scripted Python entry point runs with `-I` |
| Downloads | review of `trellis2/setup.sh`, `imagegen/setup.sh`, `trellis2/run.py` | all downloads | every clone at a pinned commit; every Python package pinned (`==` or a git commit) and, except the git ones, installed with `--require-hashes` from a hashed lock (`requirements-*.lock`), each package from one index (torch, torchvision, triton, xformers: PyTorch cu130; all others: PyPI); trianglebvh at a pinned commit from `TRIANGLEBVH_GIT` (default `https://github.com/memgrafter/trianglebvh.git`); CuMesh's cubvh submodule never fetched; every Hugging Face download at a pinned revision; weight files checked by SHA-256 (20 TRELLIS.2/DINOv3 files at install and at every run start, BiRefNet at install and every run start, 19 FLUX.2-klein-4B files at install); no `trust_remote_code` |
| Python dependencies | pip-audit 2.9.0 (OSV/PyPI advisories, `--no-deps --disable-pip` on the pinned files; torch and torchvision audited as 2.14.1 and 0.29.1 because the `+cu130` builds are not on PyPI) | `trellis2/requirements.txt`, `imagegen/requirements.txt` | see below |
| Web dependencies | bun audit (bun 1.4.4) | `web/bun.lock`, 47 packages | no vulnerabilities found |

## Known advisories (not fixed on purpose)

pip-audit reports 6 advisories for **transformers 4.57.6** (the TRELLIS.2 venv; the image venv's transformers 5.22.1
has none):

| Advisory | What it is about | Fixed in | Reached by this pipeline? |
|---|---|---|---|
| PYSEC-2025-217 (CVE-2025-14929) | X-CLIP checkpoint conversion script deserializes untrusted data | none listed | no: conversion scripts are not used |
| PYSEC-2026-2288 (CVE-2026-1839) | `Trainer._load_rng_state` uses `torch.load` without `weights_only` | 5.0.0 | no: no training |
| PYSEC-2026-2289 (CVE-2026-4372) | a crafted `config.json` can run code at model load | 5.3.0 | only through a config file: the one config this pipeline loads (DINOv3's) is SHA-256 checked against the pinned file |
| PYSEC-2026-2290 (CVE-2026-5241) | LightGlue model loading with `trust_remote_code` | none listed | no: no LightGlue, no `trust_remote_code` |
| PYSEC-2026-3929 (CVE-2026-9856) | path traversal in tokenizer and processor `save_pretrained` | 5.10.0 | no: nothing is saved with transformers |
| PYSEC-2026-4174 (CVE-2026-80047) | `load_custom_generate` writes remote Python files to disk | none listed | no: not called; the runner runs offline (`HF_HUB_OFFLINE=1`) |

The pins are not upgraded: a newer transformers (5.x needs huggingface_hub 1.x) can change the DINOv3 features and so
every output byte. An upgrade needs its own work-identical check.

## What the scans do not cover

- The third-party code and weights that `setup.sh` downloads (they are not in this repository); only their pins and
  checksums were reviewed. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for their licences.
- Files you make with the pipeline: `record.json` and `run.json` record local paths (venv, sources, weights folder,
  output folders) and the GPU's UUID. Remove them, or those fields, before you publish outputs. GLBs from `web/` carry
  no names, extras or tool strings (`web/verify_export.mjs` checks this); `asset.glb` holds the object id (a node
  name) and a note in `asset.extras`. `object.json` holds no local paths (tested); it has the props file's `noun`,
  `description` and `material` text if you give them.
