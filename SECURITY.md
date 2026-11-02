# Security

## Report a problem

Use GitHub's private vulnerability reporting: the repository's **Security** tab, then **Report a vulnerability**.
Please do not open a public issue for a security problem. Say what is affected (file, command, version or commit),
how to reproduce it, and what an attacker could do with it.

## What this repository does that matters for security

- It downloads code and weights: git clones at pinned commits, Python wheels at pinned versions, Hugging Face files at
  pinned revisions. Weight files are checked by SHA-256 before use; the runner checks them again at every start.
- It builds CUDA extensions from source (TRELLIS.2's o-voxel, CuMesh with one patch, FlexGEMM).
- No code from a model repository runs (no `trust_remote_code`). BiRefNet's code comes from its pinned GitHub commit.
- Python entry points are meant to run with `python -I` (isolated mode), so files in the current folder are not
  imported by accident.
- Scans of this repository (secrets, personal data, binaries, dependencies): [SECURITY_SCAN.md](SECURITY_SCAN.md).
- Treat images, GLBs and job files from other people as untrusted input.
