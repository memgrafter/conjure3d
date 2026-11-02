# Contributing

Issues and pull requests are welcome.

## Rules for changes

- **Keep outputs byte-identical.** The pipeline's main promise is that the same input, seed, GPU model and pins give
  the same bytes. A change that can change output bytes (a pin, a default, the patch, the order of operations in the
  runner) must say so, and must include the result of `tools/check_work_identical.sh` before and after.
- **Pins stay pinned.** Every package in `trellis2/requirements.txt` and `imagegen/requirements.txt` is pinned, and
  setup.sh installs it from a lock with file hashes, from one index. After a pin change run
  `bash tools/lock_requirements.sh` and commit the `requirements-*.lock` files. Every clone and download is at a
  fixed commit or revision; weight files have SHA-256 checks. Do not add unpinned dependencies or downloads without a
  checksum.
- **Licences.** Only add dependencies whose licences allow commercial use, and list them in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Never add nvdiffrast, nvdiffrec, cubvh, briaai/RMBG-2.0 or
  other non-commercial parts. uvrast and trianglebvh are clean-room replacements for nvdiffrast's two bake calls and
  for cubvh's distance query; do not read nvdiffrast, nvdiffrec, cubvh or instant-ngp source code.
- **No personal data or machine paths** in files or commit messages: no home folders, host names, user names, e-mail
  addresses or tokens. Configuration goes in environment variables (`config.example.env`).
- **Shell scripts** delete only literal paths (no globs or variables in `rm`), and never pipe a download into a shell.
- **Numbers** in the docs come from logged runs, with the hardware named. Do not add estimates.

## Before you open a pull request

```sh
uv run --no-project --python 3.12 --with pytest --with numpy==2.5.4 --with trimesh==5.2.0 --with scipy==1.18.1 \
  --with networkx==3.7 --with opencv-python-headless==5.0.0.93 -- python -I -m pytest -q tests
bash tools/scan.sh          # secrets, personal data and binary checks (needs gitleaks)
```

If you change the runner, the setup or the post step, also run `tools/check_work_identical.sh` on a GPU against
outputs made before your change ([docs/WORK_IDENTICAL.md](docs/WORK_IDENTICAL.md)).
