"""Compare two output folders of the pipeline file by file (MD5), to check that two runs are work-identical.

  python3 -I tools/compare_outputs.py A_DIR B_DIR [--mesh-a DIR --mesh-b DIR] [--only id,id] [--json OUT.json]

A_DIR and B_DIR are run.py OUT folders (<id>/cut.png, <id>/raw.glb, ...). --mesh-a/--mesh-b are run.py --save-mesh
folders (<id>.pt, the decoded mesh). Compared per id, where both sides have the file: the cut-out (cut.png,
mask-birefnet.png, mask.png), TRELLIS.2's input (input-trellis.png), the decoded mesh (<id>.pt), raw.glb, asset.glb
(post step) and web.glb / proxy.glb (web export). For a GLB that differs, the JSON and BIN chunks are compared too, so
a difference in metadata only (for example asset.extras) is told apart from different geometry or textures.
Exit status: 0 when every compared file matches and at least one file was compared; 1 otherwise. Standard library only.
"""
import argparse, hashlib, json, os, struct, sys

FILES = ("cut.png", "mask-birefnet.png", "mask.png", "input-trellis.png", "raw.glb", "asset.glb", "web.glb", "proxy.glb")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def glb_chunks(path):
    """-> {"json": md5 of the JSON chunk, "bin": md5 of the BIN chunk} (None for a chunk that is absent)."""
    out = {"json": None, "bin": None}
    with open(path, "rb") as f:
        magic, version, total = struct.unpack("<4sII", f.read(12))
        if magic != b"glTF" or version != 2:
            raise ValueError(f"{path}: not a glTF 2 binary file")
        while f.tell() < total:
            length, kind = struct.unpack("<II", f.read(8))
            data = f.read(length)
            name = {0x4E4F534A: "json", 0x004E4942: "bin"}.get(kind)
            if name and out[name] is None:
                out[name] = hashlib.md5(data).hexdigest()
    return out


def compare_file(a, b):
    row = dict(a=md5(a), b=md5(b))
    row["match"] = row["a"] == row["b"]
    if not row["match"] and a.endswith(".glb"):
        ca, cb = glb_chunks(a), glb_chunks(b)
        row["json_chunk_match"], row["bin_chunk_match"] = ca["json"] == cb["json"], ca["bin"] == cb["bin"]
    return row


def compare(dir_a, dir_b, mesh_a="", mesh_b="", only=()):
    ids_a = {d for d in os.listdir(dir_a) if os.path.isdir(os.path.join(dir_a, d))}
    ids_b = {d for d in os.listdir(dir_b) if os.path.isdir(os.path.join(dir_b, d))}
    ids = sorted((ids_a & ids_b) if not only else set(only))
    rows = []
    for pid in ids:
        pairs = [(name, os.path.join(dir_a, pid, name), os.path.join(dir_b, pid, name)) for name in FILES]
        if mesh_a and mesh_b:
            pairs.append(("mesh.pt", os.path.join(mesh_a, f"{pid}.pt"), os.path.join(mesh_b, f"{pid}.pt")))
        for name, a, b in pairs:
            if os.path.isfile(a) and os.path.isfile(b):
                rows.append(dict(id=pid, file=name, **compare_file(a, b)))
            elif os.path.isfile(a) or os.path.isfile(b):
                rows.append(dict(id=pid, file=name, a=md5(a) if os.path.isfile(a) else None,
                                 b=md5(b) if os.path.isfile(b) else None, match=None))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--mesh-a", default="")
    ap.add_argument("--mesh-b", default="")
    ap.add_argument("--only", default="", help="comma list of ids (default: the ids in both folders)")
    ap.add_argument("--json", default="", help="write the rows to this file")
    a = ap.parse_args(argv)
    if bool(a.mesh_a) != bool(a.mesh_b):
        ap.error("--mesh-a and --mesh-b go together")
    rows = compare(a.a, a.b, a.mesh_a, a.mesh_b, [x for x in a.only.split(",") if x])
    for r in rows:
        state = {True: "same", False: "DIFFERENT", None: "one side only"}[r["match"]]
        extra = ""
        if "bin_chunk_match" in r:
            extra = f"  (JSON chunk {'same' if r['json_chunk_match'] else 'different'}, BIN chunk {'same' if r['bin_chunk_match'] else 'different'})"
        print(f"{r['id']:<28} {r['file']:<18} {state:<13} {r['a'] or '-':<32} {r['b'] or '-':<32}{extra}")
    compared = [r for r in rows if r["match"] is not None]
    differ = [r for r in compared if not r["match"]]
    print(f"compared {len(compared)} files: {len(compared) - len(differ)} same, {len(differ)} different; "
          f"{len(rows) - len(compared)} on one side only")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0 if compared and not differ else 1


if __name__ == "__main__":
    sys.exit(main())
