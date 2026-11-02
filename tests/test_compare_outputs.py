"""tools/compare_outputs.py on small fake output folders."""
import json, struct
from conftest import load

cmp_ = load("tools/compare_outputs.py", "compare_outputs")


def glb(js, bin_):
    js = json.dumps(js).encode()
    js += b" " * (-len(js) % 4)
    bin_ += b"\0" * (-len(bin_) % 4)
    return (struct.pack("<4sII", b"glTF", 2, 28 + len(js) + len(bin_)) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(bin_), 0x004E4942) + bin_)


def make(root, pid, files):
    d = root / pid
    d.mkdir(parents=True)
    for name, data in files.items():
        (d / name).write_bytes(data)


def test_identical_folders_pass(tmp_path, capsys):
    for side in "ab":
        make(tmp_path / side, "x", {"cut.png": b"png", "raw.glb": glb({"asset": {}}, b"1234")})
    assert cmp_.main([str(tmp_path / "a"), str(tmp_path / "b")]) == 0
    assert "compared 2 files: 2 same, 0 different" in capsys.readouterr().out


def test_metadata_only_difference_is_located(tmp_path):
    make(tmp_path / "a", "x", {"asset.glb": glb({"asset": {"extras": {"k1": 1}}}, b"geom")})
    make(tmp_path / "b", "x", {"asset.glb": glb({"asset": {"extras": {"k2": 1}}}, b"geom")})
    rows = cmp_.compare(str(tmp_path / "a"), str(tmp_path / "b"))
    assert rows == [dict(id="x", file="asset.glb", a=rows[0]["a"], b=rows[0]["b"], match=False,
                         json_chunk_match=False, bin_chunk_match=True)]
    assert cmp_.main([str(tmp_path / "a"), str(tmp_path / "b")]) == 1


def test_meshes_and_one_sided_files(tmp_path):
    make(tmp_path / "a", "x", {"cut.png": b"1", "raw.glb": glb({}, b"g")})
    make(tmp_path / "b", "x", {"cut.png": b"1"})
    for side in "ab":
        (tmp_path / f"m{side}").mkdir()
        (tmp_path / f"m{side}" / "x.pt").write_bytes(b"mesh")
    out = tmp_path / "rows.json"
    code = cmp_.main([str(tmp_path / "a"), str(tmp_path / "b"), "--mesh-a", str(tmp_path / "ma"), "--mesh-b",
                      str(tmp_path / "mb"), "--json", str(out)])
    rows = {r["file"]: r for r in json.load(open(out))}
    assert code == 0 and rows["cut.png"]["match"] and rows["mesh.pt"]["match"] and rows["raw.glb"]["match"] is None


def test_nothing_compared_fails(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert cmp_.main([str(tmp_path / "a"), str(tmp_path / "b")]) == 1
