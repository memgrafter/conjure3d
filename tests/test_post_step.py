"""Post step maths on tiny meshes: raw.glb -> asset.glb (pose, scale, origin by a root node matrix, buffers unchanged)."""
import json, math, struct
import pytest

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")
from conftest import load

post = load("post/post_step.py", "post_step")


def write_raw_glb(path, vertices, faces):
    """A glTF 2 binary file like TRELLIS.2's raw.glb: one root node with one mesh, no transform, one BIN chunk."""
    v = np.asarray(vertices, np.float32)
    f = np.asarray(faces, np.uint32).reshape(-1)
    vb, fb = v.tobytes(), f.tobytes()
    gltf = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(v), "type": "VEC3",
             "min": v.min(0).tolist(), "max": v.max(0).tolist()},
            {"bufferView": 1, "componentType": 5125, "count": len(f), "type": "SCALAR"}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(vb)},
                        {"buffer": 0, "byteOffset": len(vb), "byteLength": len(fb)}],
        "buffers": [{"byteLength": len(vb) + len(fb)}],
    }
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    bin_ = vb + fb
    bin_ += b"\0" * (-len(bin_) % 4)
    with open(path, "wb") as out:
        out.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(bin_)))
        out.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        out.write(struct.pack("<II", len(bin_), 0x004E4942) + bin_)


def box_in_gltf(extents, rotate_x_deg=0.0):
    """A closed box mesh in raw glTF coordinates (Y up), optionally rotated about X, inside the [-0.5, 0.5] box."""
    m = trimesh.creation.box(extents=extents)
    if rotate_x_deg:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(rotate_x_deg), [1, 0, 0]))
    return m.vertices, m.faces


def run_post(tmp_path, vertices, faces, prop, extras_key=post.EXTRAS_KEY):
    od = tmp_path / "obj"
    od.mkdir()
    write_raw_glb(od / "raw.glb", vertices, faces)
    post.process("obj", prop, str(od), False, extras_key)
    return od, json.load(open(od / "asset.json"))


def test_upright_box_keeps_its_pose_and_gets_real_size(tmp_path):
    od, rec = run_post(tmp_path, *box_in_gltf([0.2, 0.6, 0.3]), {"size_m": 1.5})
    assert rec["pose"]["chosen"] == "trellis-up" and rec["pose"]["tilt_deg"] == 0.0
    assert rec["checks"]["passed"] and rec["checks"]["bin_chunk_byte_identical"] and rec["checks"]["raw_unchanged"]
    assert rec["height_m"] == pytest.approx(1.5, abs=1e-4)            # the longest side (glTF Y) is the height
    b = trimesh.load(od / "asset.glb", force="scene").bounds
    assert b[0][1] == pytest.approx(0, abs=1e-6)                     # on the ground
    assert (b[0][0] + b[1][0]) == pytest.approx(0, abs=1e-6) and (b[0][2] + b[1][2]) == pytest.approx(0, abs=1e-6)


def test_box_on_an_edge_is_laid_down(tmp_path):
    # rotated 30 degrees about X, a box stands on one edge: no support area, so the post step picks a stable pose
    od, rec = run_post(tmp_path, *box_in_gltf([0.6, 0.2, 0.4], rotate_x_deg=30), {"size_m": 0.9})
    assert rec["pose"]["chosen"] == "stable-pose" and rec["pose"]["tilt_deg"] > 1
    assert rec["checks"]["passed"]
    b = trimesh.load(od / "asset.glb", force="scene").bounds
    assert max(b[1] - b[0]) == pytest.approx(0.9, abs=1e-4) and b[0][1] == pytest.approx(0, abs=1e-6)


def test_rest_up_turns_the_named_direction_up(tmp_path):
    # raw glTF +Z (TRELLIS.2's front) should point up: a flat face of the box faces +Z
    od, rec = run_post(tmp_path, *box_in_gltf([0.5, 0.5, 0.1]), {"size_m": 0.05, "rest_up": [0, 0, 1], "rest_up_why": "test"})
    assert rec["pose"]["chosen"] == "stable-pose" and rec["pose"]["rest_up"]["direction_gltf_raw"] == [0, 0, 1]
    assert rec["checks"]["passed"]
    b = trimesh.load(od / "asset.glb", force="scene").bounds
    assert (b[1] - b[0])[1] == pytest.approx(0.01, abs=1e-4)          # the thin side now points up


def test_extras_key_and_node(tmp_path):
    od, _ = run_post(tmp_path, *box_in_gltf([0.2, 0.3, 0.2]), {"size_m": 0.3}, extras_key="my_key")
    j, _, _ = post.read_glb(str(od / "asset.glb"))
    assert list(j["asset"]["extras"]) == ["my_key"]
    assert j["scenes"][0]["nodes"] == [1] and j["nodes"][1]["name"] == "obj-asset" and len(j["nodes"][1]["matrix"]) == 16


@pytest.mark.parametrize("prop", [{"size_m": 1, "pose_rule": "fancy"}, {"size_m": 1, "rest_up": [0, 0, 0]},
                                  {"size_m": 1, "rest_up": "up"}])
def test_bad_pose_options_stop(prop):
    with pytest.raises(SystemExit):
        post.pose_options(prop)


def test_pose_options():
    assert post.pose_options({}) == ("", None)
    assert post.pose_options({"pose_rule": "simple"})[0]
    assert post.pose_options({"rest_up": [0, 0, 1], "rest_up_why": "w"}) == ("", ([0, 0, 1], "w"))


LOCALROOT = "/machine-root/user"   # stands in for a local folder; must never reach object.json


def fake_record(pins):
    """The record.json fields object.json reads, as run.py writes them (local paths included, like a real record)."""
    return dict(model=f"microsoft/TRELLIS.2-4B@{pins['HF_REV']}", code=f"microsoft/TRELLIS.2@{pins['T2_SHA']}", seed=7,
                gold_md5="0" * 32, gpu="NVIDIA GeForce RTX 3090", pipeline_type="512", export_note="as configured",
                export=dict(decimation_target=100000, texture_size=1024), profile=dict(name="draft"),
                cutout=dict(code="github.com/ZhengPeng7/BiRefNet@x", weights="ZhengPeng7/BiRefNet@y/model.safetensors",
                            code_path=f"{LOCALROOT}/src/BiRefNet"),
                cumesh=dict(commit=pins["CUMESH_SHA"], patch_sha256="p", cubvh="not built", no_cubvh_patch_sha256="n",
                            source=f"{LOCALROOT}/src/CuMesh", so_sha256={"_C.so": "s"}),
                bvh=dict(version="0.1.0", commit=pins["TRIANGLEBVH_SHA"], library_sha256="l", url=f"file://{LOCALROOT}/tb",
                         path=f"{LOCALROOT}/venv/tb", module=f"{LOCALROOT}/venv/tb/__init__.py"),
                uvrast=dict(version="0.1.0", commit="u", url="https://github.com/memgrafter/uvrast", path=f"{LOCALROOT}/venv/uv"),
                remesh=dict(mode="lean", sha256="r", file=f"{LOCALROOT}/repo/trellis2/cumesh_lean_remesh.py"),
                venv=f"{LOCALROOT}/venv", load=dict(weights=dict(dir=f"{LOCALROOT}/weights")))


def test_object_json_is_public_metadata(tmp_path):
    od = tmp_path / "obj"
    od.mkdir()
    write_raw_glb(od / "raw.glb", *box_in_gltf([0.2, 0.6, 0.3]))
    pins = post.run_pins()
    json.dump(fake_record(pins), open(od / "record.json", "w"))
    prop = {"size_m": 1.5, "gold": f"{LOCALROOT}/gold/box.png", "noun": "box", "material": "oak"}
    post.process("obj", prop, str(od), False)
    obj, asset = json.load(open(od / "object.json")), json.load(open(od / "asset.json"))
    assert obj["format"] == post.OBJECT_FORMAT and obj["id"] == "obj" and (obj["noun"], obj["material"]) == ("box", "oak")
    assert obj["size_m"] == 1.5 and obj["height_m"] == pytest.approx(1.5, abs=1e-4) and obj["bbox_m"] == asset["bbox_m_gltf_xyz"]
    assert obj["frame"] == "glTF: Y up, front +Z, origin = base centre" and obj["pose"]["rule"] == "trellis-up"
    assert obj["seed"] == 7 and obj["settings"] == dict(profile="draft", pipeline_type="512", texture_size=1024,
                                                        face_target=100000, export_note="as configured")
    assert obj["pins"]["trianglebvh"] == dict(version="0.1.0", commit=pins["TRIANGLEBVH_SHA"], library_sha256="l")
    assert obj["pins"]["cumesh"]["cubvh"] == "not built" and len(obj["pins"]["weights_sha256_12"]) == 1 + sum(k.endswith(".safetensors") for k in pins["WEIGHT_FILES"])
    assert obj["outputs_md5"] == {"asset.glb": post.md5(str(od / "asset.glb")), "raw.glb": post.md5(str(od / "raw.glb"))}
    text = (od / "object.json").read_text()
    assert LOCALROOT not in text and "file://" not in text and str(tmp_path) not in text


def test_object_json_does_not_change_asset_glb(tmp_path):
    md5s = []
    for name, with_record in (("a", False), ("b", True)):
        od = tmp_path / name / "obj"
        od.mkdir(parents=True)
        write_raw_glb(od / "raw.glb", *box_in_gltf([0.3, 0.2, 0.25]))
        if with_record:
            json.dump(fake_record(post.run_pins()), open(od / "record.json", "w"))
        post.process("obj", {"size_m": 0.4}, str(od), False)
        assert (od / "object.json").is_file()
        md5s.append(post.md5(str(od / "asset.glb")))
    assert md5s[0] == md5s[1]
    obj = json.load(open(tmp_path / "a" / "obj" / "object.json"))
    assert obj["seed"] is None and obj["pins"]["weights_sha256_12"] is None    # no record.json: unknown, not guessed
