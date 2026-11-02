"""Post step for TRELLIS.2 GLBs: raw.glb -> asset.glb + asset.json + object.json, by a node transform only (CPU, no GPU).

  $T2_VENV/bin/python -I post/post_step.py --props PROPS.json --dir OUT [--only id,id] [--dry-run]
  (tools/ram_guard.sh can run it under a RAM floor)

raw.glb (TRELLIS.2, glTF Y up, front +Z, inside the [-0.5, 0.5] box) is never written. asset.glb = raw.glb's binary chunk
copied byte for byte (geometry, UVs, normals, PNG textures unchanged) and a new JSON chunk whose one new root node holds
the matrix below. asset.json records the rule, the numbers and the checks.
1. pose: TRELLIS.2's up is kept unless the object rests unstably on it: the centre of mass, seen from above, lies
   outside the support polygon (the convex-hull points within CONTACT_TOL of the lowest point), or inside it by a tip
   margin under MARGINAL_TIP_DEG (long thin objects that TRELLIS.2 stands on end). Centre of mass: solid (volume) if the
   mesh, vertices merged by position, is watertight; else the area-weighted surface centroid (shell).
   Unstable -> of trimesh's stable resting poses, the one whose up is closest to TRELLIS.2's up (ties: the more
   probable), reached by the smallest tilt (the yaw, and so the front, kept as far as possible). Poses with a
   probability under MIN_POSE_PROB are skipped (unless no pose reaches it).
   Per-object options in the props file:
   - "pose_rule": "simple" (+ optional "pose_rule_why"): no marginal test and no probability floor (for objects whose
     placement is not a resting pose, e.g. something that hangs).
   - "rest_up": [x, y, z] (+ optional "rest_up_why"): always take the stable pose (same floor) whose up is closest to
     this raw-glTF direction instead of TRELLIS.2's up (e.g. [0, 0, 1] lays a watch on its back, dial up).
   Yaw, only if the footprint is elongated (principal-axis length ratio >= MIN_ELONGATION; a round footprint has no
   axis): a tipped object (tilt > 45 deg) turns so its long horizontal axis lies on X; any other turns by the smallest
   angle (<= 45 deg) that puts its principal axes on X and Z, so TRELLIS.2's front stays within 45 deg. This also makes
   "longest side" mean the object's length for long objects TRELLIS.2 laid diagonally.
2. scale: uniform; the longest side of the posed, axis-aligned bounding box = the props file's "size_m".
3. origin: the bounding-box centre in X and Z, the lowest point on y = 0.
Checks after writing: the binary chunk md5 equals raw.glb's; trimesh (an independent reader that applies node
matrices) loads asset.glb and its bounds must match (longest side = size_m, min y = 0, centred).
object.json: the public metadata of the object, written after asset.glb (object_record): size, frame, pose, seed,
settings, pins, output MD5s and licences, from the props entry, asset.json and run.py's record.json. It holds no local
paths, so it can ship with asset.glb. asset.json (with the checks) and record.json are internal.
"""
import argparse, hashlib, json, math, os, resource, runpy, struct, sys, time
from functools import lru_cache
import numpy as np
import trimesh

Z_FROM_GLTF = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)  # glTF (Y up) -> ours (Z up, front -Y)
Z_TO_GLTF = np.linalg.inv(Z_FROM_GLTF)
CONTACT_TOL = 0.01  # x the longest side: hull points this close to the lowest point carry the object
MIN_ELONGATION = 1.4  # sqrt of the footprint variance ratio
MARGINAL_TIP_DEG = 2.0  # upright but this close to tipping: treated as unstable
MIN_POSE_PROB = 0.01  # trimesh stable-pose probability floor (a crutch's upright 13 deg pose had 0.0)
CHECK_TOL = 1e-4  # metres
EXTRAS_KEY = "post_step"  # asset.glb's asset.extras key (--extras-key)
RULE = ("longest side of the posed axis-aligned bounding box = size_m; pose: TRELLIS.2 up kept unless the centre of mass "
        "lies outside the support polygon or within 2 deg of tipping, else the stable resting pose (probability >= 0.01) "
        "closest to TRELLIS.2 up by the smallest tilt, long axis on X if tipped; origin = "
        "bounding-box centre in X/Z, lowest point at y = 0; node transform only, buffers byte-identical")


def pose_options(p):
    """-> (previous_rule, rest_up) for pose() from one props entry: "pose_rule": "simple" and "rest_up": [x, y, z]."""
    rule = p.get("pose_rule", "")
    if rule not in ("", "simple"):
        raise SystemExit(f"STOP: unknown pose_rule {rule!r} (only \"simple\")")
    previous_rule = p.get("pose_rule_why", "pose_rule simple (props file)") if rule == "simple" else ""
    rest_up = None
    if "rest_up" in p:
        d = p["rest_up"]
        if not (isinstance(d, list) and len(d) == 3 and all(isinstance(x, (int, float)) for x in d) and any(d)):
            raise SystemExit(f"STOP: rest_up must be a non-zero [x, y, z] list, not {d!r}")
        rest_up = (d, p.get("rest_up_why", "rest_up (props file)"))
    return previous_rule, rest_up
COMP = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}


def md5(path, start=0, length=None):
    h = hashlib.md5()
    with open(path, "rb") as f:
        f.seek(start)
        left = length if length is not None else float("inf")
        while left > 0:
            b = f.read(int(min(1 << 22, left)))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def read_glb(path):
    """-> (gltf json, file offset of the BIN chunk header, BIN chunk total size incl. its 8-byte header)."""
    with open(path, "rb") as f:
        magic, ver, total = struct.unpack("<4sII", f.read(12))
        assert magic == b"glTF" and ver == 2, (magic, ver)
        jlen, jtype = struct.unpack("<II", f.read(8))
        assert jtype == 0x4E4F534A
        j = json.loads(f.read(jlen))
        bin_at = 20 + jlen
        blen, btype = struct.unpack("<II", f.read(8))
        assert btype == 0x004E4942 and bin_at + 8 + blen == total, "one JSON chunk then one BIN chunk expected"
    return j, bin_at, 8 + blen


def accessor(f, j, bin_data_at, i):
    a = j["accessors"][i]
    bv = j["bufferViews"][a["bufferView"]]
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
    dt = np.dtype(COMP[a["componentType"]])
    stride = bv.get("byteStride") or n * dt.itemsize
    f.seek(bin_data_at + bv.get("byteOffset", 0) + a.get("byteOffset", 0))
    raw = f.read(stride * (a["count"] - 1) + n * dt.itemsize)
    return np.ndarray((a["count"], n), dt, raw, 0, (stride, dt.itemsize)).copy()


def geometry(path, j, bin_at):
    """All triangles of the default scene, glTF frame. raw.glb from TRELLIS.2: root nodes without transforms."""
    vs, fs, off = [], [], 0
    with open(path, "rb") as f:
        for ni in j["scenes"][j.get("scene", 0)]["nodes"]:
            node = j["nodes"][ni]
            assert not any(k in node for k in ("matrix", "translation", "rotation", "scale", "children")), node
            for pr in j["meshes"][node["mesh"]]["primitives"]:
                assert pr.get("mode", 4) == 4 and "indices" in pr
                v = accessor(f, j, bin_at + 8, pr["attributes"]["POSITION"]).astype(float)
                fs.append(accessor(f, j, bin_at + 8, pr["indices"]).reshape(-1, 3).astype(np.int64) + off)
                vs.append(v)
                off += len(v)
    return np.concatenate(vs), np.concatenate(fs)


def support_margin(hull_v, com):
    """Signed distance (positive = inside) from the centre of mass, seen from above, to the support polygon; tip angle."""
    zmin = hull_v[:, 2].min()
    tol = CONTACT_TOL * float(np.ptp(hull_v, axis=0).max())
    pts = hull_v[hull_v[:, 2] <= zmin + tol][:, :2]
    c = com[:2]
    try:
        from scipy.spatial import ConvexHull
        h = ConvexHull(pts)
        margin = float(-(h.equations[:, :2] @ c + h.equations[:, 2]).max())
        area = float(h.volume)
    except Exception:  # point or segment support: never stable unless the COM is on it, which it is not exactly
        margin, area = -float(np.linalg.norm(pts - c, axis=1).min()), 0.0
    height = float(com[2] - zmin)
    return dict(margin=margin, tip_deg=math.degrees(math.atan2(margin, max(height, 1e-9))), contact_points=int(len(pts)),
                support_area=area, contact_tol=tol)


def footprint_axis_deg(v):
    xy = v[:, :2] - v[:, :2].mean(0)
    w, e = np.linalg.eigh(xy.T @ xy)
    ang = math.degrees(math.atan2(e[1, -1], e[0, -1]))
    ang = (ang + 90) % 180 - 90  # an axis, not a direction: (-90, 90]
    return ang, float(math.sqrt(w[-1] / max(w[0], 1e-18)))


def pose(V, F, previous_rule="", rest_up=None):
    """V, F in our Z-up frame. -> (4x4 rotation in our frame, record)."""
    g = trimesh.Trimesh(V, F, process=False)
    g.merge_vertices()
    labels = trimesh.graph.connected_component_labels(g.face_adjacency, node_count=len(g.faces))
    counts = np.bincount(labels)
    big = labels == int(np.argmax(counts))
    watertight = bool(g.is_watertight)
    if watertight:
        com, com_kind = np.asarray(g.center_mass, float), "solid (watertight, vertices merged by position)"
    else:
        a = g.area_faces
        com, com_kind = (g.triangles_center * a[:, None]).sum(0) / a.sum(), "surface area-weighted (not watertight: shell)"
    hull = g.convex_hull
    s0 = support_margin(hull.vertices, com)
    axis0, elong0 = footprint_axis_deg(g.vertices)
    rec = dict(watertight=watertight, centre_of_mass=com_kind, pieces=int(len(counts)),
               largest_piece_face_share=round(float(big.sum() / len(big)), 5),
               trellis_up=dict(com_inside_support=s0["margin"] > 0, tip_margin_deg=round(s0["tip_deg"], 2),
                               contact_points=s0["contact_points"], support_area_rel=round(s0["support_area"] / float(np.ptp(hull.vertices, axis=0)[:2].prod()), 4),
                               footprint_axis_deg=round(axis0, 2), footprint_elongation=round(elong0, 3)))
    marginal_tip = 0.0 if previous_rule else MARGINAL_TIP_DEG
    min_prob = 0.0 if previous_rule else MIN_POSE_PROB
    if previous_rule:
        rec["previous_rule"] = previous_rule
    target = np.array([0, 0, 1.0]) if rest_up is None else (Z_FROM_GLTF[:3, :3] @ np.array(rest_up[0], float))
    if rest_up is not None:
        rec["rest_up"] = dict(direction_gltf_raw=rest_up[0], why=rest_up[1])
    if rest_up is None and s0["margin"] > 0 and s0["tip_deg"] >= marginal_tip:
        R = np.eye(4)
        rec.update(chosen="trellis-up", tilt_deg=0.0,
                   why=f"centre of mass inside the support polygon on TRELLIS.2's up (tip margin {s0['tip_deg']:.1f} deg)"
                   + ("; MARGINAL: look at it" if s0["tip_deg"] < MARGINAL_TIP_DEG else ""))
        rec["marginal"] = bool(s0["tip_deg"] < MARGINAL_TIP_DEG)
    else:
        T, probs = g.compute_stable_poses(center_mass=com, sigma=0.0, n_samples=1)
        ups = [np.linalg.inv(t)[:3, :3] @ np.array([0, 0, 1.0]) for t in T]
        cand = [i for i in range(len(T)) if probs[i] >= min_prob] or list(range(len(T)))
        pick = max(cand, key=lambda i: (round(float(ups[i] @ target), 3), float(probs[i])))
        u = ups[pick] / np.linalg.norm(ups[pick])
        Rt = trimesh.geometry.align_vectors(u, [0, 0, 1.0])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(u[2])))))
        hv = trimesh.transform_points(hull.vertices, Rt)
        s1 = support_margin(hv, trimesh.transform_points([com], Rt)[0])
        R = Rt
        rec.update(chosen="stable-pose", tilt_deg=round(tilt, 2), stable_poses=len(T),
                   stable_pose_probability=round(float(probs[pick]), 4), stable_pose_up_dot_trellis_up=round(float(u[2]), 4),
                   after=dict(tip_margin_deg=round(s1["tip_deg"], 2), contact_points=s1["contact_points"]),
                   stable_poses_skipped_low_probability=len(T) - len(cand) if len(cand) < len(T) else 0,
                   why=(f"{rest_up[1]}: the stable pose (probability >= {min_prob}) whose up is closest to raw glTF {rest_up[0]}; "
                        f"tilt {tilt:.1f} deg" if rest_up is not None else
                        (f"centre of mass outside the support polygon on TRELLIS.2's up (by {-s0['margin']:.4f} raw units, "
                         if s0["margin"] <= 0 else f"centre of mass {s0['tip_deg']:.1f} deg from tipping on TRELLIS.2's up (< {marginal_tip}; ")
                        + f"{s0['contact_points']} contact points); nearest stable pose (probability >= {min_prob}) tilts {tilt:.1f} deg"))
        rec["marginal"] = False
    ang, el = footprint_axis_deg(trimesh.transform_points(g.vertices, R))
    yaw, how = 0.0, f"none: footprint elongation {el:.2f} < {MIN_ELONGATION} (no clear axis); TRELLIS.2's yaw kept"
    if el >= MIN_ELONGATION and rec["tilt_deg"] > 45:
        yaw, how = -ang, f"tipped: long horizontal axis on X (elongation {el:.2f})"
    elif el >= MIN_ELONGATION:
        yaw, how = -(((ang + 45) % 90) - 45), f"principal axes snapped to X/Z by the smallest turn (elongation {el:.2f}); front within 45 deg"
    R = trimesh.transformations.rotation_matrix(math.radians(yaw), [0, 0, 1]) @ R
    rec.update(yaw_deg=round(yaw, 2), yaw_rule=how)
    return R, hull, rec


def write_glb(src, dst, j, bin_at, bin_size):
    js = json.dumps(j, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    tmp = dst + ".part"
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + bin_size))
        fo.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fi.seek(bin_at)
        left = bin_size
        while left:
            b = fi.read(min(1 << 22, left))
            fo.write(b)
            left -= len(b)
    os.replace(tmp, dst)


def process(pid, p, od, dry, extras_key=EXTRAS_KEY):
    t0 = time.time()
    src, dst = os.path.join(od, "raw.glb"), os.path.join(od, "asset.glb")
    raw_md5 = md5(src)
    j, bin_at, bin_size = read_glb(src)
    Vg, F = geometry(src, j, bin_at)
    V = trimesh.transform_points(Vg, Z_FROM_GLTF)
    R, hull, prec = pose(V, F, *pose_options(p))
    hv = trimesh.transform_points(hull.vertices, R)
    ext = np.ptp(hv, axis=0)
    sc = p["size_m"] / float(ext.max())
    lo, hi = hv.min(0) * sc, hv.max(0) * sc
    Tr = trimesh.transformations.translation_matrix([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    Mz = Tr @ trimesh.transformations.scale_matrix(sc) @ R
    M = Z_TO_GLTF @ Mz @ Z_FROM_GLTF  # acts on raw.glb's glTF coordinates
    vb = trimesh.transform_points(hull.vertices, Z_TO_GLTF @ Mz)
    bbox = np.ptp(vb, axis=0)
    rec = dict(id=pid, script="post/post_step.py", rule=RULE, size_m=p["size_m"], scale=round(sc, 8),
               bbox_m_gltf_xyz=[round(float(x), 4) for x in bbox], height_m=round(float(bbox[1]), 4),
               pose=prec, matrix_gltf_column_major=[round(float(x), 9) for x in M.T.ravel()],
               frame="glTF: Y up, front +Z (TRELLIS.2's front kept unless tipped); origin = base centre",
               input=dict(file="raw.glb", md5=raw_md5, bytes=os.path.getsize(src), faces=int(len(F)), vertices=int(len(Vg))))
    if not dry:
        j2 = json.loads(json.dumps(j))
        sc_i = j2.get("scene", 0)
        assert len(j2["scenes"]) == 1
        j2["nodes"].append({"name": f"{pid}-asset", "matrix": rec["matrix_gltf_column_major"], "children": j2["scenes"][sc_i]["nodes"]})
        j2["scenes"][sc_i]["nodes"] = [len(j2["nodes"]) - 1]
        j2["asset"]["extras"] = {extras_key: f"posed and scaled by a root node matrix (scale {sc:.6f}, pose {prec['chosen']}); see asset.json"}
        write_glb(src, dst, j2, bin_at, bin_size)
        # checks: buffers byte-identical, raw.glb unchanged, trimesh's bounds of asset.glb (node matrix applied)
        _, bin_at2, bin_size2 = read_glb(dst)
        bin_same = bin_size2 == bin_size and md5(dst, bin_at2 + 8) == md5(src, bin_at + 8, bin_size - 8)
        scene = trimesh.load(dst, force="scene")
        b = np.asarray(scene.bounds)
        e = b[1] - b[0]
        chk = dict(bin_chunk_byte_identical=bool(bin_same), raw_unchanged=md5(src) == raw_md5,
                   trimesh_bounds_m=[[round(float(x), 5) for x in r] for r in b],
                   longest_side_ok=abs(float(e.max()) - p["size_m"]) < CHECK_TOL, on_ground_ok=abs(float(b[0][1])) < CHECK_TOL,
                   centred_ok=abs(float(b[0][0] + b[1][0])) < 2 * CHECK_TOL and abs(float(b[0][2] + b[1][2])) < 2 * CHECK_TOL)
        chk["passed"] = all(v for k, v in chk.items() if k != "trimesh_bounds_m")
        rec.update(output=dict(file="asset.glb", md5=md5(dst), bytes=os.path.getsize(dst)), checks=chk)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    if not dry:
        with open(os.path.join(od, "asset.json"), "w") as f:
            json.dump(rec, f, indent=1)
        write_object_json(pid, p, od, rec)
    pr = prec
    print(json.dumps(dict(id=pid, chosen=pr["chosen"], tilt=pr["tilt_deg"], yaw=pr["yaw_deg"], tip0=pr["trellis_up"]["tip_margin_deg"],
                          marginal=pr["marginal"], wt=pr["watertight"], pieces=pr["pieces"], share=pr["largest_piece_face_share"],
                          axis0=pr["trellis_up"]["footprint_axis_deg"], elong0=pr["trellis_up"]["footprint_elongation"],
                          scale=rec["scale"], bbox=rec["bbox_m_gltf_xyz"], passed=rec.get("checks", {}).get("passed"),
                          s=rec["seconds"], rss_mb=rec["peak_rss_mb"])), flush=True)


OBJECT_FORMAT = "conjure3d object.json 1"
SIZE_RULE = "longest side of the posed, axis-aligned bounding box"
FRAME = "glTF: Y up, front +Z, origin = base centre"
PROPS_TEXT = ("noun", "description", "material")   # optional props fields copied as given
LICENCES = {
    "TRELLIS.2 code and TRELLIS.2-4B weights": "MIT",
    "TRELLIS-image-large sparse-structure decoder": "MIT",
    "DINOv3 ViT-L/16 image features": "DINOv3 License (conditions: THIRD_PARTY_NOTICES.md)",
    "BiRefNet cut-out code and weights": "MIT",
    "CuMesh (built without cubvh), FlexGEMM, o-voxel, utils3d": "MIT",
    "uvrast (texture bake), trianglebvh (BVH)": "MIT",
}


@lru_cache(maxsize=1)
def run_pins():
    """run.py's pins (it is not executed: only its constants are read)."""
    return runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trellis2", "run.py"),
                          run_name="post_step")


def object_record(pid, p, asset, run_rec, pins):
    """object.json for one object: public fields only, no local paths. asset = asset.json's record; run_rec =
    run.py's record.json (None if absent). The weights' SHA-256 are run.py's pins, listed only when the record names
    the pinned TRELLIS.2-4B revision (run.py stops at start unless every pinned file has its SHA-256)."""
    r = run_rec or {}
    pose = asset["pose"]
    source = ("rest_up (props file)" if "rest_up" in pose else "pose_rule simple (props file)" if "previous_rule" in pose
              else "default")
    weights = None
    if r.get("model") == f"microsoft/TRELLIS.2-4B@{pins['HF_REV']}":
        weights = {rel: v[3][:12] for rel, v in sorted(pins["WEIGHT_FILES"].items()) if rel.endswith(".safetensors")}
        weights[f"BiRefNet@{pins['BIREFNET_REV'][:8]}/model.safetensors"] = pins["BIREFNET_SHA256"][:12]
    pick = lambda d, *keys: {k: d[k] for k in keys if k in d} if isinstance(d, dict) else None
    export = r.get("export") or {}
    obj = dict(format=OBJECT_FORMAT, id=pid, **{k: p[k] for k in PROPS_TEXT if k in p},
               size_m=asset["size_m"], size_rule=SIZE_RULE, units="metres",
               bbox_m=asset["bbox_m_gltf_xyz"], bbox_axes="glTF x, y, z", height_m=asset["height_m"], frame=FRAME,
               pose=dict(rule=pose["chosen"], rule_source=source, tilt_deg=pose["tilt_deg"], yaw_deg=pose["yaw_deg"]),
               seed=r.get("seed"), source_image_md5=r.get("gold_md5"), gpu_model=r.get("gpu"),
               settings=dict(profile=(r.get("profile") or {}).get("name"), pipeline_type=r.get("pipeline_type"),
                             texture_size=export.get("texture_size"), face_target=export.get("decimation_target"),
                             export_note=r.get("export_note")),
               pins=dict(trellis2_code=r.get("code"), trellis2_weights=r.get("model"), weights_sha256_12=weights,
                         birefnet=pick(r.get("cutout"), "code", "weights"),
                         cumesh=pick(r.get("cumesh"), "commit", "patch_sha256", "cubvh", "no_cubvh_patch_sha256"),
                         trianglebvh=pick(r.get("bvh"), "version", "commit", "library_sha256"),
                         uvrast=pick(r.get("uvrast"), "version", "commit"),
                         remesh=pick(r.get("remesh"), "mode", "sha256")),
               outputs_md5={"asset.glb": asset["output"]["md5"], "raw.glb": asset["input"]["md5"]},
               licences=LICENCES)
    return obj


def write_object_json(pid, p, od, asset):
    rec_path = os.path.join(od, "record.json")
    run_rec = json.load(open(rec_path)) if os.path.isfile(rec_path) else None
    with open(os.path.join(od, "object.json"), "w") as f:
        json.dump(object_record(pid, p, asset, run_rec, run_pins()), f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--props", required=True)
    ap.add_argument("--dir", required=True, help="<dir>/<id>/raw.glb")
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true", help="print the ids that have a raw.glb, one per line")
    ap.add_argument("--dry-run", action="store_true", help="compute and print; write nothing")
    ap.add_argument("--extras-key", default=EXTRAS_KEY, help="key of the note in asset.glb's asset.extras")
    a = ap.parse_args()
    props = json.load(open(a.props))["props"]
    ids = [k for k in props if os.path.exists(os.path.join(a.dir, k, "raw.glb"))]
    if a.list:
        print("\n".join(ids))
        return
    for pid in ids:
        if a.only and pid not in a.only.split(","):
            continue
        process(pid, props[pid], os.path.join(a.dir, pid), a.dry_run, a.extras_key)


if __name__ == "__main__":
    sys.exit(main())
