"""Blender: bake the master's textures onto a low-poly copy with fresh UVs (the web export's proxy level).
Below the seam floor (15k-90k triangles on TRELLIS meshes) a simplifier must collapse UV seams, and the old UVs then
point into the wrong charts (the speckles). So the low mesh gets new UVs (Smart UV Project) and both textures are
re-baked from the master by ray casting (Cycles, selected-to-active, EMIT pass of an emission copy of each texture,
so no lighting enters): base colour (sRGB) and metallic-roughness (non-colour, glTF G = roughness, B = metallic).
Rays start outside the low surface (cage extrusion, 4% of the diagonal; 2% let footstool's hidden green inner shell
through on one test object) and stop at the first master surface.
MR_MODE "texture" exports the baked metallic-roughness texture; "factors" exports only its mean over the baked texels
(no texture; for the proxy). Target images start fully transparent; baked and margin texels get alpha 1.
  blender -b --factory-startup --python web/bake_proxy.py -- MASTER_GLB LOW_GLB OUT_GLB BAKE_PX MR_MODE [ANGLE_DEG]
"""
import math, sys, time
import bpy

args = sys.argv[sys.argv.index("--") + 1:]
MASTER, LOW, OUT, PX, MR_MODE = args[0], args[1], args[2], int(args[3]), args[4]
assert MR_MODE in ("texture", "factors")
ANGLE_LIMIT, ISLAND_MARGIN = math.radians(float(args[5]) if len(args) > 5 else 66), 0.02  # degrees; larger = fewer UV islands, fewer split vertices
import os
EXTRUSION = float(os.environ.get("BAKE_EXTRUSION", "0.04"))  # fractions of the master's bounding-box diagonal
MAX_RAY = 4 * EXTRUSION
SAMPLES, MARGIN_PX = 4, 16
t0 = time.time()

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene


def import_meshes(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.data.objects if o not in before and o.type == "MESH"]


high = import_meshes(MASTER)
low = import_meshes(LOW)
assert len(low) == 1, f"expected one low mesh, got {len(low)}"
low = low[0]

pts = [o.matrix_world @ v.co for o in high for v in o.data.vertices]
lo = [min(p[i] for p in pts) for i in range(3)]
hi = [max(p[i] for p in pts) for i in range(3)]
diag = math.dist(lo, hi)


def find_image(mat, colorspace):
    for n in mat.node_tree.nodes:
        if n.type == "TEX_IMAGE" and n.image and n.image.colorspace_settings.name == colorspace:
            return n.image
    return None


def emission_material(name, image):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = image
    em = nt.nodes.new("ShaderNodeEmission"); nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    out = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(em.outputs[0], out.inputs[0])
    return m


src_mat = high[0].active_material
base_img, mr_img = find_image(src_mat, "sRGB"), find_image(src_mat, "Non-Color")
assert base_img is not None, "master has no sRGB base colour image"
emit = {"base": emission_material("emit_base", base_img)}
if mr_img is not None:
    emit["mr"] = emission_material("emit_mr", mr_img)

# fresh UVs on the low mesh
for uv in list(low.data.uv_layers):
    low.data.uv_layers.remove(uv)
low.data.uv_layers.new(name="UVMap")
bpy.ops.object.select_all(action="DESELECT")
bpy.context.view_layer.objects.active = low
low.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=ANGLE_LIMIT, island_margin=ISLAND_MARGIN, area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
bpy.ops.object.mode_set(mode="OBJECT")

targets = {k: bpy.data.images.new(f"bake_{k}", PX, PX, alpha=True) for k in emit}
for img in targets.values():
    img.generated_color = (0, 0, 0, 0)
targets["base"].colorspace_settings.name = "sRGB"
if "mr" in targets:
    targets["mr"].colorspace_settings.name = "Non-Color"

# the low mesh's final material: base colour + (G roughness, B metallic) from the baked images
mat = bpy.data.materials.new("low"); mat.use_nodes = True
mat.use_backface_culling = True  # exported as single-sided, like the master
nt = mat.node_tree
bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
tex_base = nt.nodes.new("ShaderNodeTexImage"); tex_base.image = targets["base"]
nt.links.new(tex_base.outputs["Color"], bsdf.inputs["Base Color"])
if "mr" in targets:
    tex_mr = nt.nodes.new("ShaderNodeTexImage"); tex_mr.image = targets["mr"]
low.data.materials.clear()
low.data.materials.append(mat)

sc.render.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.samples = SAMPLES
bk = sc.render.bake
bk.use_selected_to_active = True
bk.cage_extrusion = EXTRUSION * diag
bk.max_ray_distance = MAX_RAY * diag
bk.margin = MARGIN_PX
bk.margin_type = "EXTEND"
bk.use_clear = True
bpy.ops.object.select_all(action="DESELECT")
for o in high:
    o.select_set(True)
low.select_set(True)
bpy.context.view_layer.objects.active = low
for key, m in emit.items():
    for o in high:
        o.data.materials.clear(); o.data.materials.append(m)
    node = tex_base if key == "base" else tex_mr
    nt.nodes.active = node
    bpy.ops.object.bake(type="EMIT")
    targets[key].pack()
    print(f"BAKED {key} {PX}px", flush=True)

import numpy as np
px = np.empty(PX * PX * 4, np.float32)
targets["base"].pixels.foreach_get(px)
print(f"COVERAGE {float((px[3::4] > 0).mean()):.3f} of base texels baked or margin", flush=True)
if "mr" in targets and MR_MODE == "texture":
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(tex_mr.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
elif "mr" in targets:
    targets["mr"].pixels.foreach_get(px)
    a = px.reshape(-1, 4)
    a = a[a[:, 3] > 0]
    bsdf.inputs["Roughness"].default_value = float(a[:, 1].mean())
    bsdf.inputs["Metallic"].default_value = float(a[:, 2].mean())
    nt.nodes.remove(tex_mr)
    print(f"MR factors roughness {a[:, 1].mean():.3f} metallic {a[:, 2].mean():.3f}", flush=True)

bpy.ops.object.select_all(action="DESELECT")
low.select_set(True)
bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", use_selection=True, export_image_format="AUTO",
                          export_materials="EXPORT", export_texcoords=True, export_normals=True, export_yup=True)
print(f"PROXY {OUT} faces {len(low.data.polygons)} diag {diag:.3f} {time.time() - t0:.1f}s", flush=True)
