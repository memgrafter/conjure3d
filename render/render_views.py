"""Blender: render TRELLIS.2 GLBs (run.py raw.glb: glTF Y up, object inside the [-0.5, 0.5] box) from fixed views
with the GLB's own PBR material, for review sheets and render differences between runs and seeds.
The camera does not depend on the mesh (target = box centre, fixed distance and lens), so the same view of two GLBs
compares pixel by pixel. EEVEE, one sun + uniform world light.
  blender -b --factory-startup --python render/render_views.py -- OUT_DIR GLB [OUT_DIR GLB ...]   -> OUT_DIR/v0.png ... v8.png
"""
import math, os, sys
import bpy, mathutils

VIEWS = [(az, 20) for az in range(0, 360, 45)] + [(30, 70)]  # 8 around at 20 degrees up + 1 from high above
DIST, LENS, RES = 2.3, 50.0, 768
args = sys.argv[sys.argv.index("--") + 1:]
for out, glb in zip(args[0::2], args[1::2]):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)
    sc = bpy.context.scene
    engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    sc.render.resolution_x = sc.render.resolution_y = RES
    sc.view_settings.view_transform = "Standard"
    w = bpy.data.worlds.new("w"); w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.8
    sc.world = w
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.lens = LENS
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sun.data.energy = 3; sc.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    os.makedirs(out, exist_ok=True)
    for i, (az, el) in enumerate(VIEWS):
        a, e = math.radians(az), math.radians(el)
        cam.location = (DIST * math.cos(e) * math.sin(a), -DIST * math.cos(e) * math.cos(a), DIST * math.sin(e))  # az 0 = front (-Y)
        cam.rotation_euler = (mathutils.Vector((0, 0, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = os.path.join(out, f"v{i}.png")
        bpy.ops.render.render(write_still=True)
    print(f"RENDERED {glb} -> {out}", flush=True)
