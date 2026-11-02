"""Blender: turntable frames for TRELLIS.2 GLBs (run.py raw.glb: glTF Y up, inside the [-0.5, 0.5] box), for the
quick-succession review video (turntable_video.sh). The object turns once around the vertical axis in SECONDS, seen
from a camera 20 degrees up, fitted to the object's bounding sphere; same key light and grey world light as render_views.py, plus a rim light from behind; the film is
transparent (RGBA frames) and turntable_video.sh adds the background.
The object's name is drawn at the bottom of the frame (flat white text on a dark band, unlit), so ffmpeg needs no
text filter (Homebrew's ffmpeg has no drawtext).
  blender -b --factory-startup --python render/turntable.py -- FPS SECONDS RES|WxH OUT_DIR LABEL GLB [OUT_DIR LABEL GLB ...]
  -> OUT_DIR/f0000.png ...
"""
import math, os, sys
import bpy, mathutils

args = sys.argv[sys.argv.index("--") + 1:]
fps, seconds = int(args[0]), float(args[1])
res_w, res_h = (int(v) for v in args[2].split("x")) if "x" in args[2] else (int(args[2]),) * 2  # RES or WxH
frames = round(fps * seconds)
LENS, ELEV, MARGIN, LABEL_LIFT, RIM = 50.0, 20, 1.08, 0.12, 4.0


def flat(name, rgba):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission"); em.inputs[0].default_value = rgba
    out_node = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(em.outputs[0], out_node.inputs[0])
    return m


rest = args[3:]
for out, label, glb in zip(rest[0::3], rest[1::3], rest[2::3]):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)
    sc = bpy.context.scene
    # centre the object and fit it: the camera distance comes from the bounding sphere, so the whole object stays
    # in frame for every angle of the turn (a tall object, e.g. a book standing up, no longer runs off the top)
    pts = [ob.matrix_world @ mathutils.Vector(c) for ob in sc.objects if ob.type == "MESH" for c in ob.bound_box]
    lo = mathutils.Vector([min(p[i] for p in pts) for i in range(3)])
    hi = mathutils.Vector([max(p[i] for p in pts) for i in range(3)])
    centre, radius = (lo + hi) / 2, max(((hi - lo) / 2).length, 1e-3)
    pivot = bpy.data.objects.new("pivot", None); sc.collection.objects.link(pivot)
    for ob in [o for o in sc.objects if o.parent is None and o is not pivot]:
        ob.location -= centre
        ob.parent = pivot
    engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    sc.render.resolution_x, sc.render.resolution_y = res_w, res_h
    sc.view_settings.view_transform = "Standard"
    w = bpy.data.worlds.new("w"); w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.8
    sc.world = w
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.lens = LENS
    cam.data.sensor_fit = "VERTICAL"; cam.data.sensor_height = 36  # the fit below uses the vertical field of view
    e = math.radians(ELEV)
    half_fov = math.atan(18 / LENS)  # 36 mm sensor, square frame
    dist = radius / math.sin(half_fov) * MARGIN
    aim = mathutils.Vector((0, 0, -LABEL_LIFT * radius))  # aim a little low so the object sits above the label band
    cam.location = (0, -dist * math.cos(e), dist * math.sin(e))  # front (-Y), 20 degrees up
    cam.rotation_euler = (aim - cam.location).to_track_quat("-Z", "Y").to_euler()
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sun.data.energy = 3; sc.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    # rim light from behind and above: a bright edge that separates the silhouette from any background
    rim = bpy.data.objects.new("rim", bpy.data.lights.new("rim", "SUN")); rim.data.energy = RIM; sc.collection.objects.link(rim)
    rim.rotation_euler = (mathutils.Vector((0, 0, 0)) - mathutils.Vector((0.4, 3.0, 2.2))).to_track_quat("-Z", "Y").to_euler()
    # transparent film: turntable_video.sh puts the frames over its own background, so the world only lights the object
    sc.render.film_transparent = True
    sc.render.image_settings.color_mode = "RGBA"
    # label: 1 m in front of the camera, near the bottom edge (half-height there = tan(fov/2) ~ 0.36 m at lens 50)
    txt = bpy.data.curves.new("label", "FONT"); txt.body = label; txt.size = 0.042
    txt.align_x = "CENTER"; txt.align_y = "CENTER"
    tob = bpy.data.objects.new("label", txt); tob.parent = cam; tob.location = (0, -0.315, -1.0)
    tob.data.materials.append(flat("label", (1, 1, 1, 1))); sc.collection.objects.link(tob)
    bpy.ops.mesh.primitive_plane_add(size=1); band = bpy.context.active_object
    band.parent = cam; band.location = (0, -0.315, -1.001); band.scale = (0.74, 0.07, 1)
    band.data.materials.append(flat("band", (0.03, 0.03, 0.03, 1)))
    os.makedirs(out, exist_ok=True)
    for i in range(frames):
        pivot.rotation_euler = (0, 0, 2 * math.pi * i / frames)
        sc.render.filepath = os.path.join(out, f"f{i:04d}.png")
        bpy.ops.render.render(write_still=True)
    print(f"TURNTABLE {glb} -> {out} ({frames} frames)", flush=True)
