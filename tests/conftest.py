import importlib.util, json, os, runpy, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(rel, name):
    """Import a script of this repo by path (the scripts are not a package)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_weights(weights, with_512_texture):
    """Empty files at every pinned weight path, and a pipeline.json with TRELLIS.2's model names. Without the 512
    texture model: a weights folder installed before that model was pinned."""
    pins = runpy.run_path(os.path.join(ROOT, "trellis2", "run.py"), run_name="tests")
    t2 = os.path.join(weights, pins["T2_DIR"])
    models = {"sparse_structure_decoder": "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
              "sparse_structure_flow_model": "ckpts/ss_flow_img_dit_1_3B_64_bf16",
              "shape_slat_decoder": "ckpts/shape_dec_next_dc_f16c32_fp16",
              "shape_slat_flow_model_512": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
              "shape_slat_flow_model_1024": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16",
              "tex_slat_decoder": "ckpts/tex_dec_next_dc_f16c32_fp16",
              "tex_slat_flow_model_512": "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16",
              "tex_slat_flow_model_1024": "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16"}
    for rel in pins["WEIGHT_FILES"]:
        os.makedirs(os.path.dirname(os.path.join(weights, rel)), exist_ok=True)
        open(os.path.join(weights, rel), "w").close()
    json.dump({"args": {"models": models}}, open(os.path.join(t2, "pipeline.json"), "w"))
    if not with_512_texture:
        os.remove(os.path.join(t2, models["tex_slat_flow_model_512"] + ".safetensors"))
