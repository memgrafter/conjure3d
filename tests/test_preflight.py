"""tools/preflight.py: the parts that need no GPU and no network."""
import pytest
from conftest import install_weights, load

preflight = load("tools/preflight.py", "preflight")
PINS = preflight.pipeline_constants()


@pytest.mark.parametrize("header,want", [
    ("| NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3     |", (13, 3)),
    ("| NVIDIA-SMI 580.65.06    Driver Version: 580.65.06    CUDA Version: 13.0     |", (13, 0)),
    ("no header", None),
])
def test_driver_cuda_version(header, want):
    assert preflight.driver_cuda_version(header) == want


def test_profile_advice_follows_measured_results():
    assert preflight.profile_advice(24564 / 1024)[0] == "PASS"
    assert "6 of 10" in preflight.profile_advice(16376 / 1024)[1]
    assert "6 of 6" in preflight.profile_advice(12282 / 1024)[1] and "6 of 6" in preflight.profile_advice(8188 / 1024)[1]
    assert "under 8 GB is not tested" in preflight.profile_advice(6144 / 1024)[1]


def test_profile_model_files(tmp_path):
    install_weights(str(tmp_path / "w"), with_512_texture=False)
    absent = preflight.profile_model_files(str(tmp_path / "w"), PINS)
    assert absent["1024_cascade"] == []
    assert absent["512"] == ["ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16"]
