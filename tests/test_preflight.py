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
    assert "6 of 11" in preflight.profile_advice(16376 / 1024)[1]
    assert "6 of 6" in preflight.profile_advice(12282 / 1024)[1] and "6 of 6" in preflight.profile_advice(8188 / 1024)[1]
    assert "under 8 GB is not tested" in preflight.profile_advice(6144 / 1024)[1]


def test_profile_model_files(tmp_path):
    install_weights(str(tmp_path / "w"), with_512_texture=False)
    absent = preflight.profile_model_files(str(tmp_path / "w"), PINS)
    assert absent["1024_cascade"] == []
    assert absent["512"] == ["ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16"]


def test_bus_key_matches_nvidia_smi_and_cuda_forms():
    assert preflight.bus_key("00000000:01:00.0") == preflight.bus_key("0000:01:00.0") == "0000:01:00.0"
    assert preflight.bus_key("00000001:0A:00.0") == "0001:0a:00.0"


def test_uuid_text():
    assert preflight.uuid_text(bytes(range(16))) == "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f"


SMI = [("0", "GPU-a", "0000:01:00.0"), ("1", "GPU-b", "0000:03:00.0"), ("2", "GPU-c", "0000:06:00.0")]


def test_gpu_numbers_same_order_pass():
    lines = preflight.gpu_number_lines([(0, "GPU-a", "0000:01:00.0"), (1, "GPU-b", "0000:03:00.0"),
                                        (2, "GPU-c", "0000:06:00.0")], SMI)
    assert [level for level, _, _ in lines] == ["PASS"] * 3


def test_gpu_numbers_fastest_first_warns():
    # CUDA's default order puts the faster GPU (nvidia-smi GPU 2) first
    lines = preflight.gpu_number_lines([(0, "GPU-c", "0000:06:00.0"), (1, "GPU-a", "0000:01:00.0"),
                                        (2, "GPU-b", "0000:03:00.0")], SMI)
    assert all(level == "WARN" and "CUDA_DEVICE_ORDER=PCI_BUS_ID" in fix for level, _, fix in lines)
    assert "CUDA device 0 is nvidia-smi GPU 2" in lines[0][1]


def test_gpu_numbers_selected_with_cuda_visible_devices_pass():
    lines = preflight.gpu_number_lines([(0, "GPU-c", "0000:06:00.0")], SMI, visible="2")
    assert lines[0][0] == "PASS" and "CUDA_VISIBLE_DEVICES=2" in lines[0][1]


def test_gpu_numbers_unknown_uuid_warns():
    assert preflight.gpu_number_lines([(0, "GPU-x", "0000:09:00.0")], SMI)[0][0] == "WARN"


ALL_TARGETS = "8.0;8.6;8.9;9.0;10.0;12.0+PTX"


@pytest.mark.parametrize("arch_list, cap, want", [
    (ALL_TARGETS, "8.0", True), (ALL_TARGETS, "8.6", True), (ALL_TARGETS, "8.7", True), (ALL_TARGETS, "8.9", True),
    (ALL_TARGETS, "9.0", True), (ALL_TARGETS, "10.0", True), (ALL_TARGETS, "10.3", True), (ALL_TARGETS, "12.0", True),
    (ALL_TARGETS, "12.1", True), (ALL_TARGETS, "13.0", True), (ALL_TARGETS, "7.5", False), (ALL_TARGETS, "11.0", False),
    ("8.6", "8.9", True), ("8.6", "8.0", False), ("8.6", "9.0", False), ("8.6", "12.0", False),
    ("8.6 8.9", "8.9", True), ("8.6+PTX", "12.0", True), ("9.0a", "9.0", True), ("10.0a", "10.3", False),
])
def test_arch_list_runs_on(arch_list, cap, want):
    assert preflight.arch_list_runs_on(arch_list, cap) is want


def test_gpu_outside_the_arch_list_fails(monkeypatch, capsys):
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.6")
    preflight.check_one_gpu("0", "NVIDIA H100", 79.6, 0.0, "9.0")
    assert "FAIL  GPU 0: the CUDA code is built for TORCH_CUDA_ARCH_LIST=8.6" in capsys.readouterr().out
