"""--device selection: CUDA device pinning for the ML models.

Full multi-GPU sharding is analyzed and descoped in docs/multi-gpu.md; what
ships is single active device selection: profile detection reads the chosen
device's VRAM, the ONNX CUDA EP is pinned via provider options, and the
torch models resolve to cuda:<n>. Verdicts don't depend on the device, so
it stays out of the checkpoint config hash.
"""

from pureframe.config import Config
from pureframe.hardware import (
    get_settings,
    onnx_providers_for,
    torch_device_str,
)


class TestOnnxProvidersFor:
    def test_device_zero_leaves_providers_untouched(self):
        base = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert onnx_providers_for(base, 0) == base

    def test_cpu_only_profiles_untouched(self):
        assert onnx_providers_for(["CPUExecutionProvider"], 3) == [
            "CPUExecutionProvider"
        ]

    def test_positive_device_pins_the_cuda_ep(self):
        pinned = onnx_providers_for(
            ["CUDAExecutionProvider", "CPUExecutionProvider"], 2
        )
        assert pinned[0] == ("CUDAExecutionProvider", {"device_id": "2"})
        assert list(pinned[1:]) == ["CPUExecutionProvider"]

    def test_provider_order_preserved(self):
        base = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        pinned = onnx_providers_for(base, 1)
        assert [p if isinstance(p, str) else p[0] for p in pinned] == base


class TestTorchDeviceStr:
    def test_falls_back_to_cpu_without_cuda(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch, "cuda", torch.cuda)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert torch_device_str(1) == "cpu"

    def test_out_of_range_device_falls_back_to_cpu(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        assert torch_device_str(5) == "cpu"

    def test_valid_device_names_it(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        assert torch_device_str(1) == "cuda:1"


def test_detect_profile_rejects_unknown_device_gracefully(monkeypatch):
    """A nonexistent index must profile device 0, not crash."""
    import torch

    from pureframe.hardware import HardwareProfile, detect_profile

    seen = {}

    def fake_mem_info(device):
        seen["device"] = device
        return (12 * (1 << 30), 24 * (1 << 30))  # 12 GB free → HIGH

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", fake_mem_info)

    assert detect_profile(3) == HardwareProfile.HIGH
    assert seen["device"] == 0


class TestDeviceConfig:
    def test_device_excluded_from_config_hash(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"\x00" * 16)
        with_device = Config(input_path=video, device=1)
        without = Config(input_path=video)
        assert with_device.config_hash == without.config_hash

    def test_get_settings_threads_the_device(self):
        settings = get_settings(
            __import__(
                "pureframe.hardware", fromlist=["HardwareProfile"]
            ).HardwareProfile.HIGH,
            cuda_device=1,
        )
        assert settings.cuda_device == 1
        assert get_settings(settings.profile).cuda_device == 0
