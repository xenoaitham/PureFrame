"""Tests for hardware profile detection with mocked torch for full coverage."""

from unittest.mock import patch, MagicMock
from pureframe.hardware import HardwareProfile, get_settings


class TestDetectProfileMocked:
    def test_high_vram_returns_high(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.return_value = (
            12 * 1024**3,
            16 * 1024**3,
        )  # 12GB free

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("pureframe.hardware.torch", mock_torch, create=True):
                # Reimport to use patched module
                import importlib
                import pureframe.hardware

                importlib.reload(pureframe.hardware)

                profile = pureframe.hardware.detect_profile()
                assert profile == HardwareProfile.HIGH

                importlib.reload(pureframe.hardware)

    def test_medium_vram_returns_medium(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.return_value = (8 * 1024**3, 12 * 1024**3)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("pureframe.hardware.torch", mock_torch, create=True):
                import importlib
                import pureframe.hardware

                importlib.reload(pureframe.hardware)

                profile = pureframe.hardware.detect_profile()
                assert profile == HardwareProfile.MEDIUM

                importlib.reload(pureframe.hardware)

    def test_low_vram_returns_low(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.return_value = (4 * 1024**3, 6 * 1024**3)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("pureframe.hardware.torch", mock_torch, create=True):
                import importlib
                import pureframe.hardware

                importlib.reload(pureframe.hardware)

                profile = pureframe.hardware.detect_profile()
                assert profile == HardwareProfile.LOW

                importlib.reload(pureframe.hardware)

    def test_tiny_vram_returns_cpu(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.return_value = (1 * 1024**3, 2 * 1024**3)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("pureframe.hardware.torch", mock_torch, create=True):
                import importlib
                import pureframe.hardware

                importlib.reload(pureframe.hardware)

                profile = pureframe.hardware.detect_profile()
                assert profile == HardwareProfile.CPU

                importlib.reload(pureframe.hardware)

    def test_no_cuda_returns_cpu(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("pureframe.hardware.torch", mock_torch, create=True):
                import importlib
                import pureframe.hardware

                importlib.reload(pureframe.hardware)

                profile = pureframe.hardware.detect_profile()
                assert profile == HardwareProfile.CPU

                importlib.reload(pureframe.hardware)

    def test_torch_exception_returns_cpu(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA init failed")

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("pureframe.hardware.torch", mock_torch, create=True):
                import importlib
                import pureframe.hardware

                importlib.reload(pureframe.hardware)

                profile = pureframe.hardware.detect_profile()
                assert profile == HardwareProfile.CPU

                importlib.reload(pureframe.hardware)


class TestGetSettingsAllProfiles:
    def test_all_profiles_have_unique_resolution(self):
        resolutions = set()
        for profile in HardwareProfile:
            settings = get_settings(profile)
            resolutions.add(settings.detection_resolution)
        assert len(resolutions) == 4  # Each profile has unique resolution

    def test_densify_decreases_with_power(self):
        high = get_settings(HardwareProfile.HIGH)
        cpu = get_settings(HardwareProfile.CPU)
        assert high.densify_every_n_frames <= cpu.densify_every_n_frames
