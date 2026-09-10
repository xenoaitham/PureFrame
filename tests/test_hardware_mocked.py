"""Tests for hardware profile detection with mocked torch for full coverage.

``detect_profile`` imports torch inside the function, so patching
``sys.modules["torch"]`` is enough to drive it. The reload dance this file
once used re-executed the module and replaced its classes (HardwareProfile,
ProfileSettings) - every later ``isinstance`` against the originals failed
depending on test order. Compare by ``.value`` here instead.
"""

from unittest.mock import MagicMock, patch

from pureframe.hardware import get_settings


def _detect_with_vram(free_gb: float):
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.mem_get_info.return_value = (
        free_gb * 1024**3,
        (free_gb + 4) * 1024**3,
    )
    with patch.dict("sys.modules", {"torch": mock_torch}):
        from pureframe.hardware import detect_profile

        profile = detect_profile()
    return profile.value


class TestDetectProfileMocked:
    def test_high_vram_returns_high(self):
        assert _detect_with_vram(12) == "HIGH"

    def test_medium_vram_returns_medium(self):
        assert _detect_with_vram(8) == "MEDIUM"

    def test_low_vram_returns_low(self):
        assert _detect_with_vram(4) == "LOW"

    def test_tiny_vram_returns_cpu(self):
        assert _detect_with_vram(1) == "CPU"

    def test_no_cuda_returns_cpu(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            from pureframe.hardware import detect_profile

            assert detect_profile().value == "CPU"

    def test_torch_missing_returns_cpu(self):
        real_torch = __import__("sys").modules.get("torch", None)
        with patch.dict("sys.modules", {"torch": None}):
            from pureframe.hardware import detect_profile

            assert detect_profile().value == "CPU"
        assert (
            __import__("sys").modules.get("torch", None) is not None
            or real_torch is None
        )

    def test_torch_error_returns_cpu(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("boom")
        with patch.dict("sys.modules", {"torch": mock_torch}):
            from pureframe.hardware import detect_profile

            assert detect_profile().value == "CPU"

    def test_out_of_range_device_profiles_device_zero(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.mem_get_info.return_value = (12 * 1024**3, 16 * 1024**3)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            from pureframe.hardware import detect_profile

            assert detect_profile(3).value == "HIGH"
            mock_torch.cuda.mem_get_info.assert_called_with(0)


class TestHardwareProfiles:
    def test_all_profiles_return_valid_settings(self):
        from pureframe.hardware import HardwareProfile

        for profile in HardwareProfile:
            settings = get_settings(profile)
            assert settings.profile == profile
            assert settings.detection_resolution > 0
            assert settings.detection_batch_size > 0
            assert settings.sample_keyframes_per_shot > 0
            assert settings.onnx_providers

    def test_detect_profile_returns_valid(self):
        from pureframe.hardware import HardwareProfile, detect_profile

        assert detect_profile() in list(HardwareProfile)
