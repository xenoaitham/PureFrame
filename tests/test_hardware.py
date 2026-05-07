"""Tests for hardware profile detection and settings."""

from pureframe.hardware import (
    HardwareProfile,
    ProfileSettings,
    get_settings,
    detect_profile,
)


class TestHardwareProfiles:
    def test_all_profiles_return_valid_settings(self):
        for profile in HardwareProfile:
            settings = get_settings(profile)
            assert isinstance(settings, ProfileSettings)
            assert settings.profile == profile
            assert settings.detection_resolution > 0
            assert settings.detection_batch_size >= 1
            assert settings.sample_keyframes_per_shot >= 1
            assert settings.densify_every_n_frames >= 1
            assert len(settings.onnx_providers) >= 1

    def test_high_profile_has_highest_resolution(self):
        high = get_settings(HardwareProfile.HIGH)
        medium = get_settings(HardwareProfile.MEDIUM)
        low = get_settings(HardwareProfile.LOW)
        cpu = get_settings(HardwareProfile.CPU)

        assert high.detection_resolution >= medium.detection_resolution
        assert medium.detection_resolution >= low.detection_resolution
        assert low.detection_resolution >= cpu.detection_resolution

    def test_high_profile_has_largest_batch(self):
        high = get_settings(HardwareProfile.HIGH)
        cpu = get_settings(HardwareProfile.CPU)

        assert high.detection_batch_size > cpu.detection_batch_size

    def test_cpu_profile_no_cuda(self):
        cpu = get_settings(HardwareProfile.CPU)
        assert cpu.use_fp16 is False
        assert cpu.keep_models_loaded is False
        assert "CUDAExecutionProvider" not in cpu.onnx_providers

    def test_high_profile_uses_fp16(self):
        high = get_settings(HardwareProfile.HIGH)
        assert high.use_fp16 is True
        assert high.keep_models_loaded is True

    def test_detect_profile_returns_valid(self):
        profile = detect_profile()
        assert isinstance(profile, HardwareProfile)
        assert profile in HardwareProfile

    def test_profile_settings_serializable(self):
        for profile in HardwareProfile:
            settings = get_settings(profile)
            data = settings.model_dump()
            assert "profile" in data
            assert "detection_resolution" in data
            restored = ProfileSettings(**data)
            assert restored.profile == profile
