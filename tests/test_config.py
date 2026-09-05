"""Tests for Config model, content-type profiles, strictness, and threshold calculation."""

import os
import tempfile
from pathlib import Path

import pytest

from pureframe.config import (
    Config,
    ContentType,
    Strictness,
)


@pytest.fixture
def dummy_file():
    """Create a temporary file for Config tests."""
    tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tf.close()
    yield Path(tf.name)
    os.unlink(tf.name)


class TestContentTypeProfiles:
    def test_live_action_default(self, dummy_file):
        c = Config(input_path=dummy_file)
        assert c.content_type == ContentType.LIVE_ACTION
        assert c.strictness == Strictness.MEDIUM

    def test_all_content_types_valid(self, dummy_file):
        for ct in ContentType:
            c = Config(input_path=dummy_file, content_type=ct)
            n, cl, a = c.get_effective_thresholds()
            assert 0.0 < n < 1.0
            assert 0.0 < cl < 1.0
            assert 0.0 < a < 1.0

    def test_anime_raises_thresholds(self, dummy_file):
        c_live = Config(input_path=dummy_file, content_type=ContentType.LIVE_ACTION)
        c_anime = Config(input_path=dummy_file, content_type=ContentType.ANIME)

        n_live, _, _ = c_live.get_effective_thresholds()
        n_anime, _, _ = c_anime.get_effective_thresholds()

        # Anime should have higher thresholds (1.4x multiplier)
        assert n_anime > n_live

    def test_low_light_lowers_thresholds(self, dummy_file):
        c_live = Config(input_path=dummy_file, content_type=ContentType.LIVE_ACTION)
        c_dark = Config(input_path=dummy_file, content_type=ContentType.LOW_LIGHT)

        n_live, _, _ = c_live.get_effective_thresholds()
        n_dark, _, _ = c_dark.get_effective_thresholds()

        # Low-light should have lower thresholds (0.85x multiplier)
        assert n_dark < n_live

    def test_animation_between_live_and_anime(self, dummy_file):
        c_live = Config(input_path=dummy_file, content_type=ContentType.LIVE_ACTION)
        c_anim = Config(input_path=dummy_file, content_type=ContentType.ANIMATION)
        c_anime = Config(input_path=dummy_file, content_type=ContentType.ANIME)

        n_live, _, _ = c_live.get_effective_thresholds()
        n_anim, _, _ = c_anim.get_effective_thresholds()
        n_anime, _, _ = c_anime.get_effective_thresholds()

        assert n_live < n_anim < n_anime


class TestStrictnessLevels:
    def test_all_strictness_levels_valid(self, dummy_file):
        for s in Strictness:
            c = Config(input_path=dummy_file, strictness=s)
            n, cl, a = c.get_effective_thresholds()
            assert 0.0 < n < 1.0
            assert 0.0 < cl < 1.0
            assert 0.0 < a < 1.0

    def test_high_strictness_lower_thresholds(self, dummy_file):
        c_high = Config(input_path=dummy_file, strictness=Strictness.HIGH)
        c_low = Config(input_path=dummy_file, strictness=Strictness.LOW)

        n_high, _, _ = c_high.get_effective_thresholds()
        n_low, _, _ = c_low.get_effective_thresholds()

        # High strictness = lower thresholds = more aggressive flagging
        assert n_high < n_low

    def test_custom_uses_manual_thresholds(self, dummy_file):
        c = Config(
            input_path=dummy_file,
            strictness=Strictness.CUSTOM,
            nudity_threshold=0.42,
            clip_threshold=0.33,
            audio_threshold=0.77,
        )
        n, cl, a = c.get_effective_thresholds()
        assert n == pytest.approx(0.42)
        assert cl == pytest.approx(0.33)
        assert a == pytest.approx(0.77)

    def test_custom_with_content_type_multiplier(self, dummy_file):
        c = Config(
            input_path=dummy_file,
            strictness=Strictness.CUSTOM,
            nudity_threshold=0.50,
            content_type=ContentType.ANIME,  # 1.4x multiplier
        )
        n, _, _ = c.get_effective_thresholds()
        assert n == pytest.approx(0.50 * 1.4)

    def test_threshold_capped_at_099(self, dummy_file):
        c = Config(
            input_path=dummy_file,
            strictness=Strictness.CUSTOM,
            nudity_threshold=0.90,
            content_type=ContentType.ANIME,  # 1.4x would push to 1.26
        )
        n, _, _ = c.get_effective_thresholds()
        assert n == 0.99


class TestConfigHash:
    def test_same_config_same_hash(self, dummy_file):
        c1 = Config(input_path=dummy_file, nudity_threshold=0.55)
        c2 = Config(input_path=dummy_file, nudity_threshold=0.55)
        assert c1.config_hash == c2.config_hash

    def test_different_threshold_different_hash(self, dummy_file):
        c1 = Config(input_path=dummy_file, nudity_threshold=0.55)
        c2 = Config(input_path=dummy_file, nudity_threshold=0.65)
        assert c1.config_hash != c2.config_hash

    def test_different_content_type_different_hash(self, dummy_file):
        c1 = Config(input_path=dummy_file, content_type=ContentType.LIVE_ACTION)
        c2 = Config(input_path=dummy_file, content_type=ContentType.ANIME)
        assert c1.config_hash != c2.config_hash

    def test_hash_is_deterministic(self, dummy_file):
        c = Config(input_path=dummy_file)
        h1 = c.config_hash
        h2 = c.config_hash
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex digest


class TestConfigFromCli:
    def test_nonexistent_file_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            Config.from_cli(input_path=Path("/nonexistent/video.mp4"))

    def test_auto_output_path(self, dummy_file):
        c = Config.from_cli(input_path=dummy_file)
        expected = dummy_file.with_name(
            f"{dummy_file.stem}.pureframe{dummy_file.suffix}"
        )
        assert c.output_path == expected

    def test_explicit_output_path(self, dummy_file, tmp_path):
        out = tmp_path / "custom_output.mp4"
        c = Config.from_cli(input_path=dummy_file, output_path=out)
        assert c.output_path == out

    def test_default_values(self, dummy_file):
        c = Config.from_cli(input_path=dummy_file)
        assert c.nudity_threshold == 0.55
        assert c.content_type == ContentType.LIVE_ACTION
        assert c.strictness == Strictness.MEDIUM
        assert c.strict is False
        assert c.no_clip is False
        assert c.no_audio is False
        assert c.force is False
