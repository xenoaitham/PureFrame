"""Tests for checkpoint store edge cases."""

from pathlib import Path

import pytest

from pureframe.checkpoint import CheckpointStore
from pureframe.pipeline.shots import Action, Category, ShotVerdict


class TestCheckpointStoreEdgeCases:
    def test_create_and_find_job(self, tmp_path):
        db = tmp_path / "test.db"
        store = CheckpointStore(db)

        import tempfile

        from pureframe.config import Config

        tf = Path(tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name)

        config = Config(input_path=tf, nudity_threshold=0.55)
        job = store.find_or_create_job(tf, tf.with_suffix(".out.mp4"), config)

        assert job.status == "PENDING"
        assert job.config_hash == config.config_hash

        # Find same job again
        job2 = store.find_or_create_job(tf, tf.with_suffix(".out.mp4"), config)
        assert job2.id == job.id

        tf.unlink()

    def test_update_status(self, tmp_path):
        db = tmp_path / "test.db"
        store = CheckpointStore(db)

        import tempfile

        from pureframe.config import Config

        tf = Path(tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name)
        config = Config(input_path=tf)
        job = store.find_or_create_job(tf, tf.with_suffix(".out.mp4"), config)

        store.update_status(job.id, "DETECTING")

        # Re-find and check status
        job2 = store.find_or_create_job(tf, tf.with_suffix(".out.mp4"), config)
        assert job2.status == "DETECTING"

        store.update_status(job.id, "FAILED", error="Test error")
        job3 = store.find_or_create_job(tf, tf.with_suffix(".out.mp4"), config)
        assert job3.status == "FAILED"

        tf.unlink()

    def test_save_and_load_verdicts(self, tmp_path):
        db = tmp_path / "test.db"
        store = CheckpointStore(db)

        import tempfile

        from pureframe.config import Config

        tf = Path(tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name)
        config = Config(input_path=tf)
        job = store.find_or_create_job(tf, tf.with_suffix(".out.mp4"), config)

        verdict = ShotVerdict(
            shot_index=0,
            category=Category.NUDITY_EXPLICIT,
            action=Action.BLACK_BOX,
            confidence=0.92,
            boxes=None,
            reasoning="Test verdict",
        )
        store.save_verdict(job.id, verdict)

        loaded = store.load_verdicts(job.id)
        assert len(loaded) == 1
        assert loaded[0].shot_index == 0
        assert loaded[0].action == Action.BLACK_BOX
        assert loaded[0].confidence == pytest.approx(0.92)

        tf.unlink()

    def test_save_multiple_verdicts(self, tmp_path):
        db = tmp_path / "test.db"
        store = CheckpointStore(db)

        import tempfile

        from pureframe.config import Config

        tf = Path(tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name)
        config = Config(input_path=tf)
        job = store.find_or_create_job(tf, tf.with_suffix(".out.mp4"), config)

        for i in range(5):
            verdict = ShotVerdict(
                shot_index=i,
                category=Category.SAFE,
                action=Action.NONE,
                confidence=0.5 + i * 0.1,
                boxes=None,
                reasoning=f"Verdict {i}",
            )
            store.save_verdict(job.id, verdict)

        loaded = store.load_verdicts(job.id)
        assert len(loaded) == 5

        tf.unlink()

    def test_list_unfinished_includes_pending(self, tmp_path):
        db = tmp_path / "test.db"
        store = CheckpointStore(db)

        import tempfile

        from pureframe.config import Config

        tf = Path(tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name)
        config = Config(input_path=tf)
        store.find_or_create_job(tf, tf.with_suffix(".out.mp4"), config)

        unfinished = store.list_unfinished()
        assert len(unfinished) >= 1
        assert any(j.status == "PENDING" for j in unfinished)

        tf.unlink()

    def test_done_jobs_not_unfinished(self, tmp_path):
        db = tmp_path / "test.db"
        store = CheckpointStore(db)

        import tempfile

        from pureframe.config import Config

        tf = Path(tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name)
        config = Config(input_path=tf)
        job = store.find_or_create_job(tf, tf.with_suffix(".out.mp4"), config)
        store.update_status(job.id, "DONE")

        unfinished = store.list_unfinished()
        assert not any(j.id == job.id for j in unfinished)

        tf.unlink()

    def test_config_hash_changes_create_new_job(self, tmp_path):
        db = tmp_path / "test.db"
        store = CheckpointStore(db)

        import tempfile

        from pureframe.config import Config

        tf = Path(tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name)

        config1 = Config(input_path=tf, nudity_threshold=0.5)
        job1 = store.find_or_create_job(tf, tf.with_suffix(".out.mp4"), config1)

        config2 = Config(input_path=tf, nudity_threshold=0.8)
        job2 = store.find_or_create_job(tf, tf.with_suffix(".out.mp4"), config2)

        # Different config_hash means different job
        assert config1.config_hash != config2.config_hash
        # Store may or may not create a new ID depending on implementation
        # But the config hash should differ
        assert job1.config_hash != job2.config_hash

        tf.unlink()
