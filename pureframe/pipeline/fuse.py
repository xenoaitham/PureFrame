from pureframe.config import Config
from pureframe.pipeline.shots import Shot, ShotVerdict, Category, Action
from pureframe.pipeline.detect.nudity import Detection
from pureframe.pipeline.detect.scene_clip import ShotContext
from pureframe.pipeline.detect.audio import AudioContext


def fuse(
    shot: Shot,
    nudity_detections: list[list[Detection]],
    scene_ctx: ShotContext,
    audio_ctx: AudioContext,
    config: Config,
    strict_mode: bool = False,
) -> ShotVerdict:
    t_mod = 0.85 if strict_mode else 1.0

    # 1. Nudity
    max_nudity_score = 0.0
    has_any_nudity = False

    NUDITY_LABELS = {
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "BUTTOCKS_EXPOSED",
        "ANUS_EXPOSED",
    }

    for dets in nudity_detections:
        for d in dets:
            if d.label in NUDITY_LABELS:
                has_any_nudity = True
                if d.score > max_nudity_score:
                    max_nudity_score = d.score

    nudity_thresh = config.nudity_threshold * t_mod

    if max_nudity_score >= nudity_thresh:
        return ShotVerdict(
            shot_index=shot.index,
            category=Category.NUDITY_EXPLICIT,
            action=Action.BLACK_BOX,
            confidence=max_nudity_score,
            boxes=None,
            reasoning=f"Explicit nudity detected (score: {max_nudity_score:.2f})",
        )

    # 2. SEXUAL_ACT_VISIBLE
    if scene_ctx.explicit_act_score >= (
        0.40 * t_mod
    ) and audio_ctx.sexual_audio_score >= (0.30 * t_mod):
        action = Action.BLACK_BOX if has_any_nudity else Action.FULL_FRAME_BLUR
        return ShotVerdict(
            shot_index=shot.index,
            category=Category.SEXUAL_ACT_VISIBLE,
            action=action,
            confidence=min(scene_ctx.explicit_act_score, audio_ctx.sexual_audio_score),
            boxes=None,
            reasoning=f"Explicit sexual act: scene={scene_ctx.explicit_act_score:.2f} + audio={audio_ctx.sexual_audio_score:.2f}",
        )

    # 3. SEXUAL_CONTEXT_NO_NUDITY
    if scene_ctx.implied_sex_score >= (0.45 * t_mod) and (
        audio_ctx.moaning_score >= (0.35 * t_mod)
        or audio_ctx.sexual_audio_score >= (0.30 * t_mod)
    ):
        return ShotVerdict(
            shot_index=shot.index,
            category=Category.SEXUAL_CONTEXT_NO_NUDITY,
            action=Action.FULL_FRAME_BLUR,
            confidence=scene_ctx.implied_sex_score,
            boxes=None,
            reasoning=f"Implicit sex context: scene={scene_ctx.implied_sex_score:.2f} + audio=(moan={audio_ctx.moaning_score:.2f}, sex={audio_ctx.sexual_audio_score:.2f})",
        )

    # 4. KISS_INTENSE / KISS_LIGHT
    if scene_ctx.kissing_score >= (0.50 * t_mod):
        duration_seconds = shot.end_time - shot.start_time
        if duration_seconds >= 2.5:
            return ShotVerdict(
                shot_index=shot.index,
                category=Category.KISS_INTENSE,
                action=Action.BLACK_BOX,
                confidence=scene_ctx.kissing_score,
                boxes=None,
                reasoning=f"Intense kiss (duration {duration_seconds:.1f}s, score: {scene_ctx.kissing_score:.2f})",
            )
        else:
            return ShotVerdict(
                shot_index=shot.index,
                category=Category.KISS_LIGHT,
                action=Action.NONE,
                confidence=scene_ctx.kissing_score,
                boxes=None,
                reasoning=f"Light kiss (duration {duration_seconds:.1f}s, score: {scene_ctx.kissing_score:.2f})",
            )

    # 5. SAFE
    return ShotVerdict(
        shot_index=shot.index,
        category=Category.SAFE,
        action=Action.NONE,
        confidence=scene_ctx.safe_score,
        boxes=None,
        reasoning=f"Safe scene (score: {scene_ctx.safe_score:.2f})",
    )
