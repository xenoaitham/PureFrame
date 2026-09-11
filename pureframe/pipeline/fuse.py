from pureframe.config import Config
from pureframe.pipeline.detect.audio import AudioContext
from pureframe.pipeline.detect.nudity import Detection
from pureframe.pipeline.detect.scene_clip import ShotContext
from pureframe.pipeline.shots import Action, Category, Shot, ShotVerdict

# Primary explicit nudity labels - always flagged
NUDITY_EXPLICIT_LABELS = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}

# Partial nudity labels - flagged at higher strictness
NUDITY_PARTIAL_LABELS = {
    "FEMALE_BREAST_COVERED",
    "BELLY_EXPOSED",
    "ARMPITS_EXPOSED",
}

# All nudity-related labels
ALL_NUDITY_LABELS = NUDITY_EXPLICIT_LABELS | NUDITY_PARTIAL_LABELS


def context_audio_needed(
    scene_ctx: ShotContext, config: Config, strict_mode: bool = False
) -> bool:
    """True when the audio classifier's score can change the fused verdict.

    fuse() consults audio only in the two sexual-act branches, and both
    require the CLIP scene signal to already be at/above its own threshold -
    audio alone can never push a verdict over. When the scene scores sit
    below those thresholds, the (expensive, per-shot) PANNs run is provably
    irrelevant: skipping it and passing a neutral AudioContext yields the
    identical verdict. On typical content this skips the audio model for the
    vast majority of shots; with --no-clip (neutral scene context) it skips
    always, matching that audio could never have mattered there anyway.
    """
    _, eff_clip, eff_audio = config.get_effective_thresholds()
    t_mod = 0.85 if strict_mode else 1.0

    explicit_act_thresh = 0.40 * t_mod * (eff_clip / 0.50)
    implied_sex_thresh = 0.45 * t_mod * (eff_clip / 0.50)

    return (
        scene_ctx.explicit_act_score >= explicit_act_thresh
        or scene_ctx.implied_sex_score >= implied_sex_thresh
    )


def fuse(
    shot: Shot,
    nudity_detections: list[list[Detection]],
    scene_ctx: ShotContext,
    audio_ctx: AudioContext,
    config: Config,
    strict_mode: bool = False,
    plugin_detections: dict[str, list[list[Detection]]] | None = None,
    plugin_threshold_bases: dict[str, float] | None = None,
) -> ShotVerdict:
    # Use effective thresholds from config (includes content-type and strictness adjustments)
    eff_nudity, eff_clip, eff_audio = config.get_effective_thresholds()

    # Legacy strict_mode applies an additional 0.85 multiplier for backward compat
    t_mod = 0.85 if strict_mode else 1.0

    nudity_thresh = eff_nudity * t_mod

    # 1. Explicit Nudity
    max_nudity_score = 0.0
    has_any_nudity = False
    max_partial_score = 0.0

    for dets in nudity_detections:
        for d in dets:
            if d.label in NUDITY_EXPLICIT_LABELS:
                has_any_nudity = True
                if d.score > max_nudity_score:
                    max_nudity_score = d.score
            elif d.label in NUDITY_PARTIAL_LABELS:
                if d.score > max_partial_score:
                    max_partial_score = d.score

    if max_nudity_score >= nudity_thresh:
        return ShotVerdict(
            shot_index=shot.index,
            category=Category.NUDITY_EXPLICIT,
            action=Action.BLACK_BOX,
            confidence=max_nudity_score,
            boxes=None,
            reasoning=f"Explicit nudity detected (score: {max_nudity_score:.2f}, threshold: {nudity_thresh:.2f})",
        )

    # 2. SEXUAL_ACT_VISIBLE - requires both visual and audio signals
    explicit_act_thresh = (
        0.40 * t_mod * (eff_clip / 0.50)
    )  # Scale by effective clip threshold
    sexual_audio_thresh = 0.30 * t_mod * (eff_audio / 0.60)

    if (
        scene_ctx.explicit_act_score >= explicit_act_thresh
        and audio_ctx.sexual_audio_score >= sexual_audio_thresh
    ):
        action = Action.BLACK_BOX if has_any_nudity else Action.FULL_FRAME_BLUR
        return ShotVerdict(
            shot_index=shot.index,
            category=Category.SEXUAL_ACT_VISIBLE,
            action=action,
            confidence=min(scene_ctx.explicit_act_score, audio_ctx.sexual_audio_score),
            boxes=None,
            reasoning=f"Explicit sexual act: scene={scene_ctx.explicit_act_score:.2f} + audio={audio_ctx.sexual_audio_score:.2f}",
        )

    # 3. SEXUAL_CONTEXT_NO_NUDITY - implied sex with audio cues
    implied_sex_thresh = 0.45 * t_mod * (eff_clip / 0.50)
    moaning_thresh = 0.35 * t_mod * (eff_audio / 0.60)

    if scene_ctx.implied_sex_score >= implied_sex_thresh and (
        audio_ctx.moaning_score >= moaning_thresh
        or audio_ctx.sexual_audio_score >= sexual_audio_thresh
    ):
        return ShotVerdict(
            shot_index=shot.index,
            category=Category.SEXUAL_CONTEXT_NO_NUDITY,
            action=Action.FULL_FRAME_BLUR,
            confidence=scene_ctx.implied_sex_score,
            boxes=None,
            reasoning=f"Implicit sex context: scene={scene_ctx.implied_sex_score:.2f} + audio=(moan={audio_ctx.moaning_score:.2f}, sex={audio_ctx.sexual_audio_score:.2f})",
        )

    # 4. Plugin box-provider categories - after the nudity and sexual-
    # context branches so a plugin can never downgrade a verdict those
    # paths already made, and before the kiss branches so box-provider
    # evidence still censors a shot the scene classifier would leave at
    # KISS_LIGHT (Action.NONE, rendered uncensored). The branch never
    # consults the audio context: a box provider fires on visual
    # detections alone, at its category's effective threshold (plugin
    # override, else the plugin's declared base; content-type multiplier
    # and the strict-mode factor apply on top, same as the built-ins).
    if plugin_detections:
        bases = plugin_threshold_bases or {}
        eff_plugin = config.get_effective_plugin_thresholds(bases)
        plugin_t_mod = 0.85 if strict_mode else 1.0
        for plugin_category, per_frame in plugin_detections.items():
            threshold = eff_plugin.get(plugin_category, 0.99) * plugin_t_mod
            best = max((d.score for dets in per_frame for d in dets), default=0.0)
            if best >= threshold:
                return ShotVerdict(
                    shot_index=shot.index,
                    category=Category.PLUGIN_BOX,
                    action=Action.BLACK_BOX,
                    confidence=best,
                    boxes=None,
                    plugin_category=plugin_category,
                    reasoning=(
                        f"Plugin category {plugin_category} detected "
                        f"(score: {best:.2f}, threshold: {threshold:.2f})"
                    ),
                )

    # 5. KISS_INTENSE / KISS_LIGHT
    kissing_thresh = 0.50 * t_mod * (eff_clip / 0.50)
    if scene_ctx.kissing_score >= kissing_thresh:
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

    # 6. SAFE
    return ShotVerdict(
        shot_index=shot.index,
        category=Category.SAFE,
        action=Action.NONE,
        confidence=scene_ctx.safe_score,
        boxes=None,
        reasoning=f"Safe scene (score: {scene_ctx.safe_score:.2f})",
    )
