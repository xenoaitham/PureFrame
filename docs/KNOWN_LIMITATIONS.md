# Known Limitations & Failure Cases

> PureFrame is honest about what it can and can't do. This document
> lists where the tool falls short and what was done about it. After
> the 2026-09-17 engineering offensive it has three sections: fixed
> (merged with a regression test and, where noted, an eval-corpus
> pin), improved (engineered, with a stated residual), and accepted
> (not fixable in code, with the reason).

## Fixed in this offensive

Each row names its pull request and the test that fails without it.

| Case | What landed | Proof |
|------|-------------|-------|
| **Flash frames** | Second-pass rescan: shots with a weak-but-present signal (any detection at or above the 0.25 floor) or a parental-guide mark re-sample at the profile's densify stride, so sub-sampled brief content lands on a sampled frame. | #128, `tests/test_second_pass.py` |
| **Small/distant nudity and screen-in-screen** | Frames the full-frame pass still reads below the bar go through an overlapping 2x2 tile grid at full detection resolution, merged back with class-aware NMS. Candidates only, never the whole video. | #128, `tests/test_tiling.py`, `tests/test_second_pass.py`, eval pin EC-011 |
| **Very dark scenes** | Frames measuring dark (mean luma under 70) get a percentile stretch before inference - detection only, the render keeps the original pixels. | #129, `tests/test_preprocess.py`, eval pin DK-006 |
| **Artistic B&W / grayscale** | Near-grayscale frames (channel spread at most 12) with a compressed histogram (luma std under 30) take the same stretch; a linear map shared across channels keeps them grayscale. | #129, `tests/test_preprocess.py` |
| **Swimwear, beach/pool** | CLIP scene-context gate: confident beach/pool context (score 0.60+) with no sexual context raises the shot's nudity bar 1.4x. The gate can only raise a bar; sexual context switches it off. | #130, `tests/test_scene_context_gate.py` |
| **Male shirtless, gym/sports** | Same gate, gym/sports context category. | #130, `tests/test_scene_context_gate.py` |
| **Classical art (paintings, statues)** | `--content-type art` (1.5x bar) plus a museum/gallery context gate per shot. Real explicit scenes in the same film still flag. | #131, `tests/test_art_medical.py`, eval pin ART-001 |
| **Medical content** | `--content-type medical` (1.4x bar) plus a medical/clinical context gate per shot. | #131, `tests/test_art_medical.py`, eval pin MED-001 |
| **Extreme close-ups** | Detector-silent shots in a sexual scene re-run the detector on exact center-crop quadrants (2x zoom) of the keyframes already in memory. | #132, `tests/test_closeup.py`, eval pin EC-012 |
| **Variable frame rate** | The probe detects VFR (peak vs average rate past 2%, or frame count vs duration drift past 5%) and converts to CFR automatically into a temp dir - no more hand-running ffmpeg. | #133, `tests/test_vfr.py` |
| **AV1 sources** | AV1 joins the source-matched encoders: libsvtav1 preferred, libaom-av1 fallback, so AV1 WebM re-encodes in AV1 instead of dying at the H.264 mux. Environments whose AV1 encoder rejects or hangs on the encode skip the AV1 container tests and take the full re-encode path. | #134, `tests/test_container_formats.py` (webm-av1) |

## Improved

Engineered this offensive, with the residual stated. Nothing here
claims more than the tests show.

| Case | What changed and proof | Residual |
|------|------------------------|----------|
| **HDR tone mapping** | HDR10 mastering display and content light level survive the re-encode (re-injected into x265; read from stream or first-frame SEI side data), and the color tags (bt2020/smpte2084) stay on the output, so players no longer read the wrong transfer function. #136, `tests/test_hdr.py`. | The frame pipe is 8-bit BGR, so the full 10-bit signal is not reconstructed; the metadata and tags are. |
| **Music masking** | Loud tonal audio (RMS at or above -30 dBFS, spectral flatness at or below 0.15) raises the audio bars 25 percent, so a concert score stops flagging masking artifacts. #135, `tests/test_audio_music_gate.py`. | A raised bar can only suppress false positives; genuinely loud scenes with quiet cues stay at risk. |
| **Whispered / quiet explicit audio** | A sexual CLIP scene with a marginal audio score (between half the bar and the bar) drops the bars 20 percent. #135, `tests/test_audio_music_gate.py`. | Needs the scene signal; without it bars never drop. |
| **Non-standard skin tones at extremes** | The dark-frame normalization also helps very dark skin in extreme lighting; the eval pins the behavior. #129, `tests/test_preprocess.py`. | Color-based features at extremes remain less reliable. |
| **HDR/HLG content (was: colors shift)** | See the HDR10 row above; tags and metadata now survive. | HLG-specific tone mapping is not special-cased. |

## Accepted with reason

Not fixable in code, or correct-as-is. Each with the reason.

| Case | Reason |
|------|--------|
| **DRM content** | Protected streams cannot legally be decrypted by a third-party tool. PureFrame only works on unprotected files; this is the product's legal boundary, not a bug. |
| **Foreign-language audio** | PANNs hears sounds, not words. The audio gate (#135) adjusts thresholds but cannot make an English-trained classifier understand other languages. The nudity/scene detectors still cover visual content in any language. |
| **AVI/H.264 always full re-encode** | AVI carries no usable presentation timestamps, so smart-render copy cuts cannot land. The full re-encode is correct, just slower - by design, covered by tests. |
| **Breastfeeding scenes flag** | The detector cannot distinguish nursing from exposure at model level. Models are frozen by policy; the plan-whitelist workflow remains the answer. |
| **Body paint / already-censored content** | Same model-level limit: pixel patterns that resemble skin register on the detector. Review-first workflow covers it. |
| **Sound effects (pain, exertion)** | PANNs scores similar spectra to explicit audio. The #135 gate never drops a bar without a sexual scene signal, so these stay suppressed only by threshold choice and the whitelist workflow. |
| **Subtitle burn-in shifts at segment boundaries** | Inherent to smart rendering around censored segments; external subtitle tracks are unaffected and remain the recommendation. |
| **Multi-audio channel-layout metadata** | All audio tracks are copied losslessly; layout metadata alone can be lost in the copy. Data-preserving by design. |
| **4K+ content, 2+ hour movies, CPU batch speed, low RAM** | Compute costs, not defects: detection runs at downscaled resolution, temp disk scales with file size, GPU acceleration is documented, `--profile cpu` exists for constrained machines. |
| **Anime fanservice with in-scene censoring** | Working as intended: steam/light-censored scenes are borderline by nature; `--content-type anime` plus the plan review workflow is the designed answer. |

## Reporting Issues

If you encounter a failure case not listed here:

1. Run `pureframe plan` on the content to get the detection output
2. Use `pureframe preview` to generate the contact sheet
3. Open an issue at [GitHub Issues](https://github.com/xenoaitham/PureFrame/issues) with:
   - Content type and genre
   - Strictness level and threshold used
   - Whether it was a false positive or false negative
   - Approximate timestamp in the video (no explicit screenshots please)
