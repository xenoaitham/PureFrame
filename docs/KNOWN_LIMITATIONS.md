# Known Limitations & Failure Cases

> PureFrame is honest about what it can and can't do. This document lists known edge cases where the tool may produce incorrect results.

## Detection Limitations

### False Positives (Flagged but not explicit)

| Category | Example | Why It Happens | Mitigation |
|----------|---------|---------------|------------|
| **Swimwear** | Beach/pool scenes | High skin exposure triggers skin-tone detector | Use `--strictness low` or whitelist via `plan-whitelist` |
| **Skin-colored clothing** | Nude-colored bodysuits, tan outfits | Color closely matches skin tones | Use `plan → preview → whitelist` workflow |
| **Classical art** | Renaissance paintings, sculptures | Nudity detection doesn't distinguish art from live nudity | Use `--strictness low` for art documentaries |
| **Medical content** | Surgical footage, anatomy lessons | Exposed skin triggers detection | Use `--content-type live-action --strictness low` |
| **Breastfeeding** | Nursing scenes | Exposed breast detected regardless of context | Whitelist manually after preview |
| **Male shirtless** | Gym, sports, beach | Large skin area at high strictness | Usually only triggers at `--strictness high` |
| **Mannequins/Statues** | Store displays, museums | Realistic body shapes trigger detection | Low confidence - usually filtered at medium strictness |

### False Negatives (Missed explicit content)

| Category | Example | Why It Happens | Mitigation |
|----------|---------|---------------|------------|
| **Very dark scenes** | Dimly lit bedroom, candlelight | Low contrast reduces detector confidence | Use `--content-type low-light` |
| **Flash frames** | Single-frame nudity (1/24s) | May fall between keyframe samples | Use `--strictness high` with denser sampling |
| **Extreme close-ups** | Partial body parts filling the frame | Detector trained on body-scale views | Limited mitigation - being improved |
| **Small/distant nudity** | Background nudity, crowd scenes | Small regions below detection threshold | Lower `--threshold` to 0.3 |
| **Non-standard skin tones** | Very dark or very pale skin in extreme lighting | Color-based features less reliable at extremes | Being improved in v0.2.0 |
| **Artistic nudity** | Black & white, heavy post-processing | Color distortion affects skin detection | Use `--strictness high` |
| **Screen-in-screen** | Explicit content shown on TV/phone within scene | Small region, low resolution | May need lower threshold |

### Animation-Specific

| Issue | Description | Mitigation |
|-------|-------------|------------|
| **Anime fanservice** | Ecchi content with steam/light censoring | Use `--content-type anime --strictness high` |
| **Transformation sequences** | Magical girl style - brief silhouette nudity | Usually flagged correctly at medium+ |
| **Chibi/super-deformed** | Non-realistic body proportions | Generally not flagged (correct behavior) |
| **Cel-shaded realism** | Photo-realistic anime/CG | Works well - similar to live-action detection |

## Rendering Limitations

| Issue | Description | Status |
|-------|-------------|--------|
| **HDR tone mapping** | HDR10/HLG content may have slightly different colors after re-encode | Preserved via codec passthrough when possible |
| **Subtitle burn-in** | Burned-in subtitles may shift slightly on segment boundaries | Use external subtitle tracks when possible |
| **Variable frame rate** | VFR content (screen recordings) may have timing issues | Convert to CFR first: `ffmpeg -i input.mp4 -vsync cfr output.mp4` |
| **Multi-audio tracks** | All audio tracks are preserved but may lose channel layout metadata | Lossless audio copy - metadata only |
| **DRM content** | Protected streams cannot be processed | PureFrame only works on unprotected files |
| **Containers & codecs** | Output keeps the input's container and video codec: MP4/MKV (H.264, HEVC), WebM (VP8, VP9) and AVI (MPEG-4, H.264) are covered by tests. Censored segments are re-encoded with the matching software encoder - VP9 re-encodes are CPU-bound and noticeably slower than H.264. | AVI/H.264 files carry no presentation timestamps, so they always take the full re-encode path (correct, just slower). AV1 and other codecs re-encode to H.264, which WebM rejects: transcode AV1 WebM first, e.g. `ffmpeg -i in.webm -c:v libvpx-vp9 -c:a copy in.vp9.webm` |

## Performance Limitations

| Scenario | Impact | Workaround |
|----------|--------|------------|
| **4K+ content** | Detection runs at downscaled resolution, render is full-res | Detection quality is the same; render time scales with resolution |
| **>2 hour movies** | May use significant temp disk space during smart rendering | Ensure 2x file size in free disk space |
| **Batch processing** | CPU-only mode on large folders is slow | Use GPU (`pip install torch --index-url https://download.pytorch.org/whl/cu121`) |
| **Low RAM (<8GB)** | Model loading may fail on constrained systems | Use `--profile cpu` to force lightweight mode |

## Audio Detection Limitations

| Issue | Description |
|-------|-------------|
| **Music masking** | Loud music may mask moaning/explicit audio cues |
| **Foreign languages** | Audio classifier trained primarily on English content |
| **Whispered audio** | Very quiet explicit audio may fall below detection threshold |
| **Sound effects** | Some non-explicit sounds (pain, exertion) may score similarly to explicit audio |

## Reporting Issues

If you encounter a failure case not listed here:

1. Run `pureframe plan` on the content to get the detection output
2. Use `pureframe preview` to generate the contact sheet
3. Open an issue at [GitHub Issues](https://github.com/xenoaitham/PureFrame/issues) with:
   - Content type and genre
   - Strictness level and threshold used
   - Whether it was a false positive or false negative
   - Approximate timestamp in the video (no explicit screenshots please)
