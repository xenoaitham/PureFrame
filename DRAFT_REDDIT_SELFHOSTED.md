# r/selfhosted draft — for LO to review/post

**Suggested title:** PureFrame — self-hosted, offline "family filter" for your own movie files. Local AI blurs explicit scenes instead of cutting them.

**Where:** r/selfhosted (also fits r/coolgithubprojects, r/opensource — slightly reword the hook per sub)

---

## Post body

I built a tool for a problem my household kept hitting: watching movies with
kids, without either (a) sitting with a remote ready to skip scenes or (b)
paying a subscription that only works on a curated catalog.

**PureFrame** (MIT, free) runs entirely on your own machine. You point it at a
video file, it detects explicit visuals with local AI models, and it applies a
localized blur over just those regions — the movie keeps playing normally:
same length, same audio, same pacing. Nothing is uploaded anywhere, ever.
After the one-time model download it works fully offline (air-gap it if you
want — it never phones home).

The part I care most about: **you review before anything renders.** It writes
a plain JSON plan of every flagged shot (category, confidence, bounding
boxes, reasoning). You can whitelist false positives in the desktop app or in
a text editor, then render. No surprise outputs.

- Works on: MP4 / MKV / AVI / WebM — any local file, any movie, not a curated list
- Platforms: Windows / macOS / Linux — prebuilt installers (Tauri GUI) or `pip install pureframe` (CLI)
- Detection: NudeNet (visual) + CLIP (scene context) + PANNs (audio), fused with per-category thresholds; strictness presets and content-type profiles (animation, anime, low-light)
- Output: re-encodes only flagged segments, so most of the file is untouched bit-for-bit
- Hardware: GPU recommended, CPU works (int8-quantized inference; a 30 s clip benchmarks at ~3 s on a mid CPU, and there's a built-in `pureframe bench` to test your own machine)

Source + docs: https://github.com/xenoaitham/PureFrame

Honest limits: it's computer vision, not magic — swimwear and skin-tone
backgrounds sometimes false-positive, dark scenes are harder, and some things
will slip through. It's a tool, not a babysitter.

Setup is intentionally boring: install, run `pureframe process movie.mp4`,
get `movie_clean.mp4`. Happy to answer questions.

---

## Notes for LO (delete before posting)

- r/selfhosted is touchy about "is this self-hosted?": it runs on your own
  hardware with no cloud dependency, which fits the sub's spirit; leading
  with "fully offline, no account, no telemetry" preempts that.
- Don't post the same text to multiple subs the same day (spam filters +
  etiquette); space them out.
- If someone asks about DRM/streaming rips: it doesn't and won't touch DRM;
  local files only.
- Screenshots: the README now has GUI screenshots you can link directly
  (raw GitHub asset URLs embed on Reddit).
