# Show HN draft — for LO to review/post

**Title:** Show HN: PureFrame – Watch any movie family-friendly without cutting a frame

**Link:** https://github.com/xenoaitham/PureFrame

---

## First comment (post this as the author comment)

Hi HN! PureFrame is a local, open-source (MIT) tool that watches a movie with
your kids — sort of. It detects explicit visuals (nudity, sexual activity,
intense kissing) and overlays a localized, smoothly-tracked Gaussian blur on
just those regions. No scene skipping, no audio cuts, no re-timing: the film
plays exactly as the director cut it; you just don't see the parts you'd
rather not.

The existing options bugged me: VidAngel/ClearPlay need a subscription and
only support a curated catalog, and the DIY route means sitting with a video
editor cutting scenes by hand — which ruins pacing and chops dialog. I wanted
something that works on *any* file I legally possess — a foreign film, an old
DVD rip, an indie movie — fully offline.

How it works:

1. Scene detection splits the video into shots (PySceneDetect)
2. NudeNet samples frames and returns nudity bounding boxes
3. CLIP scores the scene context (is this a sexual situation?), and PANNs
   classifies the audio — but the audio model only runs on shots where the
   scene signal could change the verdict (the gate is pinned by parity tests
   that prove skipping it is verdict-identical)
4. Everything fuses into a `.censorplan.json` you can review and edit *before*
   anything renders — whitelist a false positive, then apply
5. The renderer re-encodes only the flagged segments and tracks the blur
   frame-by-frame

It never uploads anything: models download once (~400–500 MB) and after that
it's fully offline, zero telemetry. There's a CLI (`pip install pureframe`)
and an experimental Tauri desktop app with a plan-review UI.

Some numbers from the built-in benchmark (`pureframe bench`) on my RTX 3060 /
12-thread machine: a 30 s clip processes in 3.0 s on CPU, 16.2 s on MEDIUM,
23.7 s on HIGH (max-quality per-frame sampling). A recent optimization pass
(seek-based frame extraction, lazy audio classification, int8 CPU
quantization gated by an eval-parity CI job) targets ~10–20 min for a
90-minute movie on CPU-only hardware.

Honest limitations: false positives happen (swimwear, skin-tone backgrounds,
stylized animation), dark scenes reduce confidence, and it is *not* a
replacement for parental judgment — some things will slip through. There's a
full limitations doc in the repo.

Repo: https://github.com/xenoaitham/PureFrame

Happy to answer questions about the detection fusion, the render pipeline, or
why the plan-review step is non-negotiable for me.

---

## Notes for LO (delete before posting)

- Post timing: weekday morning US hours is the classic window; avoid posting
  the same day as the PyPI release announcement anywhere else.
- If asked "is this legal": the README/docs/legal.md position is private,
  local use on media you legally possess; no DRM circumvention; not legal
  advice. Don't improvise beyond that.
- If asked about piracy: it doesn't download, stream, or unlock anything.
- If someone asks for a real-movie demo GIF: we deliberately keep the shipped
  demo synthetic (copyright + taste); offer the synthetic regenerator script
  instead.
- Known sharp edges you might get called on: the GUI is experimental; macOS
  builds are unsigned; the bench numbers are synthetic clips, not movies
  (real-movie benchmarks are on the roadmap for v1.0).
