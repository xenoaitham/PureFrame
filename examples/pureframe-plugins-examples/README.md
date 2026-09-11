# pureframe-plugins-examples

First-party example detector plugins for PureFrame. Install one and it
shows up in `pureframe plugins list`; enable it per run with
`--enable-plugin`. See `docs/plugins.md` for the full user page and
`docs/plugin-api.md` for the developer contract.

## motionblob

Censors whatever moves. Frames are differenced against the previous
keyframe; changed regions above a minimum area become boxes in the
`MOTION_VISIBLE` category and render like any other blur box. No model,
no weights, no network.

```bash
pip install pureframe-plugins-examples
pureframe plugins list
pureframe process input.mp4 --enable-plugin motionblob
```

Tune it through the standard threshold controls:

```bash
pureframe process input.mp4 --enable-plugin motionblob --thresholds '{"MOTION_VISIBLE": 0.3}'
```

Use it as-is for privacy-first jobs (censor anything that moves in a
region), or copy `src/pureframe_plugins_examples/motionblob.py` as the
skeleton for a real detector.
