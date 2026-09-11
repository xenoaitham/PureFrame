# Plan Sharing (Roadmap, v0.2+)

Censor plans are tiny JSON files with no copyrighted data - just timestamps, bounding box coordinates, and detection metadata. The community can share them safely.

## Why this matters

Running PureFrame on a 90-minute movie takes 12–55 minutes depending on hardware. Doing this work once and sharing the plan lets the community save thousands of hours collectively. A user with a low-end laptop or only a phone can apply a community plan to their own legal copy in under a minute.

## How it stays legal

PureFrame's plan files contain **no video data**. They contain only:
- Per-shot decisions (kiss, nudity, neutral, etc.)
- Bounding box coordinates and frame indices
- Confidence scores and reasoning text
- A hash of the source file so the plan applies cleanly

When User B downloads User A's plan and applies it to User B's own legal copy of the file, no copyrighted bytes ever moved between users. This fits the spirit of allowing families to filter content they possess, though it is not legal advice.

## Planned design

- A simple HTTP API + GitHub-hosted index of plan files keyed by file hash + media metadata.
- A `pureframe plan-fetch <title>` CLI command to look up community plans.
- Community moderation for plans (avoiding plans that *un*-censor or otherwise misuse the format).
- Multiple plans per title (strict / mild / standard) so users can pick their preferred filtering level.

This feature is **not** yet implemented. It is in the roadmap for v0.2.
