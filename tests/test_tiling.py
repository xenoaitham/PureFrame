"""Tiling utilities: grid geometry, NMS merge, native rescale."""

import numpy as np
import pytest

from pureframe.pipeline.detect.nudity import Detection
from pureframe.pipeline.detect.tiling import (
    map_tile_detections,
    nms_detections,
    tile_grid,
    tiled_detect_frame,
    tiled_detect_frames,
)


def _covers_frame(tiles, width, height) -> bool:
    """Every pixel of the frame is inside at least one tile."""
    grid = np.zeros((height, width), dtype=bool)
    for x, y, w, h in tiles:
        grid[y : y + h, x : x + w] = True
    return bool(grid.all())


class TestTileGrid:
    def test_two_by_two_covers_and_overlaps(self):
        tiles = tile_grid(320, 240, grid=(2, 2), overlap=0.15)
        assert len(tiles) == 4
        assert _covers_frame(tiles, 320, 240)
        # First tile anchored top-left, last anchored bottom-right.
        assert tiles[0][0] == 0 and tiles[0][1] == 0
        xs = sorted({t[0] for t in tiles})
        ys = sorted({t[1] for t in tiles})
        assert xs[1] + tiles[0][2] >= 320 - 1  # right column reaches the edge
        assert ys[1] + tiles[0][3] >= 240 - 1
        # Neighbors genuinely overlap (the small-object guarantee).
        assert xs[1] < tiles[0][2]
        assert ys[1] < tiles[0][3]

    def test_three_by_three_yields_nine_tiles(self):
        tiles = tile_grid(640, 480, grid=(3, 3), overlap=0.15)
        assert len(tiles) == 9
        assert _covers_frame(tiles, 640, 480)

    def test_single_cell_is_whole_frame(self):
        assert tile_grid(320, 240, grid=(1, 1), overlap=0.15) == [(0, 0, 320, 240)]

    def test_microscopic_frame_degenerates_to_whole_frame(self):
        # A cell of 1px inflated by overlap already covers the whole axis.
        assert tile_grid(2, 2, grid=(2, 2), overlap=0.15) == [(0, 0, 2, 2)]

    def test_microscopic_grid_still_covers(self):
        tiles = tile_grid(4, 4, grid=(2, 2), overlap=0.15)
        assert _covers_frame(tiles, 4, 4)

    def test_zero_overlap_gives_exact_halves(self):
        tiles = tile_grid(320, 240, grid=(2, 2), overlap=0.0)
        assert sorted({t[2] for t in tiles}) == [160]
        assert sorted({t[0] for t in tiles}) == [0, 160]

    def test_rejects_empty_frame(self):
        with pytest.raises(ValueError):
            tile_grid(0, 240)


class TestNMS:
    def test_same_label_high_iou_suppressed(self):
        a = Detection(label="X", score=0.9, box=(0, 0, 10, 10))
        b = Detection(label="X", score=0.8, box=(1, 1, 11, 11))
        kept = nms_detections([a, b], iou_threshold=0.5)
        assert len(kept) == 1 and kept[0].score == 0.9

    def test_different_labels_never_suppress_each_other(self):
        a = Detection(label="X", score=0.9, box=(0, 0, 10, 10))
        b = Detection(label="Y", score=0.8, box=(0, 0, 10, 10))
        assert len(nms_detections([a, b])) == 2

    def test_disjoint_boxes_both_kept(self):
        a = Detection(label="X", score=0.9, box=(0, 0, 10, 10))
        b = Detection(label="X", score=0.8, box=(100, 100, 110, 110))
        assert len(nms_detections([a, b])) == 2

    def test_chain_keeps_ends_and_drops_middle(self):
        # a overlaps b (IoU 0.67), c is disjoint from both: greedy NMS keeps
        # the highest scorer of each surviving overlap chain.
        a = Detection(label="X", score=0.95, box=(0, 0, 10, 10))
        b = Detection(label="X", score=0.90, box=(2, 0, 12, 10))
        c = Detection(label="X", score=0.85, box=(20, 0, 30, 10))
        kept = nms_detections([a, b, c], iou_threshold=0.5)
        assert {d.box for d in kept} == {a.box, c.box}

    def test_empty_input(self):
        assert nms_detections([]) == []


class TestMapTileDetections:
    def test_rescale_and_offset(self):
        tile = (10, 20, 100, 50)
        dets = [Detection(label="X", score=0.9, box=(0, 0, 20, 10))]
        mapped = map_tile_detections(
            dets, tile, scale=2.0, frame_width=320, frame_height=240
        )
        assert mapped[0].box == (10, 20, 20, 25)

    def test_clips_to_frame_and_drops_degenerate(self):
        tile = (300, 200, 100, 100)
        dets = [
            Detection(label="X", score=0.9, box=(0, 0, 400, 400)),
            Detection(label="X", score=0.5, box=(900, 900, 1000, 1000)),
        ]
        mapped = map_tile_detections(
            dets, tile, scale=1.0, frame_width=320, frame_height=240
        )
        assert len(mapped) == 1
        assert mapped[0].box == (300, 200, 320, 240)


class _WholeFrameDetector:
    """Fake detector: one whole-frame box per call, mimicking a tile-sized hit."""

    def __init__(self, score=0.9, label="X"):
        self.score = score
        self.label = label
        self.calls = 0

    def detect_batch(self, frames):
        self.calls += len(frames)
        out = []
        for f in frames:
            h, w = f.shape[:2]
            out.append(
                [Detection(label=self.label, score=self.score, box=(0, 0, w, h))]
            )
        return out


class TestTiledDetectFrame:
    def test_tiles_map_back_to_tile_rects(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detector = _WholeFrameDetector()
        dets = tiled_detect_frame(frame, detector, grid=(2, 2), overlap=0.15)
        tiles = tile_grid(320, 240, grid=(2, 2), overlap=0.15)
        # Each tile's whole-frame detection maps back to exactly that tile's
        # rect; neighboring tiles overlap by ~15% so NMS keeps all four.
        assert sorted(dets, key=lambda d: (d.box[0], d.box[1])) == sorted(
            [
                Detection(
                    label="X", score=0.9, box=(t[0], t[1], t[0] + t[2], t[1] + t[3])
                )
                for t in tiles
            ],
            key=lambda d: (d.box[0], d.box[1]),
        )

    def test_duplicate_hits_across_tiles_collapse(self):
        # A small object sitting fully inside the overlap zone of all four
        # tiles is reported by every tile; NMS must keep exactly one box.
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[104:134, 150:180] = 255

        class BlobDetector:
            def detect_batch(self, frames):
                out = []
                for f in frames:
                    mask = f[:, :, 0] > 10
                    if not mask.any():
                        out.append([])
                        continue
                    ys, xs = np.nonzero(mask)
                    out.append(
                        [
                            Detection(
                                label="X",
                                score=0.9,
                                box=(
                                    int(xs.min()),
                                    int(ys.min()),
                                    int(xs.max()) + 1,
                                    int(ys.max()) + 1,
                                ),
                            )
                        ]
                    )
                return out

        dets = tiled_detect_frame(frame, BlobDetector(), grid=(2, 2), overlap=0.15)
        assert len(dets) == 1
        x1, y1, x2, y2 = dets[0].box
        # Within upscaling/rounding slop of the true object rect.
        assert abs(x1 - 150) <= 3 and abs(y1 - 104) <= 3
        assert abs(x2 - 180) <= 3 and abs(y2 - 134) <= 3

    def test_frames_dict_maps_by_index(self):
        frames = {3: np.zeros((240, 320, 3), dtype=np.uint8)}
        out = tiled_detect_frames(frames, _WholeFrameDetector())
        assert set(out) == {3}
        assert out[3]
