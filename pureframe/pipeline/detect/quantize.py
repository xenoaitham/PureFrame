"""Dynamic int8 quantization of the NudeNet ONNX model (CPU profiles).

Dynamic int8 quantization typically speeds up CPU ONNX inference 2-4x at
near-identical accuracy for CNN detectors. The quantized model is derived
once from nudenet's bundled weights, cached in the user cache dir, and the
eval-parity CI gate proves detection behavior is unchanged before any PR
ships with it.
"""

import hashlib
import logging
from pathlib import Path

import platformdirs

logger = logging.getLogger(__name__)


def _nudenet_model_path() -> Path:
    """Path to the ONNX weights nudenet ships with."""
    import nudenet.nudenet as _n

    return (Path(_n.__file__).parent / "320n.onnx").resolve()


def quantized_model_path(
    source: Path | None = None, cache_root: Path | None = None
) -> Path:
    """Return the cached int8 model path, quantizing on first call.

    The cache key embeds a digest of the source weights so a nudenet upgrade
    invalidates stale quantized artifacts. Raises on quantization failure -
    callers are expected to fall back to the fp32 model.
    """
    src = source or _nudenet_model_path()
    cache_dir = Path(cache_root or platformdirs.user_cache_dir("PureFrame"))
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
    dst = cache_dir / f"nudenet_320n_int8_{digest}.onnx"
    if dst.exists():
        return dst

    from onnxruntime.quantization import QuantType, quantize_dynamic

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_suffix(".onnx.tmp")
    logger.info("Quantizing %s -> %s (one-time)", src.name, dst.name)
    quantize_dynamic(str(src), str(tmp_dst), weight_type=QuantType.QUInt8)
    tmp_dst.replace(dst)
    return dst
