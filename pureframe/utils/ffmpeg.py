import subprocess
import json
import logging
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Callable, Any
from pydantic import BaseModel
import ffmpeg
import numpy as np

from pureframe.hardware import HardwareProfile, ProfileSettings

logger = logging.getLogger(__name__)

class PureFrameError(Exception):
    pass

class VideoMetadata(BaseModel):
    width: int
    height: int
    fps: Fraction
    duration_seconds: float
    total_frames: int
    has_audio: bool
    audio_streams: list[dict]
    subtitle_streams: list[dict]
    container: str
    video_codec: str
    pixel_format: str
    color_space: str
    is_hdr: bool

def probe(path: Path) -> dict:
    try:
        probe_result = ffmpeg.probe(str(path))
        return probe_result
    except ffmpeg.Error as e:
        msg = f"FFprobe error on {path}"
        if e.stderr:
            msg += f": {e.stderr.decode()}"
        raise PureFrameError(msg)
    except Exception as e:
        raise PureFrameError(f"Failed to probe {path}: {e}")

def extract_metadata(probe_result: dict) -> VideoMetadata:
    format_info = probe_result.get('format', {})
    streams = probe_result.get('streams', [])
    
    video_stream = next((s for s in streams if s.get('codec_type') == 'video'), None)
    if not video_stream:
        raise PureFrameError("No video stream found.")
    
    audio_streams = [s for s in streams if s.get('codec_type') == 'audio']
    subtitle_streams = [s for s in streams if s.get('codec_type') == 'subtitle']
    
    # Parse FPS
    r_frame_rate = video_stream.get('r_frame_rate', '0/1')
    try:
        num, den = map(int, r_frame_rate.split('/'))
        fps = Fraction(num, den) if den != 0 else Fraction(0)
    except Exception:
        fps = Fraction(24, 1) # Fallback

    duration = float(format_info.get('duration', 0.0))
    if 'duration' in video_stream:
        duration = float(video_stream['duration'])
        
    width = int(video_stream.get('width', 0))
    height = int(video_stream.get('height', 0))
    
    total_frames = int(video_stream.get('nb_frames', 0))
    if total_frames == 0 and fps > 0:
        total_frames = int(duration * float(fps))
        
    color_space = video_stream.get('color_space', 'unknown')
    color_transfer = video_stream.get('color_transfer', 'unknown')
    is_hdr = 'smpte2084' in color_transfer or 'arib-std-b67' in color_transfer
    
    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        duration_seconds=duration,
        total_frames=total_frames,
        has_audio=len(audio_streams) > 0,
        audio_streams=audio_streams,
        subtitle_streams=subtitle_streams,
        container=format_info.get('format_name', 'unknown'),
        video_codec=video_stream.get('codec_name', 'unknown'),
        pixel_format=video_stream.get('pix_fmt', 'unknown'),
        color_space=color_space,
        is_hdr=is_hdr
    )

def select_hw_encoder(profile: HardwareProfile, codec: str) -> str:
    # We map 'h264' or 'hevc'
    base = 'libx264' if codec == 'h264' else 'libx265'
    if profile == HardwareProfile.CPU:
        return base
        
    # Attempt to read encoders
    try:
        output = subprocess.check_output(['ffmpeg', '-encoders'], stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return base
        
    encoders = [line.split()[1] for line in output.splitlines() if line.strip().startswith('V')]
    
    if codec == 'h264':
        if 'h264_nvenc' in encoders: return 'h264_nvenc'
        if 'h264_videotoolbox' in encoders: return 'h264_videotoolbox'
        if 'h264_qsv' in encoders: return 'h264_qsv'
        if 'h264_amf' in encoders: return 'h264_amf'
    elif codec == 'hevc' or codec == 'h265':
        if 'hevc_nvenc' in encoders: return 'hevc_nvenc'
        if 'hevc_videotoolbox' in encoders: return 'hevc_videotoolbox'
        if 'hevc_qsv' in encoders: return 'hevc_qsv'
        if 'hevc_amf' in encoders: return 'hevc_amf'
        
    return base

def frames_iter(path: Path, downscale_max_edge: int | None = None) -> Iterator[np.ndarray]:
    probe_res = probe(path)
    meta = extract_metadata(probe_res)
    
    stream = ffmpeg.input(str(path))
    if downscale_max_edge:
        # scale down
        w, h = meta.width, meta.height
        if w > h and w > downscale_max_edge:
            h = int(h * (downscale_max_edge / w))
            w = downscale_max_edge
        elif h > w and h > downscale_max_edge:
            w = int(w * (downscale_max_edge / h))
            h = downscale_max_edge
        # Ensure dimensions are even
        w = w - (w % 2)
        h = h - (h % 2)
        stream = ffmpeg.filter(stream, 'scale', w, h)
        out_w, out_h = w, h
    else:
        out_w, out_h = meta.width, meta.height
        
    process = (
        stream
        .output('pipe:', format='rawvideo', pix_fmt='bgr24')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    
    frame_size = out_w * out_h * 3
    try:
        while True:
            in_bytes = process.stdout.read(frame_size)
            if not in_bytes or len(in_bytes) != frame_size:
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape([out_h, out_w, 3])
            yield frame
    finally:
        process.stdout.close()
        process.wait()

def write_video_with_overlay(input_path: Path, output_path: Path, overlay_callback: Callable[[int, np.ndarray], np.ndarray], settings: ProfileSettings, encoder: str, crf: int) -> None:
    from pureframe.config import Config
    import tempfile
    import os
    
    meta = extract_metadata(probe(input_path))
    
    # temp file for video
    temp_fd, temp_video = tempfile.mkstemp(suffix=".mkv")
    os.close(temp_fd)
    
    logger.info(f"Writing temp video to {temp_video}")
    
    # 1. decode frames with -vsync 0 (which is -fps_mode passthrough in newer ffmpeg)
    # ffmpeg-python doesn't explicitly have -vsync 0 easily accessible on input, but we can pass it as kwargs
    in_kwargs = {'vsync': 0}
    
    process_in = (
        ffmpeg
        .input(str(input_path), **in_kwargs)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    
    # We don't know exact count from the pipe, we just read
    out_kwargs = {
        'vcodec': encoder,
        'crf': crf,
        'pix_fmt': 'yuv420p' # safe default
    }
    
    # Copy color space if any
    if meta.color_space != 'unknown':
        out_kwargs['colorspace'] = meta.color_space
        
    process_out = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f"{meta.width}x{meta.height}", r=float(meta.fps))
        .output(temp_video, **out_kwargs)
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=True)
    )
    
    frame_size = meta.width * meta.height * 3
    frame_idx = 0
    
    try:
        while True:
            in_bytes = process_in.stdout.read(frame_size)
            if not in_bytes or len(in_bytes) != frame_size:
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape([meta.height, meta.width, 3]).copy()
            
            # modify
            frame = overlay_callback(frame_idx, frame)
            
            process_out.stdin.write(frame.tobytes())
            frame_idx += 1
            if frame_idx % 1000 == 0:
                logger.debug(f"Rendered {frame_idx} frames...")
    finally:
        process_in.stdout.close()
        process_in.wait()
        process_out.stdin.close()
        process_out.wait()

    # Mux back
    logger.info("Muxing audio and subtitles...")
    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', str(input_path),
            '-map', '0:v:0',
            '-map', '1:a?',
            '-map', '1:s?',
            '-c', 'copy',
            str(output_path)
        ], check=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as sub_e:
        raise PureFrameError(f"Muxing failed: {sub_e}")
    finally:
        os.remove(temp_video)

