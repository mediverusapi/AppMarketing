from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def _coerce_segment(segment: Union[object, Dict[str, float]]) -> Tuple[float, float, str]:
    """Return (start, end, speaker) from either a dataclass-like object or a dict."""
    if hasattr(segment, "start") and hasattr(segment, "end") and hasattr(segment, "speaker"):
        return float(segment.start), float(segment.end), str(segment.speaker)
    if isinstance(segment, dict):
        return float(segment["start"]), float(segment["end"]), str(segment["speaker"]) 
    raise TypeError("Unsupported segment type. Expected object with attributes or dict.")


def compute_overlay_events(
    segments: List[Union[object, Dict[str, float]]],
    speaker_a: Optional[str],
    speaker_b: Optional[str],
) -> List[Dict[str, Union[float, str]]]:
    """
    Convert diarization segments into overlay events. Events are active only during speech.

    Rules:
    - If current segment speaker == speaker_a, use overlay "A"; else use overlay "B".
    - Adjacent events with the same overlay are merged to reduce clip count.
    """
    if not segments:
        return []

    events: List[Dict[str, Union[float, str]]] = []
    for seg in segments:
        start, end, spk = _coerce_segment(seg)
        if end <= start:
            continue
        overlay = "A" if (speaker_a is not None and spk == speaker_a) else "B"
        events.append({"start": start, "end": end, "overlay": overlay})

    # Merge contiguous or near-contiguous events with same overlay
    events.sort(key=lambda e: (e["start"], e["end"]))
    merged: List[Dict[str, Union[float, str]]] = []
    for e in events:
        if not merged:
            merged.append(dict(e))
            continue
        prev = merged[-1]
        if e["overlay"] == prev["overlay"] and e["start"] <= float(prev["end"]) + 1e-3:
            prev["end"] = max(float(prev["end"]), float(e["end"]))
        else:
            merged.append(dict(e))

    return merged


def _resolve_position(
    video_w: int, video_h: int, overlay_w: int, overlay_h: int, position: str, margin_px: int
) -> Tuple[int, int]:
    position = (position or "top-right").lower()
    if position in {"bottom-center", "bottom_center", "bottom"}:
        return max(0, (video_w - overlay_w) // 2), max(0, video_h - overlay_h - margin_px)
    if position == "top-left":
        return margin_px, margin_px
    if position == "bottom-left":
        return margin_px, max(0, video_h - overlay_h - margin_px)
    if position == "bottom-right":
        return max(0, video_w - overlay_w - margin_px), max(0, video_h - overlay_h - margin_px)
    # default top-right
    return max(0, video_w - overlay_w - margin_px), margin_px


def _load_and_remove_gif_background(
    gif_path: Union[str, Path], target_width: int
):
    """
    Load a GIF, remove background per frame using rembg, and return a MoviePy
    ImageSequenceClip with an alpha mask.
    """
    remove = None
    try:
        from rembg import remove  # type: ignore
    except Exception:
        remove = None  # proceed without background removal

    try:
        import imageio
    except Exception as exc:  # pragma: no cover
        raise ImportError("imageio is required to read GIFs.") from exc

    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise ImportError("Pillow is required for image resizing.") from exc

    from moviepy.editor import ImageSequenceClip
    from moviepy.video.VideoClip import VideoClip
    import numpy as np

    reader = imageio.get_reader(str(Path(gif_path).expanduser().resolve()))
    meta = {}
    try:
        meta = reader.get_meta_data()
    except Exception:
        meta = {}

    frames_rgb: List["np.ndarray"] = []
    masks: List["np.ndarray"] = []
    durations: List[float] = []

    index = 0
    for frame in reader:
        # frame: HxWx3 (RGB) or HxWx4
        if getattr(frame, "ndim", 3) == 2:
            # grayscale → RGB
            frame = np.stack([frame, frame, frame], axis=2)
        if remove is not None:
            # Remove background -> returns RGBA
            rgba = remove(frame)
            if rgba.shape[2] == 3:
                alpha = np.full(rgba.shape[:2], 255, dtype=np.uint8)
                rgb = rgba
            else:
                rgb = rgba[:, :, :3]
                alpha = rgba[:, :, 3]
        else:
            # Fallback: use native alpha if present, otherwise keep opaque
            if frame.shape[2] == 4:
                rgb = frame[:, :, :3]
                alpha = frame[:, :, 3]
            else:
                rgb = frame[:, :, :3]
                alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)

        # Resize both rgb and alpha to target width
        h, w = rgb.shape[:2]
        if w != target_width:
            scale = float(target_width) / float(w)
            new_size = (target_width, max(1, int(round(h * scale))))
            rgb_img = Image.fromarray(rgb)
            alpha_img = Image.fromarray(alpha)
            rgb = np.array(rgb_img.resize(new_size, resample=Image.LANCZOS))
            alpha = np.array(alpha_img.resize(new_size, resample=Image.BILINEAR))

        frames_rgb.append(rgb)
        # Keep masks as 2D for MoviePy ImageSequenceClip with ismask=True
        masks.append(alpha.astype("float32") / 255.0)

        # Determine frame duration
        dur = None
        try:
            md = reader.get_meta_data(index=index)
            dur = md.get("duration")
        except Exception:
            pass
        if dur is None:
            dur = meta.get("duration") if isinstance(meta, dict) else None
        if dur is None:
            dur = 0.1  # default 100ms
        # Heuristic: if duration looks like milliseconds
        if isinstance(dur, (int, float)) and dur > 5.0:
            dur = float(dur) / 1000.0
        durations.append(float(dur))

        index += 1

    # Guard against empty frames or masks
    if not frames_rgb:
        raise ValueError("No frames decoded from GIF.")
    if not masks or any(m.ndim != 2 for m in masks):
        # Build opaque masks as fallback (2D for MoviePy)
        masks = [np.ones((fr.shape[0], fr.shape[1]), dtype="float32") for fr in frames_rgb]

    clip = ImageSequenceClip(frames_rgb, durations=durations)
    # Build a VideoClip mask to avoid ImageSequenceClip indexing assumptions (expects 3 channels)
    import numpy as _np
    cum = _np.cumsum([0.0] + durations).astype("float64")
    total = float(cum[-1]) if len(cum) else float(len(masks))
    def mask_make_frame(t: float):
        # Map time t to mask index using cumulative durations
        if t <= 0:
            idx = 0
        elif t >= total:
            idx = len(masks) - 1
        else:
            idx = int(_np.searchsorted(cum, t, side="right") - 1)
            if idx < 0:
                idx = 0
            if idx >= len(masks):
                idx = len(masks) - 1
        return masks[idx]
    mask_clip = VideoClip(make_frame=mask_make_frame, ismask=True, duration=clip.duration)
    clip = clip.set_mask(mask_clip)
    return clip


def render_video_with_overlays(
    input_video_path: Union[str, Path],
    output_video_path: Union[str, Path],
    overlay_a_path: Union[str, Path],
    overlay_b_path: Union[str, Path],
    events: List[Dict[str, Union[float, str]]],
    position: str = "bottom-center",
    overlay_width_ratio: float = 0.6,
    margin_px: int = 20,
    fps: Optional[int] = None,
    codec: str = "libx264",
) -> Path:
    """
    Render a new video with background-removed GIF overlays placed during segments.

    Args:
        input_video_path: Source video.
        output_video_path: Destination mp4 path.
        overlay_a_path: GIF used when speaker A is active.
        overlay_b_path: GIF used when speaker B is active.
        events: List of {start, end, overlay} dicts.
        position: One of top-left, top-right, bottom-left, bottom-right.
        overlay_width_ratio: Width of overlay relative to video width (0-1).
        margin_px: Margin from video edges in pixels.
        fps: Optional override FPS; defaults to source video fps or 24.
        codec: Video codec to use (e.g., libx264 or h264_videotoolbox on macOS).
    """
    from moviepy.editor import CompositeVideoClip, VideoFileClip, vfx

    in_path = Path(input_video_path).expanduser().resolve()
    out_path = Path(output_video_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ov_a = Path(overlay_a_path).expanduser().resolve()
    ov_b = Path(overlay_b_path).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input video not found: {in_path}")
    if not ov_a.exists() or not ov_b.exists():
        raise FileNotFoundError("Overlay GIFs not found. Ensure both A and B GIFs exist.")

    base_clip = VideoFileClip(str(in_path))
    composite = None
    a_anim = None
    b_anim = None
    try:
        video_w, video_h = base_clip.size
        target_w = max(1, int(video_w * float(overlay_width_ratio)))

        # Build background-removed animated overlays
        a_anim = _load_and_remove_gif_background(ov_a, target_w)
        b_anim = _load_and_remove_gif_background(ov_b, target_w)

        ax, ay = a_anim.size
        bx, by = b_anim.size
        pos_a = _resolve_position(video_w, video_h, ax, ay, position, margin_px)
        pos_b = _resolve_position(video_w, video_h, bx, by, position, margin_px)

        overlay_clips = []
        for e in events:
            start = float(e["start"])  # seconds
            end = float(e["end"])      # seconds
            if end <= start:
                continue
            duration = end - start
            which = str(e["overlay"]).upper()
            if which == "A":
                segment = a_anim.fx(vfx.loop, duration=duration).set_start(start).set_position(pos_a)
            else:
                segment = b_anim.fx(vfx.loop, duration=duration).set_start(start).set_position(pos_b)
            overlay_clips.append(segment)

        composite = CompositeVideoClip([base_clip, *overlay_clips])
        try:
            target_fps = int(fps or (base_clip.fps or 24))
        except Exception:
            target_fps = 24

        composite.write_videofile(
            str(out_path),
            fps=target_fps,
            codec=codec,
            audio_codec="aac",
            temp_audiofile=str(out_path.with_suffix(".temp-audio.m4a")),
            remove_temp=True,
            threads=4,
        )
    finally:
        # Best-effort cleanup of resources
        try:
            base_clip.close()
        except Exception:
            pass
        try:
            if a_anim is not None:
                a_anim.close()
        except Exception:
            pass
        try:
            if b_anim is not None:
                b_anim.close()
        except Exception:
            pass
        try:
            if composite is not None:
                composite.close()
        except Exception:
            pass

    return out_path


