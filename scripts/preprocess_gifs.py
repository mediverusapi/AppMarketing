from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np


def _read_gif_frames_with_durations(path: Path):
    import imageio

    reader = imageio.get_reader(str(path))
    meta = {}
    try:
        meta = reader.get_meta_data()
    except Exception:
        meta = {}

    frames: List["np.ndarray"] = []
    durations: List[float] = []

    index = 0
    for frame in reader:
        frames.append(frame)
        dur = None
        try:
            md = reader.get_meta_data(index=index)
            dur = md.get("duration")
        except Exception:
            pass
        if dur is None:
            dur = meta.get("duration") if isinstance(meta, dict) else None
        if dur is None:
            dur = 0.1
        if isinstance(dur, (int, float)) and dur > 5.0:
            dur = float(dur) / 1000.0
        durations.append(float(dur))
        index += 1

    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames, durations


def _remove_background_rgba(frames):
    from rembg import remove  # type: ignore
    import numpy as np

    out: List["np.ndarray"] = []
    for f in frames:
        if getattr(f, "ndim", 3) == 2:
            f = np.stack([f, f, f], axis=2)
        # Ensure RGB input to rembg
        if f.shape[2] == 4:
            rgb = f[:, :, :3]
        else:
            rgb = f[:, :, :3]
        rgba = remove(rgb)
        if rgba.shape[2] == 3:
            # No alpha returned; make fully opaque
            alpha = np.full(rgba.shape[:2], 255, dtype=np.uint8)
            rgba = np.dstack([rgba, alpha])
        out.append(rgba)
    return out


def _write_gif_rgba(path: Path, frames_rgba, durations):
    """Write RGBA frames using Pillow with disposal=2 to avoid stacking.

    - Uses save_all + append_images
    - Forces non-zero frame durations
    - Sets disposal=2 (restore to background) for all frames
    - Disables optimization to prevent unwanted frame differencing
    """
    from PIL import Image

    if not frames_rgba:
        raise ValueError("No frames to write")

    pil_frames = [Image.fromarray(arr, "RGBA") for arr in frames_rgba]
    durations_ms = [max(10, int(round(float(d) * 1000))) for d in durations]
    # Ensure durations length matches number of frames
    if len(durations_ms) != len(pil_frames):
        if len(durations_ms) == 1:
            durations_ms = durations_ms * len(pil_frames)
        else:
            durations_ms = [max(10, int(round(float(durations[0]) * 1000)))] * len(pil_frames)

    # Set disposal for each appended frame
    for frm in pil_frames:
        frm.info["disposal"] = 2

    first, rest = pil_frames[0], pil_frames[1:]
    first.save(
        str(path),
        format="GIF",
        save_all=True,
        append_images=rest,
        loop=0,
        duration=durations_ms,
        disposal=2,
        optimize=False,
    )


def preprocess_gif_inplace(path: Path) -> None:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    frames, durations = _read_gif_frames_with_durations(path)
    frames_rgba = _remove_background_rgba(frames)
    _write_gif_rgba(path, frames_rgba, durations)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess GIFs by removing background once (in-place).")
    parser.add_argument("gifs", nargs="+", help="Paths to GIF files to preprocess in-place")
    args = parser.parse_args(argv)

    for p in args.gifs:
        preprocess_gif_inplace(Path(p))
        print(f"Preprocessed {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


