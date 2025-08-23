from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


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
    import imageio

    # imageio will handle palette conversion; transparency becomes binary (as per GIF spec)
    imageio.mimsave(
        str(path),
        frames_rgba,
        duration=durations,
        loop=0,
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


