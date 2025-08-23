from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .ffmpeg_utils import default_wav_path_for_video, extract_audio_to_wav
from .diarize import Diarizer
from .overlays import compute_overlay_events, render_video_with_overlays


def run_pipeline(
    input_video_path: Path,
    overlay_a_path: Path,
    overlay_b_path: Path,
    working_dir: Path,
    output_video_path: Optional[Path] = None,
    position: str = "top-right",
    overlay_width_ratio: float = 0.2,
    fps: Optional[int] = None,
    codec: str = "libx264",
    hf_token: Optional[str] = None,
    device: Optional[str] = None,
) -> Path:
    load_dotenv(override=False)

    input_video_path = input_video_path.expanduser().resolve()
    overlay_a_path = overlay_a_path.expanduser().resolve()
    overlay_b_path = overlay_b_path.expanduser().resolve()
    working_dir = working_dir.expanduser().resolve()
    working_dir.mkdir(parents=True, exist_ok=True)

    if output_video_path is None:
        output_dir = (Path.cwd() / "data" / "output").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_video_path = (output_dir / f"{input_video_path.stem}_overlaid.mp4").resolve()
    else:
        output_video_path = output_video_path.expanduser().resolve()
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")
    if not overlay_a_path.exists():
        raise FileNotFoundError(f"Overlay A not found: {overlay_a_path}")
    if not overlay_b_path.exists():
        raise FileNotFoundError(f"Overlay B not found: {overlay_b_path}")

    wav_path = default_wav_path_for_video(input_video_path, working_dir)
    print(f"[1/4] Extracting audio to {wav_path}")
    extract_audio_to_wav(input_video_path, wav_path, sample_rate=16000, channels=1)

    print("[2/4] Running speaker diarization (this can take a while)...")
    diarizer = Diarizer(hf_token=hf_token, device_preference=device)
    segments = diarizer.diarize(wav_path)
    spk_a, spk_b = diarizer.top_two_speakers(segments)
    print(f"Detected top speakers: A={spk_a} B={spk_b}")

    print("[3/4] Preparing overlay events...")
    events = compute_overlay_events(segments, spk_a, spk_b)
    print(f"Prepared {len(events)} overlay events.")

    print(f"[4/4] Rendering output video to {output_video_path}")
    final_path = render_video_with_overlays(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        overlay_a_path=overlay_a_path,
        overlay_b_path=overlay_b_path,
        events=events,
        position=position,
        overlay_width_ratio=overlay_width_ratio,
        fps=fps,
        codec=codec,
    )
    print(f"Done: {final_path}")
    return final_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Speaker-aware overlay pipeline: diarize audio and overlay images per speaker.",
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file (e.g., data/input/sample.mp4)",
    )
    parser.add_argument(
        "--overlay-a",
        type=str,
        default="assets/overlayGifs/sampleA.gif",
        help="Path to Speaker A overlay GIF (background removed automatically)",
    )
    parser.add_argument(
        "--overlay-b",
        type=str,
        default="assets/overlayGifs/sampleB.gif",
        help="Path to Speaker B overlay GIF (background removed automatically)",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default="data/working",
        help="Directory for intermediate files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output mp4 (default: data/output/<stem>_overlaid.mp4)",
    )
    parser.add_argument(
        "--position",
        type=str,
        choices=["top-left", "top-right", "bottom-left", "bottom-right"],
        default="top-right",
        help="Overlay position",
    )
    parser.add_argument(
        "--overlay-width-ratio",
        type=float,
        default=0.2,
        help="Overlay width as fraction of video width (0-1)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Output frame rate (default: source fps or 24)",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="libx264",
        help="Video codec (e.g., libx264 or h264_videotoolbox for macOS HW encode)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token (falls back to HUGGINGFACE_TOKEN env)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
        help="Device preference for diarization (auto/cpu/cuda/mps)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_video = Path(args.video)
    overlay_a = Path(args.overlay_a)
    overlay_b = Path(args.overlay_b)
    working_dir = Path(args.working_dir)
    output = Path(args.output) if args.output else None
    device = None if args.device == "auto" else args.device

    try:
        run_pipeline(
            input_video_path=input_video,
            overlay_a_path=overlay_a,
            overlay_b_path=overlay_b,
            working_dir=working_dir,
            output_video_path=output,
            position=args.position,
            overlay_width_ratio=float(args.overlay_width_ratio),
            fps=args.fps,
            codec=args.codec,
            hf_token=args.hf_token,
            device=device,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


