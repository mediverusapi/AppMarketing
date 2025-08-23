from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Union


def check_ffmpeg_available() -> bool:
    """Return True if ffmpeg binary is available via PATH or imageio-ffmpeg."""
    if shutil.which("ffmpeg") is not None:
        return True
    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        return bool(exe and Path(exe).exists())
    except Exception:
        return False


def extract_audio_to_wav(
    input_video_path: Union[str, Path],
    output_audio_path: Union[str, Path],
    sample_rate: int = 16000,
    channels: int = 1,
    overwrite: bool = True,
) -> Path:
    """
    Extract a mono PCM WAV track from a video using ffmpeg.

    Args:
        input_video_path: Path to input video file.
        output_audio_path: Desired output WAV path.
        sample_rate: Audio sample rate (Hz), typically 16000 for diarization.
        channels: Number of audio channels. 1 for mono.
        overwrite: Whether to overwrite existing WAV.

    Returns:
        Path to the extracted WAV file.
    """
    input_path = Path(input_video_path).expanduser().resolve()
    output_path = Path(output_audio_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    # Resolve ffmpeg binary: PATH first, else imageio-ffmpeg bundled binary
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        try:
            import imageio_ffmpeg  # type: ignore

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "ffmpeg not found. Install via Homebrew (brew install ffmpeg) or pip install imageio-ffmpeg"
            ) from exc

    ff_overwrite_flag = "-y" if overwrite else "-n"
    cmd = [
        str(ffmpeg_bin),
        ff_overwrite_flag,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg timed out while extracting audio") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed extracting audio: {stderr[:1000]}") from exc

    return output_path


def default_wav_path_for_video(
    input_video_path: Union[str, Path], working_dir: Union[str, Path]
) -> Path:
    """Return a sane default WAV path inside the working directory for a given video."""
    video_path = Path(input_video_path)
    work = Path(working_dir)
    work.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    return (work / f"{stem}_mono_16k.wav").resolve()


