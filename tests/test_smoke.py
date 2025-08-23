from pathlib import Path

from src.ffmpeg_utils import default_wav_path_for_video
from src.overlays import compute_overlay_events
from src.diarize import SpeakerSegment


def test_imports_and_basic_flow():
    # Ensure util builds a path in working dir
    video = Path("data/input/example.mp4")
    wav_path = default_wav_path_for_video(video, "data/working")
    assert wav_path.name.endswith("_mono_16k.wav")

    # Prepare two simple segments for two speakers
    segments = [
        SpeakerSegment(start=0.0, end=1.0, speaker="SPEAKER_00"),
        SpeakerSegment(start=1.0, end=2.0, speaker="SPEAKER_01"),
    ]
    events = compute_overlay_events(segments, speaker_a="SPEAKER_00", speaker_b="SPEAKER_01")
    assert len(events) == 2
    assert events[0]["overlay"] == "A"
    assert events[1]["overlay"] == "B"


