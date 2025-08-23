from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
import os
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import traceback
from fastapi import HTTPException

from src.main import run_pipeline
from src.ffmpeg_utils import default_wav_path_for_video, extract_audio_to_wav
from src.diarize import Diarizer
from src.overlays import compute_overlay_events, render_video_with_overlays

# Load .env early so background workers inherit it
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(override=False)
except Exception:
    pass


app = FastAPI(title="AppMarketing Pipeline API", version="0.1.0")
# Prefer repo-local HF cache when present; allow override by process env
os.environ.setdefault("HF_HOME", str((Path("models")/".hf_cache").resolve()))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/env")
def env_status():  # pragma: no cover - diagnostic
    return {
        "hasHuggingFaceToken": bool(os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")),
        "hf_home": os.getenv("HF_HOME"),
        "offline": os.getenv("HF_HUB_OFFLINE") == "1",
    }


@app.get("/models")
def models_status():  # pragma: no cover - diagnostic
    base = (Path("models")/".hf_cache"/"hub").resolve()
    diar = (base/"models--pyannote--speaker-diarization-3.1").exists()
    seg = (base/"models--pyannote--segmentation-3.0").exists()
    return {"speaker_diarization_3_1": diar, "segmentation_3_0": seg, "path": str(base)}


@app.get("/api/logs/{job_id}")
def stream_logs(job_id: str):
    log_file = (Path("data") / "working" / "web" / job_id / "progress.log").resolve()
    if not log_file.exists():
        raise HTTPException(status_code=404, detail="No logs yet")
    with open(log_file, "r", encoding="utf-8") as f:
        return JSONResponse({"text": f.read()})


@app.post("/api/process")
async def process_video(
    video: UploadFile = File(...),
    overlayA: Optional[UploadFile] = File(None),
    overlayB: Optional[UploadFile] = File(None),
    position: str = Form("bottom-center"),
    overlayWidthRatio: float = Form(0.35),
    fps: Optional[int] = Form(None),
    codec: str = Form("h264_videotoolbox"),
    device: str = Form("auto"),
    hfToken: Optional[str] = Form(None),
    jobId: Optional[str] = Form(None),
):
    """Accept a video and optional GIFs, run pipeline, and stream back mp4."""
    try:
        work_root = (Path("data") / "working" / "web").resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        upload_id = jobId or uuid.uuid4().hex
        work_dir = (work_root / upload_id)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Save input video
        input_video_path = work_dir / "input.mp4"
        with open(input_video_path, "wb") as f:
            f.write(await video.read())

        # Resolve overlays: uploaded or fall back to defaults
        def save_upload(u: Optional[UploadFile], filename: str) -> Optional[Path]:
            if not u:
                return None
            dest = work_dir / filename
            with open(dest, "wb") as out:
                out.write(u.file.read())
            return dest

        overlay_a_path = save_upload(overlayA, "overlayA.gif")
        overlay_b_path = save_upload(overlayB, "overlayB.gif")

        if overlay_a_path is None:
            overlay_a_path = Path("assets/overlayGifs/sampleA.gif").resolve()
        if overlay_b_path is None:
            overlay_b_path = Path("assets/overlayGifs/sampleB.gif").resolve()

        output_dir = (Path("data") / "output").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (output_dir / f"{upload_id}.mp4").resolve()

        device_pref = None if device == "auto" else device
        # Resolve token from request or environment only (no hardcoded fallback)
        hfToken = hfToken or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

        # Progress logging
        log_path = work_dir / "progress.log"
        def log(msg: str) -> None:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(msg.rstrip() + "\n")

        # Run pipeline inline for better control/logging
        def run_inline() -> Path:
            log("[1/4] Extracting audio…")
            wav_path = default_wav_path_for_video(input_video_path, work_dir)
            extract_audio_to_wav(input_video_path, wav_path, sample_rate=16000, channels=1)
            log("[1/4] Audio extracted OK")

            log("[2/4] Running speaker diarization…")
            diarizer = Diarizer(hf_token=hfToken, device_preference=device_pref, num_speakers_hint=2)
            segments = diarizer.diarize(wav_path)
            spk_a, spk_b = diarizer.top_two_speakers(segments)
            log(f"Top speakers A={spk_a} B={spk_b}")

            log("[3/4] Preparing overlay events…")
            events = compute_overlay_events(segments, spk_a, spk_b)
            log(f"Prepared {len(events)} events")

            log("[4/4] Rendering output video…")
            final = render_video_with_overlays(
                input_video_path=input_video_path,
                output_video_path=output_path,
                overlay_a_path=overlay_a_path,
                overlay_b_path=overlay_b_path,
                events=events,
                position="bottom-center",
                overlay_width_ratio=0.6,
                fps=fps,
                codec=codec,
            )
            log("DONE")
            return final

        final_path = await asyncio.to_thread(run_inline)

        def iterfile(path: Path):
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk

        headers = {"Content-Disposition": f"inline; filename={final_path.name}"}
        return StreamingResponse(iterfile(final_path), media_type="video/mp4", headers=headers)

    except Exception as exc:  # pragma: no cover
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(exc)})


