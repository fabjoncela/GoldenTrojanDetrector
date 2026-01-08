import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional

from .config import THRESHOLD, WINDOW_SIZE
from .evaluate import _load_model, _load_scaler, _prepare_windows, anomaly_scores
from .pipeline import run_pipeline

app = FastAPI(title="Trojan Detector API", version="0.1.0")


# Simple cache to avoid reloading on every request
class _Artifacts:
    model: Optional[torch.nn.Module] = None
    scaler = None
    model_path: Optional[str] = None
    scaler_path: Optional[str] = None


ARTIFACTS = _Artifacts()


def _ensure_artifacts(model_path: str, scaler_path: str, device: torch.device):
    if ARTIFACTS.model is None or ARTIFACTS.model_path != model_path:
        if not Path(model_path).exists():
            raise HTTPException(status_code=400, detail=f"Model not found at {model_path}")
        ARTIFACTS.model = _load_model(model_path, device)
        ARTIFACTS.model_path = model_path

    if ARTIFACTS.scaler is None or ARTIFACTS.scaler_path != scaler_path:
        if not Path(scaler_path).exists():
            raise HTTPException(status_code=400, detail=f"Scaler not found at {scaler_path}")
        ARTIFACTS.scaler = _load_scaler(scaler_path)
        ARTIFACTS.scaler_path = scaler_path


@app.post("/score")
async def score(
    normal: UploadFile = File(..., description="CSV of normal reference data"),
    sample: UploadFile = File(..., description="CSV of sample to score"),
    threshold: float = Form(THRESHOLD),
    model_path: str = Form("siamese_model.pt"),
    scaler_path: str = Form("data/processed/scaler.npz"),
    window_size: int = Form(WINDOW_SIZE),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _ensure_artifacts(model_path, scaler_path, device)
    scaler = ARTIFACTS.scaler
    model = ARTIFACTS.model

    # Persist uploads to temporary files to reuse preprocessing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f_norm:
        content = await normal.read()
        f_norm.write(content)
        normal_path = f_norm.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f_samp:
        content = await sample.read()
        f_samp.write(content)
        sample_path = f_samp.name

    try:
        normal_windows = _prepare_windows(normal_path, scaler, window_size)
        sample_windows = _prepare_windows(sample_path, scaler, window_size)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(normal_path).unlink(missing_ok=True)
        Path(sample_path).unlink(missing_ok=True)

    normal_t = torch.tensor(normal_windows, dtype=torch.float32, device=device)
    sample_t = torch.tensor(sample_windows, dtype=torch.float32, device=device)

    scores = anomaly_scores(model, normal_t, sample_t)
    avg_score = scores.mean().item()
    max_score = scores.max().item()
    is_anomaly = avg_score > threshold

    return {
        "avg_score": float(avg_score),
        "max_score": float(max_score),
        "threshold": float(threshold),
        "is_anomaly": bool(is_anomaly),
        "num_sample_windows": int(len(scores)),
        "num_normal_windows": int(len(normal_t)),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def read_root():
    index_path = Path(__file__).parent.parent / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Open index.html to use the detector"}


@app.post("/train")
def train_pipeline(
    normal_path: str = Form("data/raw/normal"),
    trojan_path: str = Form("data/raw/trojan/triggered"),
    processed_path: str = Form("data/processed/data.npz"),
    scaler_path: str = Form("data/processed/scaler.npz"),
    window_size: int = Form(WINDOW_SIZE),
    epochs: Optional[int] = Form(None),
):
    # Run full pipeline (preprocess + pair gen + train) and return epoch losses
    result = run_pipeline(
        normal_path=normal_path,
        trojan_path=trojan_path,
        processed_path=processed_path,
        scaler_out_path=scaler_path,
        window_size=window_size,
        epochs=epochs,
    )

    # Invalidate cached artifacts so /score reloads the fresh model/scaler
    ARTIFACTS.model = None
    ARTIFACTS.scaler = None
    ARTIFACTS.model_path = None
    ARTIFACTS.scaler_path = None

    return result


# For local testing: uvicorn src.server:app --reload --port 8000
