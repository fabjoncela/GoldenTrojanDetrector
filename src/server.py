import json
import tempfile
import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional

from .config import THRESHOLD, WINDOW_SIZE
from .evaluate import _load_model, _load_scaler, _prepare_windows, anomaly_scores
from .pipeline import run_pipeline
from .visualize import generate_all_plots

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
    normal: Optional[UploadFile] = File(None, description="CSV of normal reference data"),
    sample: Optional[UploadFile] = File(None, description="CSV of sample to score"),
    threshold: float = Form(THRESHOLD),
    model_path: str = Form("siamese_model.pt"),
    scaler_path: str = Form("data/processed/scaler.npz"),
    window_size: int = Form(WINDOW_SIZE),
    normal_path: str = Form("data/raw/normal/run_001.csv"),
    sample_path: str = Form("data/raw/trojan/triggered/run_001.csv"),
    use_default: bool = Form(False),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _ensure_artifacts(model_path, scaler_path, device)
    scaler = ARTIFACTS.scaler
    model = ARTIFACTS.model

    # Determine input sources: uploaded files or default paths
    if not use_default and (normal is None or sample is None):
        raise HTTPException(status_code=400, detail="Please upload normal and sample CSVs or set use_default=true")

    temp_paths = []
    try:
        if normal is not None and not use_default:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f_norm:
                content = await normal.read()
                f_norm.write(content)
                normal_path = f_norm.name
                temp_paths.append(normal_path)

        if sample is not None and not use_default:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f_samp:
                content = await sample.read()
                f_samp.write(content)
                sample_path = f_samp.name
                temp_paths.append(sample_path)

        normal_windows = _prepare_windows(normal_path, scaler, window_size)
        sample_windows = _prepare_windows(sample_path, scaler, window_size)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        for p in temp_paths:
            Path(p).unlink(missing_ok=True)

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


@app.post("/visualize")
def create_visualizations():
    """Generate all visualization plots"""
    try:
        result = generate_all_plots()
        return {"status": "success", "message": "Plots generated successfully", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate plots: {str(e)}")


@app.get("/plots/{plot_name}")
def get_plot(plot_name: str):
    """Serve a specific plot image"""
    plot_path = Path("plots") / plot_name
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail=f"Plot {plot_name} not found")
    return FileResponse(plot_path, media_type="image/png")


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


@app.get("/generate-sample-normal")
async def generate_sample_normal():
    """Generate a sample normal/reference CSV file for testing"""
    np.random.seed(42)
    num_rows = 500
    num_features = 3
    
    # Generate normal data with consistent patterns
    time = np.linspace(0, 10, num_rows)
    feature1 = np.sin(time) + np.random.normal(0, 0.1, num_rows)
    feature2 = np.cos(time * 1.5) + np.random.normal(0, 0.1, num_rows)
    feature3 = np.sin(time * 0.5) * 0.5 + np.random.normal(0, 0.05, num_rows)
    
    data = np.column_stack([feature1, feature2, feature3])
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
    
    # Convert to CSV in memory
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    
    return StreamingResponse(
        io.BytesIO(stream.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=normal_sample.csv"}
    )


@app.get("/generate-sample-trojan")
async def generate_sample_trojan():
    """Generate a sample trojan/anomaly CSV file for testing"""
    np.random.seed(123)
    num_rows = 500
    
    # Generate trojan data with anomalies
    time = np.linspace(0, 10, num_rows)
    feature1 = np.sin(time) + np.random.normal(0, 0.1, num_rows)
    feature2 = np.cos(time * 1.5) + np.random.normal(0, 0.1, num_rows)
    feature3 = np.sin(time * 0.5) * 0.5 + np.random.normal(0, 0.05, num_rows)
    
    # Inject anomalies (spikes and pattern changes)
    anomaly_indices = [100, 150, 200, 250, 300, 350]
    for idx in anomaly_indices:
        if idx < num_rows:
            feature1[idx:idx+10] += np.random.uniform(2, 4)
            feature2[idx:idx+10] *= np.random.uniform(1.5, 2.5)
            feature3[idx:idx+10] += np.random.uniform(-1, -0.5)
    
    data = np.column_stack([feature1, feature2, feature3])
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
    
    # Convert to CSV in memory
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    
    return StreamingResponse(
        io.BytesIO(stream.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=trojan_sample.csv"}
    )


# For local testing: uvicorn src.server:app --reload --port 8000
