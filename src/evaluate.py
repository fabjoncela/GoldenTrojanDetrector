import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, THRESHOLD, WINDOW_SIZE
from .model import SiameseNet
from .preprocessing import windowize


def _load_scaler(path: str) -> StandardScaler:
    data = np.load(path)
    scaler = StandardScaler()
    scaler.mean_ = data["mean"]
    scaler.scale_ = data["scale"]
    scaler.var_ = data["var"]
    scaler.n_features_in_ = int(data["n_features"])
    return scaler


def _load_model(model_path: str, device: torch.device) -> SiameseNet:
    model = SiameseNet(FEATURES).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _prepare_windows(csv_path: str, scaler: StandardScaler, window_size: int) -> np.ndarray:
    df = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if df.ndim == 1:
        df = df.reshape(-1, FEATURES)
    arr = scaler.transform(df)
    windows = windowize(arr, window_size)
    if len(windows) == 0:
        raise ValueError(f"Not enough rows in {csv_path} to form a window of {window_size}")
    return windows


def anomaly_scores(model: SiameseNet, normal_ref: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        z_ref = model.encoder(normal_ref)
        z_s = model.encoder(sample)
        dist = torch.cdist(z_s, z_ref)  # [num_sample_win, num_ref_win]
        return dist.mean(dim=1)  # per-sample-window score


def main():
    parser = argparse.ArgumentParser(description="Score samples against a trained Siamese model.")
    parser.add_argument("--model", default="siamese_model.pt", help="Path to trained model file")
    parser.add_argument("--scaler", default="data/processed/scaler.npz", help="Path to saved scaler params")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="Window size used in preprocessing")
    parser.add_argument("--normal", required=True, help="CSV path for normal reference data")
    parser.add_argument("--sample", required=True, help="CSV path for sample to score")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Anomaly threshold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = _load_scaler(args.scaler)
    normal_windows = _prepare_windows(args.normal, scaler, args.window_size)
    sample_windows = _prepare_windows(args.sample, scaler, args.window_size)

    model = _load_model(args.model, device)

    normal_t = torch.tensor(normal_windows, dtype=torch.float32, device=device)
    sample_t = torch.tensor(sample_windows, dtype=torch.float32, device=device)

    scores = anomaly_scores(model, normal_t, sample_t)

    avg_score = scores.mean().item()
    max_score = scores.max().item()
    is_anomaly = avg_score > args.threshold

    result = {
        "avg_score": float(avg_score),
        "max_score": float(max_score),
        "threshold": args.threshold,
        "is_anomaly": bool(is_anomaly),
        "num_sample_windows": int(len(scores)),
        "num_normal_windows": int(len(normal_t)),
    }

    # Print a compact JSON-like output for easy frontend parsing
    import json

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()