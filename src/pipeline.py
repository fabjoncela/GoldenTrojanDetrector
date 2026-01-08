import numpy as np
from typing import Optional

from .config import WINDOW_SIZE, EPOCHS
from .pair_generator import make_pairs
from .preprocessing import preprocess
from .train import train


def run_pipeline(
    normal_path: str = "data/raw/normal",
    trojan_path: str = "data/raw/trojan/triggered",
    processed_path: str = "data/processed/data.npz",
    scaler_out_path: str = "data/processed/scaler.npz",
    window_size: int = WINDOW_SIZE,
    epochs: Optional[int] = None,
):
    preprocess(normal_path, trojan_path, processed_path, scaler_out_path, window_size)

    data = np.load(processed_path)
    p1, p2, y = make_pairs(data["normal"], data["trojan"])

    history = train(p1, p2, y, epochs=epochs or EPOCHS)
    return {
        "epochs": len(history),
        "losses": history,
        "processed_path": processed_path,
        "scaler_path": scaler_out_path,
        "window_size": window_size,
        "epochs_used": epochs or EPOCHS,
    }
