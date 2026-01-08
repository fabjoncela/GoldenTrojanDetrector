import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from .config import WINDOW_SIZE




def load_csv_folder(folder):
    data = []
    for file in Path(folder).glob("*.csv"):
        try:
            df = pd.read_csv(file)
            if df.empty:
                print(f"Warning: {file} is empty, skipping...")
                continue
            data.append(df.values)
        except pd.errors.EmptyDataError:
            print(f"Warning: {file} has no data, skipping...")
            continue

    if not data:
        raise ValueError(f"No valid CSV files found in {folder}")

    return np.vstack(data)




def windowize(data, window):
    windows = []
    for i in range(0, len(data) - window, window):
        windows.append(data[i : i + window])
    return np.array(windows)




def preprocess(normal_path, trojan_path, out_path, scaler_out_path="data/processed/scaler.npz", window_size=WINDOW_SIZE):
    scaler = StandardScaler()

    normal = load_csv_folder(normal_path)
    trojan = load_csv_folder(trojan_path)

    all_data = np.vstack([normal, trojan])
    scaler.fit(all_data)

    normal = scaler.transform(normal)
    trojan = scaler.transform(trojan)

    normal_w = windowize(normal, window_size)
    trojan_w = windowize(trojan, window_size)

    np.savez(out_path, normal=normal_w, trojan=trojan_w)
    np.savez(
        scaler_out_path,
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=scaler.var_,
        n_features=scaler.n_features_in_,
        window_size=window_size,
    )