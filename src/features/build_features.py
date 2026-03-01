import sys
import numpy as np
import pandas as pd
from pathlib import Path
import click
import time
import os
import sys

# Add src to sys.path
from src.utils.helpers import load_config, find_project_root
from src.utils.progress import update_progress

PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

cfg = load_config()

def compute_rolling_stats(df, windows=[1, 3, 6]):
    """Computes rolling mean and std for vibration and torque."""
    for win in windows:
        for col in ["vibration", "torque"]:
            df[f"{col}_roll_mean_{win}"] = df.groupby(["robot_id", "joint_id"])[col].transform(
                lambda x: x.rolling(window=win, min_periods=1).mean()
            )
            df[f"{col}_roll_std_{win}"] = df.groupby(["robot_id", "joint_id"])[col].transform(
                lambda x: x.rolling(window=win, min_periods=1).std()
            ).fillna(0)
    return df

def compute_fft_features(df):
    """Placeholder for frequency-domain features (simplified for demo)."""
    df["vibration_fft_low"] = df["vibration"] * np.log1p(df["vibration"].abs())
    df["vibration_fft_high"] = df["vibration"]**2
    return df

def run_feature_engineering(task_id="features", mode="sample", input_dir="data/02_intermediate/", output_dir="data/03_features/"):
    update_progress(task_id, 10, "Loading intermediate data...")
    
    input_path = find_project_root() / input_dir / f"cleaned_{mode}.parquet"
    output_dir_path = find_project_root() / output_dir
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"ERROR: Input file missing: {input_path}")
        update_progress(task_id, 0, f"Error: Input file missing", status="failed")
        return
        
    df = pd.read_parquet(input_path)
    
    update_progress(task_id, 30, "Computing rolling statistics...")
    df = compute_rolling_stats(df, windows=cfg["features"]["windows_sec"])
    
    update_progress(task_id, 60, "Computing FFT features...")
    df = compute_fft_features(df)
    
    update_progress(task_id, 90, "Saving feature matrix...")
    out_file = output_dir_path / f"features_{mode}.parquet"
    df.to_parquet(out_file, index=False)
    
    update_progress(task_id, 100, f"Done! Features saved to {out_file}")
    print(f"Feature matrix saved to {out_file}")

@click.command()
@click.option("--mode", default="sample")
def main(mode):
    run_feature_engineering("features", mode)

if __name__ == "__main__":
    main()