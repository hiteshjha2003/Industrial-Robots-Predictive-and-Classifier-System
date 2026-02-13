# src/features/build_features.py

import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path
import click
from tqdm.auto import tqdm
import time
import psutil
import os
import sys

# Add project root to sys.path (robust way)
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # go up 2 levels from src/features/
sys.path.append(str(PROJECT_ROOT))

from src.utils.helpers import load_config

cfg = load_config()

# ── Configurable parameters ───────────────────────────────────────────────
DEBUG_MODE = False              # Change to True only when debugging
MAX_GROUPS_TO_PROCESS = None    # None = process all groups

# Original window sizes — should give good features without skew
WINDOWS_SEC = [1, 3, 6, 10, 30]
FFT_LAST_SECONDS = 12
MIN_SAMPLES_FOR_FFT = 512
WELCH_NPERSEG = 1024            # fixed value — faster and more consistent

# Convert seconds → samples
WINDOWS_SAMPLES = [int(s * cfg["data"]["freq_hz"]) for s in WINDOWS_SEC]


def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2


def compute_rolling_stats(group, col, w_samples):
    """Fast rolling statistics — no skew"""
    minp = max(1, w_samples // 4)   # reasonable min_periods to avoid too many NaNs
    return pd.DataFrame({
        f"{col}_mean_{w_samples}":  group[col].rolling(w_samples, min_periods=minp).mean(),
        f"{col}_std_{w_samples}":   group[col].rolling(w_samples, min_periods=minp).std(),
        f"{col}_min_{w_samples}":   group[col].rolling(w_samples, min_periods=minp).min(),
        f"{col}_max_{w_samples}":   group[col].rolling(w_samples, min_periods=minp).max(),
        # skew intentionally removed — it was extremely slow and usually low value
    })


def compute_fft_features(group, col, fs=100):
    """FFT features — safe for small groups"""
    x = group[col].values[-int(FFT_LAST_SECONDS * fs):]
    if len(x) < MIN_SAMPLES_FOR_FFT:
        return pd.Series({
            f"{col}_fft_rms": 0.0,
            f"{col}_fft_peak_freq": 0.0,
            f"{col}_fft_entropy": 0.0
        })

    freqs, psd = signal.welch(x, fs=fs, nperseg=min(WELCH_NPERSEG, len(x)))
    if len(psd) == 0:
        return pd.Series({
            f"{col}_fft_rms": 0.0,
            f"{col}_fft_peak_freq": 0.0,
            f"{col}_fft_entropy": 0.0
        })

    rms = np.sqrt(np.mean(psd))
    peak_freq = freqs[np.argmax(psd)]
    psd_norm = psd / (psd.sum() + 1e-12)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    return pd.Series({
        f"{col}_fft_rms": rms,
        f"{col}_fft_peak_freq": peak_freq,
        f"{col}_fft_entropy": spectral_entropy
    })


@click.command()
@click.option('--debug/--no-debug', default=DEBUG_MODE, help="Enable detailed prints and memory tracking")
def main(debug):
    print("Starting full feature engineering pipeline...")
    print(f"Config mode       : {cfg['data']['mode']}")
    print(f"Windows (seconds) : {WINDOWS_SEC}")
    print(f"Debug prints      : {debug}")
    print(f"Process all groups: {MAX_GROUPS_TO_PROCESS is None}")

    in_path = PROJECT_ROOT / "data" / "02_intermediate" / f"cleaned_{cfg['data']['mode']}.parquet"
    out_path = PROJECT_ROOT / "data" / "03_features" / f"features_{cfg['data']['mode']}.parquet"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input file        : {in_path}")
    print(f"Output file       : {out_path}")
    print(f"Input exists?     : {in_path.exists()}")

    if not in_path.exists():
        print("ERROR: Cleaned parquet file not found. Run clean.py first.")
        return

    start_total = time.time()

    df = pd.read_parquet(in_path)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"Memory after load : {get_memory_usage_mb():.1f} MB\n")

    group_cols = ["robot_id", "joint_id", "cycle_id"]

    # Sort once (critical for correct rolling windows)
    df = df.sort_values(["robot_id", "joint_id", "cycle_id", "timestamp"])

    feature_dfs = []
    group_count = 0

    for name, group in tqdm(df.groupby(group_cols, sort=False), desc="Processing groups"):
        group_start = time.time()
        group_rows = len(group)
        group_count += 1

        if debug:
            print(f"\n── Group {group_count:3d} ── {name} ── {group_rows:,} rows ──")

        if group_rows < 200:
            if debug:
                print("  → Skipping (too few rows)")
            continue

        # Rolling features
        rolling_start = time.time()
        for col in ["vibration", "torque", "temperature"]:
            for w in WINDOWS_SAMPLES:
                stats_df = compute_rolling_stats(group, col, w)
                group = pd.concat([group, stats_df], axis=1)
        rolling_time = time.time() - rolling_start

        # FFT features
        fft_start = time.time()
        for col in ["vibration", "torque"]:
            fft_feats = compute_fft_features(group, col)
            for k, v in fft_feats.items():
                group[k] = v
        fft_time = time.time() - fft_start

        # Gradients (simple differences)
        group["vib_grad"]    = group["vibration"].diff().fillna(0)
        group["torque_grad"] = group["torque"].diff().fillna(0)
        group["temp_grad"]   = group["temperature"].diff().fillna(0)

        feature_dfs.append(group)

        group_time = time.time() - group_start
        mem = get_memory_usage_mb()

        if debug:
            print(f"  Rolling: {rolling_time:5.2f}s   |  FFT: {fft_time:5.2f}s")
            print(f"  Group time: {group_time:5.2f}s   |  Mem: {mem:.1f} MB")

        if MAX_GROUPS_TO_PROCESS is not None and group_count >= MAX_GROUPS_TO_PROCESS:
            print(f"\nEarly exit after {group_count} groups (debug limit)")
            break

    if not feature_dfs:
        print("No groups processed. Check data.")
        return

    print("\nConcatenating features...")
    concat_start = time.time()
    df_feat = pd.concat(feature_dfs, ignore_index=True)
    concat_time = time.time() - concat_start

    # Add cyclic hour encoding
    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["timestamp"].dt.hour / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["timestamp"].dt.hour / 24)

    print(f"Concat + cyclic encoding: {concat_time:.2f}s")
    print(f"Final shape             : {df_feat.shape[0]:,} rows × {df_feat.shape[1]} columns")
    print(f"Memory final            : {get_memory_usage_mb():.1f} MB")

    print("\nSaving features to disk...")
    save_start = time.time()
    df_feat.to_parquet(out_path, index=False, compression="zstd", engine="pyarrow")
    save_time = time.time() - save_start

    total_time = time.time() - start_total

    print(f"Saved in {save_time:.1f} seconds")
    print(f"\nFinished successfully!")
    print(f"Total runtime           : {total_time/60:.1f} minutes")
    print(f"Output file             : {out_path}")


if __name__ == "__main__":
    main()