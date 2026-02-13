import pandas as pd
import numpy as np
from pathlib import Path
import click

import os 
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



from utils.helpers import load_config

cfg = load_config()

@click.command()
@click.option("--mode", default="sample")
def main(mode):
    raw_path = Path(cfg["paths"]["raw_dir"]) / f"robots_{mode}.parquet"
    out_path = Path(cfg["paths"]["intermediate_dir"]) / f"cleaned_{mode}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    


    df = pd.read_parquet(raw_path)
    print(f"Raw data loaded → {len(df):,} rows")

    # Forward fill per robot-joint-cycle
    group_cols = ["robot_id", "joint_id", "cycle_id"]
    df = df.sort_values(["robot_id", "joint_id", "cycle_id", "timestamp"])
    df[["vibration", "torque", "temperature"]] = df.groupby(group_cols)[["vibration", "torque", "temperature"]].ffill()

    # Clip physically impossible values
    df["temperature"] = df["temperature"].clip(10, 150)
    df["vibration"] = df["vibration"].clip(0, 25)

    # Remove very short cycles (< 10 min)
    cycle_len = df.groupby(group_cols).size()
    valid_cycles = cycle_len[cycle_len >= 60000].index  # ~10 min @ 100 Hz
    df = df[df.set_index(group_cols).index.isin(valid_cycles)].reset_index(drop=True)

    df.to_parquet(out_path, index=False, compression="zstd")
    print(f"Cleaned data saved -> {len(df):,} rows")

if __name__ == "__main__":
    main()