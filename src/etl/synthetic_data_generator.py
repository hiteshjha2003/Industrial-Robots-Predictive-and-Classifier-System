import numpy as np
import pandas as pd
from pathlib import Path
import click
from tqdm.auto import tqdm
import yaml
import os 
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.helpers import load_config



cfg = load_config()

def generate_one_robot_cycle(robot_id: int, joint_id: int, cycle_id: int, max_rul: float, failure_mode: str, fs: int = 100):
    duration_hours = np.random.uniform(0.6 * max_rul, 1.1 * max_rul)
    n_samples = int(duration_hours * 3600 * fs)
    t = np.linspace(0, duration_hours * 3600, n_samples)

    # Base signals
    vibration = 0.4 + 0.08 * np.sin(2 * np.pi * 12 * t / 60)          # bearing-like
    torque     = 1.0 + 0.15 * np.sin(2 * np.pi * 8 * t / 60)
    temp       = 45 + 8 * np.sin(2 * np.pi * 0.02 * t / 3600)

    rul = np.linspace(max_rul, 0, n_samples)

    # Failure mode specific degradation
    if failure_mode == "bearing_wear":
        vibration += 0.8 * (1 - rul / max_rul) ** 1.8
        vibration += 0.35 * np.random.normal(0, 0.07 * (1 - rul / max_rul), n_samples)
    elif failure_mode == "gear_tooth":
        torque += 1.1 * (1 - rul / max_rul) ** 2.2
        torque += 0.5 * np.random.normal(0, 0.12 * (1 - rul / max_rul), n_samples)
    elif failure_mode == "overheat":
        temp += 35 * (1 - rul / max_rul) ** 1.5
    elif failure_mode == "lubrication":
        vibration += 0.6 * (1 - rul / max_rul) ** 1.4
        torque += 0.4 * (1 - rul / max_rul) ** 1.6
    else:  # normal wear
        vibration += 0.25 * (1 - rul / max_rul) ** 2.5

    # Add sensor noise
    vibration += np.random.normal(0, 0.04 + 0.12 * (1 - rul / max_rul), n_samples)
    torque    += np.random.normal(0, 0.03 + 0.09 * (1 - rul / max_rul), n_samples)
    temp      += np.random.normal(0, 0.8  + 3.5  * (1 - rul / max_rul), n_samples)

    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=n_samples, freq=f"{1e9/fs:.0f}ns"),
        "robot_id": robot_id,
        "joint_id": joint_id,
        "cycle_id": cycle_id,
        "failure_mode": failure_mode,
        "RUL_hours": rul,
        "vibration": np.clip(vibration, 0, None),
        "torque": torque,
        "temperature": np.clip(temp, 20, 140),
    })

    return df


@click.command()
@click.option("--mode", default="sample", type=click.Choice(["sample", "medium", "full"]))
@click.option("--output_dir", default="data/01_raw/")
def main(mode, output_dir):
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    if mode == "sample":
        n_robots = 2
        n_cycles_per_robot_joint = 1
        failure_modes = ["normal", "bearing_wear", "gear_tooth", "overheat"]
        degradation_periods = [1.0, 1.5]
    else:
        n_robots = cfg["data"]["n_robots"]
        n_cycles_per_robot_joint = 5
        failure_modes = ["normal", "bearing_wear", "gear_tooth", "overheat", "lubrication"]
        degradation_periods = cfg["data"]["degradation_period_hours"]

    dfs = []

    for robot in tqdm(range(n_robots), desc="Robots"):
        for joint in range(6):
            for cycle in range(n_cycles_per_robot_joint):
                max_rul = np.random.choice(degradation_periods)
                fm = np.random.choice(failure_modes, p=[0.35, 0.25, 0.20, 0.15, 0.05] if len(failure_modes)==5 else [0.4,0.3,0.2,0.1])
                df_cycle = generate_one_robot_cycle(robot, joint, cycle, max_rul, fm)
                dfs.append(df_cycle)

    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(f"{output_dir}/robots_sample.parquet", index=False, compression="zstd")
    print(f"Saved {len(df):,} rows | {df['robot_id'].nunique()} robots | {df['failure_mode'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()