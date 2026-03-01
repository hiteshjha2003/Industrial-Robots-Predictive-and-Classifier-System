import numpy as np
import pandas as pd
from pathlib import Path
import click
from tqdm.auto import tqdm
import yaml
import os 
import sys 

# Add src to sys.path
from src.utils.helpers import load_config, find_project_root
from src.utils.progress import update_progress

PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

cfg = load_config()

def generate_one_robot_cycle(robot_id: int, joint_id: int, cycle_id: int, max_rul: float, failure_mode: str, fs: int = 100):
    duration_hours = np.random.uniform(0.6 * max_rul, 1.1 * max_rul)
    n_samples = int(duration_hours * 3600 * fs)
    
    # Base telemetry
    time = np.linspace(0, duration_hours, n_samples)
    
    # Degrade vibration as RUL approaches 0
    rul = duration_hours - time
    # Normalize RUL
    rul_norm = rul / duration_hours
    
    # Vibration increases exponentially near failure
    base_vib = 0.5 + 0.1 * np.random.randn(n_samples)
    vib_spike = (1 - rul_norm)**4 * 5.0
    vibration = base_vib + vib_spike
    
    # Temperature increases
    temperature = 40 + (1 - rul_norm) * 30 + np.random.randn(n_samples)
    
    # Torque behavior depends on failure mode
    if failure_mode == "friction_increase":
        torque = 10 + (1 - rul_norm) * 15 + np.random.randn(n_samples)
    elif failure_mode == "motor_efficiency_loss":
        torque = 10 - (1 - rul_norm) * 5 + np.random.randn(n_samples)
    else:
        torque = 10 + np.random.randn(n_samples)
        
    current = torque / 5.0 + np.random.uniform(0.1, 0.3, n_samples)
    joint_angle = 45 + 45 * np.sin(2 * np.pi * 0.1 * time)
    
    df = pd.DataFrame({
        "timestamp": pd.to_datetime('2024-01-01') + pd.to_timedelta(time, unit='h'),
        "vibration": vibration.astype('float32'),
        "torque": torque.astype('float32'),
        "temperature": temperature.astype('float32'),
        "current": current.astype('float32'),
        "joint_angle": joint_angle.astype('float32'),
        "RUL_hours": rul.astype('float32'),
        "robot_id": robot_id,
        "joint_id": joint_id,
        "cycle_id": cycle_id,
        "failure_mode": failure_mode
    })
    
    return df

def run_generator(mode="sample", output_dir="data/01_raw/"):
    update_progress("generate", 5, "Initializing generator...")
    
    # Ensure absolute path
    output_dir_path = find_project_root() / output_dir
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Config from mode
    robots = cfg["data"]["n_robots"] if mode == "full" else (6 if mode == "medium" else 2)
    modes = ["normal", "friction_increase", "motor_efficiency_loss", "sensor_drift"]
    
    all_data = []
    total_steps = robots * 6 # 6 joints per robot
    current_step = 0
    
    print(f"Generating data for {robots} robots in {mode} mode...")
    
    for rid in range(robots):
        for jid in range(6):
            current_step += 1
            progress = int((current_step / total_steps) * 90) + 5
            update_progress("generate", progress, f"Generating Robot {rid} Joint {jid}...")
            
            # Generate 3 cycles per joint
            for cid in range(3):
                fmode = np.random.choice(modes)
                max_rul = np.random.choice(cfg["data"]["degradation_period_hours"])
                df = generate_one_robot_cycle(rid, jid, cid, max_rul, fmode)
                all_data.append(df)
    
    update_progress("generate", 95, "Saving generated data...")
    final_df = pd.concat(all_data, ignore_index=True)
    out_file = output_dir_path / f"raw_{mode}.parquet"
    final_df.to_parquet(out_file, index=False)
    
    update_progress("generate", 100, "Done! Data saved to 01_raw")
    print(f"Saved {len(final_df)} rows to {out_file}")

@click.command()
@click.option("--mode", default="sample", help="sample / medium / full")
def main(mode):
    run_generator(mode)

if __name__ == "__main__":
    main()