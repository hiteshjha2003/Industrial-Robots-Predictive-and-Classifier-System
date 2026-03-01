import sys
import numpy as np
import pandas as pd
from pathlib import Path
import click

# Add src to sys.path
from src.utils.helpers import load_config, find_project_root
from src.utils.progress import update_progress

PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

cfg = load_config()

def run_cleaning(mode="sample", input_dir="data/01_raw/", output_dir="data/02_intermediate/"):
    update_progress("clean", 5, "Loading raw data...")
    
    input_path = find_project_root() / input_dir / f"raw_{mode}.parquet"
    output_dir_path = find_project_root() / output_dir
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"ERROR: Input file not found at {input_path}")
        update_progress("clean", 0, "Error: Input file missing", status="failed")
        return
        
    df = pd.read_parquet(input_path)
    
    update_progress("clean", 40, "Cleaning and normalizing...")
    # Basic cleaning
    df = df.dropna()
    
    # Sort for time-series consistency
    df = df.sort_values(["robot_id", "joint_id", "timestamp"])
    
    update_progress("clean", 80, "Saving cleaned data...")
    out_file = output_dir_path / f"cleaned_{mode}.parquet"
    df.to_parquet(out_file, index=False)
    
    update_progress("clean", 100, "Done! Data cleaned and saved.")
    print(f"Cleaned data saved to {out_file}")

@click.command()
@click.option("--mode", default="sample", help="sample / medium / full")
def main(mode):
    run_cleaning(mode)

if __name__ == "__main__":
    main()