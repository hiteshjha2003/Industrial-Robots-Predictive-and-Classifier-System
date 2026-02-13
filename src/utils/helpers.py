# src/utils/helpers.py
import yaml
from pathlib import Path

def load_config():
    path = "D:\COACHXLIVE\RESUME OF HITESH\Wifey Docs\projects\end_to_end_ml_pipeline\industrial-robot-predictive-mtce\config\config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)