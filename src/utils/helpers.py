import os
from pathlib import Path
import yaml

def find_project_root():
    """Robust project root detection that avoids site-packages."""
    # 1. Check environment variable (highest priority, set by api.py)
    env_root = os.environ.get("ROBOT_PROJECT_ROOT")
    if env_root:
        return Path(env_root)
        
    # 2. Search upwards from this file
    current = Path(__file__).resolve()
    for _ in range(10):
        if (current / "config").exists() and (current / "src").exists():
            if "site-packages" not in str(current).lower():
                return current
        if current.parent == current:
            break
        current = current.parent
        
    # 3. Fallback to CWD if it looks right
    cwd = Path.cwd()
    if (cwd / "config").exists():
        return cwd
        
    # Final fallback (hardcoded as a last resort for this specific user's system)
    return Path(r"D:\COACHXLIVE\RESUME OF HITESH\Wifey Docs\projects\end_to_end_ml_pipeline\industrial-robot-predictive-mtce")

def load_config():
    root = find_project_root()
    path = root / "config" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)