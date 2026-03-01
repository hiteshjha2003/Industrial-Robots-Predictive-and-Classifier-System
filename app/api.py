# app/api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from fastapi.middleware.cors import CORSMiddleware

# Find project root
def find_root():
    # 1. Check environment variable
    env_root = os.environ.get("ROBOT_PROJECT_ROOT")
    if env_root:
        return Path(env_root)

    # 2. Try looking for markers from the file location
    current = Path(__file__).resolve().parent
    for _ in range(10):  # go up at most 10 levels
        if (current / "config").exists() and (current / "src").exists():
            if "site-packages" not in str(current).lower():
                return current
        if current.parent == current:
            break
        current = current.parent
    
    # 3. Fallback: check current working directory
    cwd = Path.cwd()
    if (cwd / "config").exists() and (cwd / "src").exists():
        return cwd
    
    # Final fallback: Hardcoded for this specific user's environment
    hardcoded = Path(r"D:\COACHXLIVE\RESUME OF HITESH\Wifey Docs\projects\end_to_end_ml_pipeline\industrial-robot-predictive-mtce")
    if hardcoded.exists():
        return hardcoded
        
    return None

PROJECT_ROOT = find_root()
if PROJECT_ROOT:
    print(f"Project root found at: {PROJECT_ROOT}")
    os.environ["ROBOT_PROJECT_ROOT"] = str(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
else:
    print("Warning: Project root not explicitly found. Assuming environment is correctly configured.")

from src.etl.synthetic_data_generator import run_generator
from src.etl.clean import run_cleaning
from src.features.build_features import run_feature_engineering
from src.modeling.train import run_training
from src.modeling.inference import Predictor
from src.utils.progress import get_progress, update_progress, reset_task

app = FastAPI(
    title="RobotGuard AI API",
    description="End-to-End Predictive Maintenance API for Industrial Robots",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor lazily
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        try:
            # Use absolute path to models directory
            models_path = PROJECT_ROOT / "models" if PROJECT_ROOT else Path("models")
            predictor = Predictor(models_dir=str(models_path))
        except Exception as e:
            print(f"Error loading predictor: {e}")
            return None
    return predictor

# --- Models ---

class PredictionInput(BaseModel):
    vibration: float = Field(..., json_schema_extra={"example": 0.5})
    torque: float = Field(..., json_schema_extra={"example": 10.0})
    temperature: float = Field(..., json_schema_extra={"example": 45.0})
    current: float = Field(..., json_schema_extra={"example": 2.0})
    joint_angle: float = Field(..., json_schema_extra={"example": 90.0})
    hours_since_service: float = Field(..., json_schema_extra={"example": 100.0})

class PredictionResponse(BaseModel):
    predicted_rul_hours: float
    failure_probabilities: Dict[str, float]
    status: str

class PipelineResponse(BaseModel):
    message: str
    status: str

# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "Welcome to RobotGuard AI API"}

@app.get("/health")
async def health():
    p = get_predictor()
    return {
        "status": "healthy",
        "predictor_loaded": p is not None
    }

def safe_run_task(task_func, task_id, *args, **kwargs):
    """Wrapper to run background tasks with error handling and progress updates."""
    try:
        task_func(*args, **kwargs)
    except Exception as e:
        print(f"Error in background task {task_id}: {e}")
        import traceback
        traceback.print_exc()
        update_progress(task_id, 0, f"Error: {str(e)}", status="failed")

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Returns the current progress of a specific background task."""
    return get_progress(task_id)

@app.post("/generate-data", response_model=PipelineResponse)
async def api_generate_data(background_tasks: BackgroundTasks, mode: str = "sample"):
    try:
        reset_task("generate")
        output_dir = str(PROJECT_ROOT / "data" / "01_raw") if PROJECT_ROOT else "data/01_raw"
        background_tasks.add_task(safe_run_task, run_generator, "generate", mode=mode, output_dir=output_dir)
        return {"message": f"Data generation started (mode: {mode})", "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clean-data", response_model=PipelineResponse)
async def api_clean_data(background_tasks: BackgroundTasks, mode: str = "sample"):
    try:
        reset_task("clean")
        input_dir = str(PROJECT_ROOT / "data" / "01_raw") if PROJECT_ROOT else "data/01_raw"
        output_dir = str(PROJECT_ROOT / "data" / "02_intermediate") if PROJECT_ROOT else "data/02_intermediate"
        background_tasks.add_task(safe_run_task, run_cleaning, "clean", mode=mode, input_dir=input_dir, output_dir=output_dir)
        return {"message": f"Data cleaning started (mode: {mode})", "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/build-features", response_model=PipelineResponse)
async def api_build_features(background_tasks: BackgroundTasks):
    try:
        reset_task("features")
        mode = "sample" # or get from session/config
        input_dir = str(PROJECT_ROOT / "data" / "02_intermediate") if PROJECT_ROOT else "data/02_intermediate"
        output_dir = str(PROJECT_ROOT / "data" / "03_features") if PROJECT_ROOT else "data/03_features"
        background_tasks.add_task(safe_run_task, run_feature_engineering, "features", mode=mode, input_dir=input_dir, output_dir=output_dir)
        return {"message": "Feature engineering started", "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", response_model=PipelineResponse)
async def api_train(background_tasks: BackgroundTasks, mode: str = "sample"):
    try:
        reset_task("train")
        background_tasks.add_task(safe_run_task, run_training, "train", mode=mode)
        return {"message": f"Model training started (mode: {mode})", "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def api_predict(data: PredictionInput):
    p = get_predictor()
    if p is None:
        raise HTTPException(status_code=503, detail="Predictor models not loaded or not found")
    
    try:
        input_dict = data.model_dump()
        prediction = p.predict(input_dict)
        return {
            "predicted_rul_hours": prediction.get("predicted_rul_hours", 0.0),
            "failure_probabilities": prediction.get("failure_probabilities", {}),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data-summary")
async def get_data_summary(mode: str = "sample"):
    """Returns a summary of the dataset for EDA."""
    try:
        path = PROJECT_ROOT / "data" / "03_features" / f"features_{mode}.parquet"
        if not path.exists():
            return {"status": "error", "message": "Features not found. Run buildup first."}
        
        df = pd.read_parquet(path)
        
        # Summary statistics
        summary = {
            "total_rows": len(df),
            "robot_count": int(df["robot_id"].nunique()),
            "failure_mode_dist": df["failure_mode"].value_counts().to_dict(),
            "avg_rul": float(df["RUL_hours"].mean()),
            "max_vibration": float(df["vibration"].max()),
            "max_torque": float(df["torque"].max())
        }
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/telemetry-sample")
async def get_telemetry_sample(robot_id: int = 0, joint_id: int = 0, mode: str = "sample"):
    """Returns a time-series sample for a specific robot/joint."""
    try:
        path = PROJECT_ROOT / "data" / "02_intermediate" / f"cleaned_{mode}.parquet"
        if not path.exists():
            return {"status": "error", "message": "Cleaned data not found."}
        
        df = pd.read_parquet(path)
        sample = df[(df["robot_id"] == robot_id) & (df["joint_id"] == joint_id)].tail(500)
        
        return {
            "timestamps": sample["timestamp"].astype(str).tolist(),
            "vibration": sample["vibration"].tolist(),
            "torque": sample["torque"].tolist(),
            "temperature": sample["temperature"].tolist(),
            "rul": sample["RUL_hours"].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
