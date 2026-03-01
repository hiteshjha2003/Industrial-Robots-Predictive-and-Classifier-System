Raw Telemetry (100Hz)
        ↓
Synthetic Generator
        ↓
Cleaning + Validation
        ↓
Feature Engineering (rolling + FFT + degradation)
        ↓
Time-based Split
        ↓
Scaling
        ↓
Train:
   - XGBoost Regressor (RUL)
   - XGBoost Classifier (Failure Mode)
        ↓
Save:
   - models
   - scalers
   - feature list
        ↓
Streamlit + FastAPI Inference



