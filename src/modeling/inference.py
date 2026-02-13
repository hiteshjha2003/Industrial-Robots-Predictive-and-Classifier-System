# src/modeling/inference.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class Predictor:
    """
    Real-time inference class for RobotGuard AI.
    Loads saved models and scalers, approximates features from raw inputs,
    and predicts RUL + failure mode.
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize predictor by loading all saved artifacts.
        """
        self.models_dir = Path(models_dir).resolve()
        self._load_artifacts()
        print("Predictor initialized successfully.")

    def _load_artifacts(self):
        """Load XGBoost models, scalers, and label encoder."""
        try:
            self.regressor = joblib.load(self.models_dir / "xgb_regressor.joblib")
            self.classifier = joblib.load(self.models_dir / "xgb_classifier.joblib")
            self.feature_scaler = joblib.load(self.models_dir / "feature_scaler.joblib")
            self.target_scaler = joblib.load(self.models_dir / "target_scaler.joblib")
            self.label_encoder = joblib.load(self.models_dir / "label_encoder.joblib")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model artifacts missing in {self.models_dir}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")

    def _approximate_features(self, raw_input: dict) -> np.ndarray:
        """
        Approximate features from single raw reading.
        Safe even if scaler has no feature_names_in_.
        """
        # Create dict with all expected features (default 0)
        features = {col: 0.0 for col in self.feature_scaler.get_feature_names_out() 
                    if hasattr(self.feature_scaler, 'get_feature_names_out')}

        # Fill in what we can from input
        sensor_map = {
            "vibration": ["vibration_mean_300", "vibration_min_300", "vibration_max_300"],
            "torque": ["torque_mean_300", "torque_min_300", "torque_max_300"],
            "temperature": ["temperature_mean_300", "temperature_min_300", "temperature_max_300"],
        }

        for sensor, feature_list in sensor_map.items():
            if sensor in raw_input:
                val = raw_input[sensor]
                for f in feature_list:
                    if f in features:
                        features[f] = val

        # Gradients / FFT placeholders remain 0

        # Create DataFrame
        feature_df = pd.DataFrame([features])

        # If scaler has known columns → reorder
        if hasattr(self.feature_scaler, 'feature_names_in_'):
            expected = self.feature_scaler.feature_names_in_
            feature_df = feature_df.reindex(columns=expected, fill_value=0.0)
        else:
            # Fallback: assume current order is correct (risky but works if retrained)
            pass

        # Scale
        scaled = self.feature_scaler.transform(feature_df)
        return scaled

    def predict(self, raw_input: dict) -> dict:
        """
        Main prediction method.
        
        Input: dict with raw sensor values
        Output: dict with RUL and failure probabilities
        """
        # Approximate features from raw input
        X_scaled = self._approximate_features(raw_input)

        # Predict RUL
        rul_scaled = self.regressor.predict(X_scaled)[0]
        predicted_rul_hours = self.target_scaler.inverse_transform([[rul_scaled]])[0][0]

        # Predict failure mode probabilities
        proba = self.classifier.predict_proba(X_scaled)[0]
        classes = self.label_encoder.classes_

        failure_probabilities = dict(zip(classes, proba))

        # Most likely failure mode
        predicted_mode_idx = np.argmax(proba)
        predicted_mode = classes[predicted_mode_idx]
        predicted_mode_conf = proba[predicted_mode_idx]

        return {
            "predicted_rul_hours": round(float(predicted_rul_hours), 2),
            "predicted_failure_mode": predicted_mode,
            "failure_mode_confidence": round(float(predicted_mode_conf), 3),
            "failure_probabilities": {k: round(float(v), 3) for k, v in failure_probabilities.items()},
            "input_used": raw_input
        }