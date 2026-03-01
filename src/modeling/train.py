# src/modeling/train.py

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
from pathlib import Path
import os
import sys
import time
import psutil
import click
# Add src to sys.path
from src.utils.helpers import load_config, find_project_root
from src.utils.progress import update_progress

PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

cfg = load_config()


def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2


def run_training(mode="sample"):
    print("Starting model training pipeline...")
    print(f"Mode              : {mode}")
    print(f"Project root      : {PROJECT_ROOT}")

    t_total = time.time()
    update_progress("train", 5, "Loading feature matrix...")

    # ── Paths ───────────────────────────────────────────────────────────────
    mode = cfg["data"]["mode"] # Overwrite mode from config
    features_path = PROJECT_ROOT / "data" / "03_features" / f"features_{mode}.parquet"
    artifacts_dir = PROJECT_ROOT / cfg["paths"]["model_dir"]
    artifacts_dir.mkdir(exist_ok=True) # Ensure artifacts directory exists

    print(f"Features file     : {features_path}")
    print(f"Artifacts dir     : {artifacts_dir}")
    print(f"Features exist?   : {features_path.exists()}")

    if not features_path.exists():
        print("ERROR: Features file not found. Run feature engineering first.")
        update_progress("train", 0, "Error: Features file not found", status="failed")
        return

    start_total = time.time()

    # ── Optimized data loading ──────────────────────────────────────────────
    load_start = time.time()

    print("Reading schema to select columns...")
    schema = pq.read_schema(features_path)
    all_cols = schema.names

    # Exclude columns not needed for modeling
    exclude = {"timestamp", "robot_id", "joint_id", "cycle_id"}
    keep_cols = [col for col in all_cols if col not in exclude]

    print(f"Loading only {len(keep_cols)} columns instead of {len(all_cols)}")

    df = pd.read_parquet(
        features_path,
        columns=keep_cols,
        engine="pyarrow",
        use_threads=False,
    )

    load_time = time.time() - load_start
    initial_mem = get_memory_usage_mb()
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns in {load_time:.2f} s")
    print(f"Initial memory usage: {initial_mem:.1f} MB\n")

    # ── Immediate type downcasting ─────────────────────────────────────────
    print("Downcasting numeric types for memory optimization...")

    # Float64 → float32 (biggest saving)
    float_cols = df.select_dtypes(include='float64').columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype('float32')

    # int64 → int32 or category
    int_cols = df.select_dtypes(include='int64').columns
    for col in int_cols:
        if df[col].nunique() < 2000:
            df[col] = df[col].astype('category')
        else:
            df[col] = df[col].astype('int32')

    # Force RUL_hours to float32 if present
    if "RUL_hours" in df.columns:
        df["RUL_hours"] = df["RUL_hours"].astype('float32')

    mem_after_downcast = get_memory_usage_mb()
    print(f"Memory after downcasting: {mem_after_downcast:.1f} MB")
    print(f"Memory saved: ~ {initial_mem - mem_after_downcast:.1f} MB\n")

    update_progress("train", 20, "Preprocessing and splitting data...")

    # ── Prepare data ────────────────────────────────────────────────────────
    # Drop IDs and timestamps that are NOT features
    drop_cols = ["failure_mode", "RUL_hours", "timestamp", "robot_id", "cycle_id", "joint_id"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # Identify numeric features for scaling
    numeric_features = [c for c in feature_cols if df[c].dtype != 'category' and not pd.api.types.is_object_dtype(df[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    X_numeric = df[numeric_features].astype('float32')
    
    # Simple encoding for categories if any exist (usually ID columns we already dropped)
    if categorical_features:
        X_cat = pd.get_dummies(df[categorical_features], drop_first=True)
        X = pd.concat([X_numeric, X_cat], axis=1)
    else:
        X = X_numeric

    y_reg = df["RUL_hours"].values
    y_cls = df["failure_mode"].values

    update_progress("train", 25, "Scaling features...")
    # Scale target (regression)
    y_scaler = StandardScaler()
    y_reg_scaled = y_scaler.fit_transform(y_reg.reshape(-1, 1)).ravel()

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    update_progress("train", 30, "Splitting data for training...")
    # Random stratified split
    X_train, X_temp, y_reg_train_s, y_reg_temp_s, y_cls_train, y_cls_temp = train_test_split(
        X_scaled, y_reg_scaled, y_cls, test_size=0.25, random_state=42, stratify=y_cls
    )
    X_val, X_test, y_reg_val_s, y_reg_test_s, y_cls_val, y_cls_test = train_test_split(
        X_temp, y_reg_temp_s, y_cls_temp, test_size=0.5, random_state=42, stratify=y_cls_temp
    )

    print(f"Train / Val / Test split: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}\n")

    update_progress("train", 40, "Training RUL Regressor (XGBoost)...")
    # ── Regression ──────────────────────────────────────────────────────────
    reg = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cpu",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30
    )

    reg.fit(
        X_train, y_reg_train_s,
        eval_set=[(X_val, y_reg_val_s)],
        verbose=False
    )

    update_progress("train", 70, "Training Failure Classifier...")
    # ── Classification ──────────────────────────────────────────────────────
    le = LabelEncoder()
    y_cls_train_enc = le.fit_transform(y_cls_train)
    y_cls_val_enc   = le.transform(y_cls_val)
    y_cls_test_enc  = le.transform(y_cls_test)

    clf = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        tree_method="hist",
        device="cpu",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20
    )

    clf.fit(
        X_train, y_cls_train_enc,
        eval_set=[(X_val, y_cls_val_enc)],
        verbose=False
    )

    update_progress("train", 90, "Evaluating and saving artifacts...")

    # Regression
    y_reg_pred_s = reg.predict(X_test)
    y_reg_pred   = y_scaler.inverse_transform(y_reg_pred_s.reshape(-1, 1)).ravel()
    y_reg_test   = y_scaler.inverse_transform(y_reg_test_s.reshape(-1, 1)).ravel()

    print("\nRegression Performance:")
    print(f"MAE   : {mean_absolute_error(y_reg_test, y_reg_pred):.2f} hours")
    print(f"RMSE  : {np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)):.2f} hours")
    print(f"R²    : {r2_score(y_reg_test, y_reg_pred):.3f}")

    # Classification
    y_cls_pred = clf.predict(X_test)

    print("\nClassification Performance:")
    acc = accuracy_score(y_cls_test_enc, y_cls_pred)
    f1_macro = f1_score(y_cls_test_enc, y_cls_pred, average='macro', zero_division=0)
    print(f"Accuracy  : {acc:.3f}")
    print(f"Macro F1  : {f1_macro:.3f}")

    # Safe classification report
    print("\nClassification Report:")
    unique_test_classes = np.unique(y_cls_test_enc)

    if len(unique_test_classes) == 0:
        print("No samples in test set for classification.")
    elif len(unique_test_classes) == 1:
        print("Only one class present in test set → cannot compute full report.")
        print(f"Single class: {le.inverse_transform(unique_test_classes)[0]}")
    else:
        # Map back to original names only for classes actually present
        present_names = [le.classes_[i] for i in unique_test_classes]
        print(classification_report(
            y_cls_test_enc,
            y_cls_pred,
            labels=unique_test_classes,
            target_names=present_names,
            zero_division=0
        ))

    # ── Save artifacts ──────────────────────────────────────────────────────
    print("\nSaving models and scalers...")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(reg,       artifacts_dir / "xgb_regressor.joblib")
    joblib.dump(clf,       artifacts_dir / "xgb_classifier.joblib")
    joblib.dump(scaler,    artifacts_dir / "feature_scaler.joblib")
    joblib.dump(y_scaler,  artifacts_dir / "target_scaler.joblib")
    joblib.dump(le,        artifacts_dir / "label_encoder.joblib")

    total_time = time.time() - start_total
    print(f"\nTraining pipeline completed in {total_time/60:.1f} minutes")
    print(f"Artifacts saved to: {artifacts_dir}")


@click.command()
@click.option("--mode", default="sample", help="Dataset mode: sample / medium / full")
def main(mode):
    run_training(mode)


if __name__ == "__main__":
    main()