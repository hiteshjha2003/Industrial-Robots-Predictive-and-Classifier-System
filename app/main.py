# app/main.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path
import os
import sys
import time
from datetime import datetime

# Find project root
def find_root():
    # Try looking for markers from the file location
    current = Path(__file__).resolve().parent
    for _ in range(5):  # go up at most 5 levels
        if (current / "pyproject.toml").exists() or (current / "src").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    
    # Fallback: check current working directory
    cwd = Path.cwd()
    if (cwd / "src").exists() or (cwd / "pyproject.toml").exists():
        return cwd
    
    return None

PROJECT_ROOT = find_root()
if PROJECT_ROOT:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))  # highest priority
else:
    print("Warning: Project root not explicitly found.")

print("Detected project root:", PROJECT_ROOT)
print("Updated sys.path:", sys.path[:5])  # debug

from src.etl.synthetic_data_generator import main as generate_data
from src.etl.clean import main as clean_data
from src.features.build_features import main as build_features
from src.modeling.train import main as train_models
from src.modeling.inference import Predictor



# ── Paths (relative to project root) ────────────────────────────────────────
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "01_raw"
INTERMEDIATE_DIR = DATA_ROOT / "02_intermediate"
FEATURES_DIR = DATA_ROOT / "03_features"
MODELS_DIR = PROJECT_ROOT / "models"

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RobotGuard AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
with open(PROJECT_ROOT / "app" / "style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Session State Initialization ────────────────────────────────────────────
if "pipeline_status" not in st.session_state:
    st.session_state.pipeline_status = {
        "data_generated": False,
        "cleaned": False,
        "features_built": False,
        "trained": False,
        "last_prediction": None
    }

if "predictor" not in st.session_state:
    try:
        st.session_state.predictor = Predictor()
    except Exception as e:
        st.session_state.predictor = None
        st.warning(f"Predictor not loaded: {e}")

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 RobotGuard AI")
    st.markdown("**End-to-End Predictive Maintenance**")

    status = st.session_state.pipeline_status
    st.markdown("### Pipeline Status")
    st.markdown(f"Data Generated: {'✅' if status['data_generated'] else '⏳'}")
    st.markdown(f"Cleaned: {'✅' if status['cleaned'] else '⏳'}")
    st.markdown(f"Features: {'✅' if status['features_built'] else '⏳'}")
    st.markdown(f"Trained: {'✅' if status['trained'] else '⏳'}")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Generate Data", "🧹 Clean Data", "🔧 Feature Engineering", 
         "🧠 Train Models", "🚀 Predict & Monitor"]
    )

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🤖 RobotGuard AI")
st.markdown("**Prevent costly robot downtime with AI-powered Remaining Useful Life & Failure Mode prediction**")
st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── Page Routing ────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.header("Welcome to RobotGuard AI")
    st.markdown("""
    This is a complete **end-to-end predictive maintenance dashboard** for industrial six-axis robots.
    
    **Pipeline Steps:**
    1. Generate realistic synthetic telemetry data
    2. Clean & preprocess
    3. Engineer 80+ time-series & spectral features
    4. Train XGBoost regression + classification models
    5. Real-time prediction with beautiful gauges & alerts
    
    Use the sidebar to run each step sequentially.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dataset Size", "5.34M rows", delta="Sample mode")
    with col2:
        st.metric("Best R² (RUL)", "0.710", delta="Regression model")

    st.image("https://images.unsplash.com/photo-1581091226825-a6a2a5aee158", 
             caption="Industrial Robot Predictive Maintenance", width=1000)

elif page == "📊 Generate Data":
    st.header("Step 1: Generate Synthetic Telemetry Data")

    mode = st.selectbox("Dataset Mode", ["sample", "full"], index=0)
    n_robots = st.slider("Number of robots (sample mode)", 2, 20, 8)

    if st.button("🚀 Generate Data", type="primary"):
        with st.spinner("Generating synthetic robot data..."):
            st.session_state.pipeline_status["data_generated"] = False
            try:
                # Call your generator (adjust arguments if needed)
                generate_data(mode=mode)  # or your function signature
                st.success(f"Data generated successfully! Mode: {mode}")
                st.session_state.pipeline_status["data_generated"] = True
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")

elif page == "🧹 Clean Data":
    st.header("Step 2: Clean Raw Telemetry")

    if st.button("🧼 Run Cleaning Pipeline", type="primary"):
        with st.spinner("Cleaning data..."):
            st.session_state.pipeline_status["cleaned"] = False
            try:
                clean_data(mode="sample")  # adjust if needed
                st.success("Data cleaned successfully!")
                st.session_state.pipeline_status["cleaned"] = True
            except Exception as e:
                st.error(f"Cleaning failed: {str(e)}")

elif page == "🔧 Feature Engineering":
    st.header("Step 3: Advanced Feature Engineering")

    if st.button("⚙️ Build Features", type="primary"):
        with st.spinner("Computing 80+ features (rolling + FFT + gradients)..."):
            st.session_state.pipeline_status["features_built"] = False
            try:
                build_features()  # your function
                st.success("Features engineered successfully!")
                st.session_state.pipeline_status["features_built"] = True
            except Exception as e:
                st.error(f"Feature engineering failed: {str(e)}")

elif page == "🧠 Train Models":
    st.header("Step 4: Model Training")

    if st.button("🏋️ Train XGBoost Models", type="primary"):
        with st.spinner("Training regression + classification models (may take 5–10 min)..."):
            st.session_state.pipeline_status["trained"] = False
            try:
                train_models(mode="sample")
                st.success("Models trained and saved!")
                st.session_state.pipeline_status["trained"] = True
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

elif page == "🚀 Predict & Monitor":
    st.header("Real-time Robot Monitoring & Prediction")

    if st.session_state.predictor is None:
        st.error("Predictor not initialized. Check model files in /models/")
    else:
        st.subheader("Enter Latest Sensor Readings")

        col1, col2, col3 = st.columns(3)

        with col1:
            vibration = st.number_input("Vibration (RMS)", min_value=0.0, value=0.5, step=0.1)
            torque = st.number_input("Torque (Nm)", min_value=0.0, value=10.0, step=0.5)
        with col2:
            temperature = st.number_input("Temperature (°C)", min_value=20.0, value=45.0, step=1.0)
            current = st.number_input("Motor Current (A)", min_value=0.0, value=2.0, step=0.1)
        with col3:
            joint_angle = st.number_input("Joint Angle (deg)", min_value=-180.0, value=90.0, step=5.0)
            hours_since_service = st.number_input("Hours Since Last Service", min_value=0.0, value=100.0, step=10.0)

        if st.button("🔮 Predict RUL & Failure Mode", type="primary"):
            with st.spinner("Computing features & predicting..."):
                try:
                    # Prepare input as dict (adjust keys to match your predictor)
                    input_dict = {
                        "vibration": vibration,
                        "torque": torque,
                        "temperature": temperature,
                        "current": current,
                        "joint_angle": joint_angle,
                        "hours_since_service": hours_since_service,
                    }

                    pred = st.session_state.predictor.predict(input_dict)

                    # ── Results ─────────────────────────────────────────────────
                    st.subheader("Prediction Results")

                    col1, col2 = st.columns([3, 2])

                    with col1:
                        # RUL Gauge
                        rul = pred.get("predicted_rul_hours", 0)
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=rul,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Remaining Useful Life (hours)"},
                            delta={'reference': 100, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                            gauge={
                                'axis': {'range': [0, 400]},
                                'bar': {'color': "cyan"},
                                'steps': [
                                    {'range': [0, 50], 'color': "red"},
                                    {'range': [50, 150], 'color': "orange"},
                                    {'range': [150, 400], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': rul
                                }
                            }
                        ))
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, width='stretch')

                    with col2:
                        # Failure Mode Probabilities
                        if "failure_probabilities" in pred:
                            probs = pred["failure_probabilities"]
                            fig_prob = px.bar(
                                x=list(probs.keys()),
                                y=list(probs.values()),
                                labels={'x': 'Failure Mode', 'y': 'Probability'},
                                title="Failure Mode Probabilities",
                                color=list(probs.values()),
                                color_continuous_scale="RdYlGn_r"
                            )
                            fig_prob.update_layout(height=350)
                            st.plotly_chart(fig_prob, width='stretch')

                    st.json(pred)

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #888;'>
        RobotGuard AI • End-to-End Predictive Maintenance • Powered by xAI & Streamlit
        <br/>© {datetime.now().year} Hitesh • Last run: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
    """,
    unsafe_allow_html=True
)