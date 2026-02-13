🤖 RobotGuard AI: Revolutionizing Predictive Maintenance for Industrial Robots
RobotGuard AI Banner
(Imagine a futuristic robot arm glowing with AI insights – preventing breakdowns before they happen!)
Welcome to RobotGuard AI, an end-to-end machine learning pipeline that transforms raw sensor data into actionable predictions for six-axis industrial robots. Picture this: in a bustling car manufacturing line, 2,000 robots hum along, but unplanned downtime costs $80,000 per hour. RobotGuard AI steps in like a super-smart mechanic, predicting Remaining Useful Life (RUL) and failure modes to enable just-in-time maintenance – potentially saving millions annually!
This project isn't just code; it's a complete adventure in ML engineering. From generating synthetic telemetry data (because real datasets are rare and expensive) to deploying a sleek Streamlit dashboard, we've built a system that's robust, scalable, and ready for real-world action. Let's dive in!

Introduction
In the fast-paced world of manufacturing, robots are the unsung heroes – but they break down too. Traditional rule-based systems fall short with complex sensor patterns from vibration, torque, and temperature. Enter RobotGuard AI: a hybrid ML solution that uses XGBoost for regression (RUL in hours) and classification (failure modes like bearing wear or overheating).
Built as a portfolio showcase, this project demonstrates full-stack ML skills: data synth, ETL, feature engineering, modeling, inference, and an interactive UI. Whether you're a recruiter eyeing a demo or a fellow dev tweaking for your use case, RobotGuard AI is designed to impress and perform.
Key Wins from Our Journey:

Overcame memory errors and import issues to handle 5M+ row datasets.
Achieved ~70% R² for RUL prediction on synthetic data.
Balanced classification challenges with stratified splits and class weights.
Deployed via Docker for easy scaling.

Ready to explore? Let's gear up!

Project Overview
RobotGuard AI solves predictive maintenance by:

Generating high-fidelity synthetic data (multi-modal telemetry with degradation cycles).
Cleaning for quality (missing values, outliers).
Engineering 80+ features (rolling stats, FFT spectral analysis, gradients, cyclic encodings).
Training hybrid models: XGBoost regressor for RUL, classifier for 5 failure modes.
Inferring real-time predictions from raw sensors.
Visualizing in a dark, futuristic Streamlit dashboard with gauges and progress bars.

Business Impact (Hypothetical Story): In a car plant, RobotGuard spots a bearing wear 24 hours early – averting $1.92M in downtime. That's AI in action!
Project Highlights:

Scale-Ready: Handles 500K–10M+ rows on CPU.
Best Practices: Time-series aware splits, early stopping, memory optimization.
Creative Touches: Futuristic UI with animations, Plotly visuals, and emojis for fun.

We've iterated through bugs (MemoryErrors, import issues, class imbalances) to create a polished system. Now, it's your turn to run, tweak, or deploy!

Project Structure
The repository is organized for clarity and modularity. Here's a tree view (generated with tree -L 3 -I '__pycache__|.git|*.pyc|venv' for creativity):
textindustrial-robot-predictive-mtce/
├── app/                       # Streamlit dashboard
│   ├── main.py                # Core app logic & UI
│   └── style.css              # Futuristic dark theme CSS
├── config/                    # Configuration files
│   └── config.yaml            # Pipeline params (windows, paths, etc.)
├── data/                      # Data stages (ignored in git)
│   ├── 01_raw/                # Synthetic raw telemetry
│   ├── 02_intermediate/       # Cleaned data
│   └── 03_features/           # Engineered features
├── models/                    # Trained models & scalers (ignored in git)
├── src/                       # Core source code
│   ├── etl/                   # Data generation & cleaning
│   │   ├── synthetic_robot_data.py  # Data generator
│   │   └── clean.py           # Cleaning pipeline
│   ├── features/              # Feature engineering
│   │   └── build_features.py  # Rolling, FFT, etc.
│   ├── modeling/              # Training & inference
│   │   ├── train.py           # Model training
│   │   └── inference.py       # Real-time predictor
│   └── utils/                 # Helpers
│       └── helpers.py         # Config loader, etc.
├── tests/                     # Unit tests (add your own!)
├── .dockerignore              # Docker ignore rules
├── .gitignore                 # Git ignore rules
├── docker-compose.yml         # Multi-service Docker setup
├── Dockerfile                 # Docker image definition
├── Makefile                   # Common commands
├── pyproject.toml             # Project metadata & deps
├── README.md                  # This file!
└── requirements.txt           # Dependencies
Why this structure? It follows ML best practices: separate concerns (ETL, features, modeling), ignore large data/models, and use tools like Makefile for automation.

Tech Stack and Concepts
RobotGuard AI leverages a modern, efficient tech stack. Here's the creative breakdown – think of it as a robot's toolkit!
Core Technologies

Python 3.10+ – The brain: fast, readable code.
Streamlit – The face: interactive dashboard with gauges & charts.
Pandas/Numpy/PyArrow – Data wrangling: efficient loading & manipulation.
Scikit-learn – Preprocessing: scalers, encoders, metrics.
XGBoost – Models: gradient boosting for RUL regression & failure classification.
SciPy – Features: FFT for spectral analysis.
Plotly – Visuals: interactive gauges & bars.

Key ML Concepts Covered

Synthetic Data Generation: High-fidelity simulation of degradation cycles (e.g., vibration spikes for bearing wear).
Data Preparation: Forward-fill missing, Isolation Forest outliers, Parquet for efficiency.
Feature Engineering: 118+ features – rolling stats (mean/std/min/max), FFT (RMS/peak/entropy), gradients, cyclic encodings.
Modeling: Hybrid task – regression (MAE ~0.10h, R² ~0.71), classification (balanced with weights for imbalanced modes).
Deployment: Dockerfile for containerization, docker-compose for services, Makefile for automation.
Optimization: Memory downcasting (6.8GB → 1.5GB), early stopping, class weights.
Creative Touches: Dark UI with gradients, animations, responsive design.

We've tackled pitfalls like time-series leakage, class imbalance, and memory errors – making this a real-world-ready project.

Detailed Technical Guide to Run the Project (A Quick Start)
Let's get you up and running – think of this as your robot assembly manual! Assume you have Python 3.10+ installed.
Prerequisites

Python 3.10+ (use pyenv or conda).
Git for cloning.
Optional: Docker for containerized runs.


Clone the Repo:Bashgit clone <your-repo-url>
cd industrial-robot-predictive-mtce
Install Dependencies:Bashpip install -r requirements.txt
Configure (if needed):
Edit config/config.yaml for paths/windows/etc.


Running the Pipeline (Manual Mode)

Generate Data:Bashpython src/etl/synthetic_robot_data.py --mode sample
Clean Data:Bashpython src/etl/clean.py --mode sample
Build Features:Bashpython src/features/build_features.py
Train Models:Bashpython src/modeling/train.py --mode sample
Launch Dashboard:Bashstreamlit run app/main.py
Open http://localhost:8501
Follow sidebar steps for UI-driven pipeline.


Using Makefile (Automation Magic! 🎩)
With Makefile, one command does it all:
Bashmake all  # Installs, generates, cleans, features, trains, launches app
Other handy commands:
Bashmake data      # Just generate
make train     # Just train
make streamlit # Launch app
make clean     # Wipe data/models
Deployment Guide
Deploy like a pro – from local Docker to cloud!
Using Dockerfile

Build:Bashdocker build -t robotguard-ai .
Run:Bashdocker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models robotguard-ai
Access http://localhost:8501


Using docker-compose.yml

Start dashboard:Bashdocker-compose up -d web
Run training:Bashdocker-compose run train
Stop:Bashdocker-compose down

Using pyproject.toml (Advanced Dev)
Install as package:
Bashpip install -e ".[dev]"
Run scripts:
Bashgenerate-data --mode sample
train-models --mode sample
Cloud Deployment (e.g., Render.com or Streamlit Cloud)

Push to GitHub.
On Render: New Web Service → Docker → Auto-deploy.
Add volumes for /app/data and /app/models if persistent storage needed.


Dashboard Features
The Streamlit dashboard is your command center – dark, futuristic, and interactive!

Home: Overview with metrics & robot image.
Generate Data: Select mode, sliders for robots – progress bar & success animation.
Clean Data: One-click cleaning with spinner.
Feature Engineering: Builds features, shows sample DF & bar chart viz.
Train Models: Trains & loads metrics JSON for display.
Predict & Monitor: Input form for sensors, predicts with Plotly gauge (color zones) & probability bar chart. JSON output for details.

Pro Tip: Use session state – steps remember if completed (✅).

Expected Results
In sample mode (~5M rows):

Regression: MAE ~0.10h, RMSE ~0.17h, R² ~0.71 (good for synthetic).
Classification: Accuracy ~0.70+, Macro F1 ~0.65+ (after balancing – was 0.001 due to imbalance).
Dashboard: Real-time predictions (e.g., RUL 120h, bearing_wear 80% prob).
Runtime: Full pipeline ~10–15 min on CPU laptop.

If results differ: Check class balance (add prints), regenerate data.

Customization
Make it yours – fork & tweak!

Data: Add more failure modes in generator.
Features: Expand windows/FFT in build_features.py.
Models: Tune XGBoost params or try LightGBM.
UI: Add SHAP explanations or live charts in main.py.
Scale: For 10M+ rows, use Dask in loading.
Env: Add .env for secrets (e.g., API keys if added).

Contribute: PRs welcome!

License
MIT License – free to use, modify, and distribute. See LICENSE file for details.
(Project by Hitesh – Built with ❤️ and xAI inspiration. Questions? Open an issue!)