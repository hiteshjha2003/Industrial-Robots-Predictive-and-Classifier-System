# Multi-stage Dockerfile will go here
# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libatlas-base-dev gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy only installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure Python finds the packages
ENV PATH=/root/.local/bin:$PATH

# Copy app code
COPY . .

# Create dirs
RUN mkdir -p data/01_raw data/02_intermediate data/03_features models

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]