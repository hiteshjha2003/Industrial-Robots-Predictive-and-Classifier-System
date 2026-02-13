# Add your make commands here
# Makefile for industrial-robot-predictive-mtce
# Common commands for development, training and demo

# See all commands
# make help

# Run full pipeline
# make all

# Just generate data + train + run app
# make data train streamlit

# Clean everything
# make clean

.PHONY: help all install data clean-data features train streamlit docker-build docker-run clean

help:  ## Show this help message
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

all: install data clean-data features train streamlit  ## Run full pipeline (install → generate → clean → features → train → app)

install:  ## Install all dependencies
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	# If you use a separate CPU file
	# pip install -r requirements-cpu.txt

data:  ## Generate synthetic data (sample mode)
	@echo "Generating sample data..."
	python src/etl/synthetic_data_generator.py --mode sample

clean-data:  ## Clean generated raw data
	@echo "Cleaning data..."
	python src/etl/clean.py --mode sample

features:  ## Build engineered features
	@echo "Building features..."
	python src/features/build_features.py

train:  ## Train models
	@echo "Training models..."
	python src/modeling/train.py --mode sample

streamlit:  ## Launch Streamlit dashboard
	@echo "Starting RobotGuard AI dashboard..."
	streamlit run app/main.py

docker-build:  ## Build Docker image
	@echo "Building Docker image..."
	docker build -t robotguard-ai:latest .

docker-run:  ## Run Docker container (detached)
	@echo "Running Docker container..."
	docker run -d -p 8501:8501 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		--name robotguard robotguard-ai:latest

docker-stop:  ## Stop and remove running container
	@echo "Stopping Docker container..."
	docker stop robotguard || true
	docker rm robotguard || true

clean:  ## Clean up generated files (data + models + caches)
	@echo "Cleaning up..."
	rm -rf data/01_raw/* data/02_intermediate/* data/03_features/*
	rm -rf models/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.py[cod]" -delete
	rm -rf .pytest_cache .coverage htmlcov

lint:  ## Run linting (black + ruff if installed)
	black .
	ruff check --fix .

test:  ## Run tests (if you add pytest later)
	pytest

freeze:  ## Export current environment to requirements.txt
	pip freeze > requirements.txt

update:  ## Update all dependencies
	pip install --upgrade pip
	pip install -r requirements.txt --upgrade

docker-up:          ## Start dashboard in background
	docker compose up -d web

docker-down:        ## Stop and remove containers
	docker compose down

docker-logs:        ## Follow dashboard logs
	docker compose logs -f web

docker-train:       ## Run training once
	docker compose run --rm train

docker-rebuild:     ## Rebuild images without cache
	docker compose build --no-cache