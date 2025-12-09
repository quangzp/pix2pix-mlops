# ===============================
# MLOps Makefile – Sketch to Image
# ===============================

PYTHON := python
CONFIG_DIR := mlops/config

# -------- BASIC TASKS --------

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make env        - Create virtual environment"
	@echo "  make install    - Install dependencies"
	@echo "  make data       - Run data processing"
	@echo "  make features   - Run feature generation"
	@echo "  make train      - Train model"
	@echo "  make predict    - Run inference"
	@echo "  make clean      - Clean cache/build files"


# -------- ENV SETUP --------

env:
	python -m venv .venv
	@echo "Virtual environment created."

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt


# -------- DATA PIPELINE --------

data:
	$(PYTHON) mlops/dataset.py --config $(CONFIG_DIR)/dataset.yaml

features:
	$(PYTHON) mlops/features.py --config $(CONFIG_DIR)/features.yaml


# -------- MODELING --------

train:
	$(PYTHON) mlops/modeling/train.py --config $(CONFIG_DIR)/train.yaml

predict:
	$(PYTHON) mlops/modeling/predict.py --config $(CONFIG_DIR)/predict.yaml


# -------- CLEAN --------

clean:
	rm -rf __pycache__ */__pycache__
	find . -name "*.pyc" -delete
