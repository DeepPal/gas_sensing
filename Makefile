# ============================================================
#  Au-MIP LSPR Gas Sensing Platform — Developer Makefile
#  Usage: make <target>
# ============================================================

PYTHON  := .venv/Scripts/python.exe
PIP     := .venv/Scripts/pip.exe
PYTEST  := $(PYTHON) -m pytest
RUFF    := $(PYTHON) -m ruff
MLFLOW  := $(PYTHON) -m mlflow

.DEFAULT_GOAL := help

# ── Help ─────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  Au-MIP LSPR Gas Sensing Platform"
	@echo ""
	@echo "  Setup"
	@echo "    make install        Install all dependencies into .venv"
	@echo "    make install-ml     Install + PyTorch (CNN support)"
	@echo ""
	@echo "  Quality"
	@echo "    make lint           Run ruff linter"
	@echo "    make format         Auto-format with ruff"
	@echo "    make test           Run full test suite"
	@echo "    make test-fast      Run tests, stop on first failure"
	@echo "    make coverage       Run tests with coverage report"
	@echo "    make check          lint + test (CI equivalent)"
	@echo ""
	@echo "  Run"
	@echo "    make dashboard      Launch Streamlit dashboard"
	@echo "    make serve          Launch FastAPI inference server"
	@echo "    make simulate       Run pipeline in simulation mode"
	@echo ""
	@echo "  MLflow"
	@echo "    make mlflow-ui      Open MLflow experiment tracking UI"
	@echo ""
	@echo "  Training"
	@echo "    make train-gpr      Train GPR calibration model"
	@echo "    make train-cnn      Train CNN gas classifier"
	@echo "    make cross-eval     Leave-one-gas-out cross-validation"
	@echo "    make ablation       Preprocessing ablation study"
	@echo ""
	@echo "  Maintenance"
	@echo "    make clean          Remove build artefacts and caches"
	@echo ""

# ── Setup ────────────────────────────────────────────────────
.PHONY: install
install:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

.PHONY: install-ml
install-ml: install
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ── Quality ──────────────────────────────────────────────────
.PHONY: lint
lint:
	$(RUFF) check src/

.PHONY: format
format:
	$(RUFF) format src/ tests/
	$(RUFF) check src/ --fix --unsafe-fixes

.PHONY: test
test:
	$(PYTEST) tests/ -q

.PHONY: test-fast
test-fast:
	$(PYTEST) tests/ -q -x --tb=short

.PHONY: coverage
coverage:
	$(PYTEST) tests/ --cov=src --cov-report=term-missing --cov-report=html:output/coverage -q
	@echo "HTML report: output/coverage/index.html"

.PHONY: check
check: lint test

# ── Run ──────────────────────────────────────────────────────
.PHONY: dashboard
dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py

.PHONY: serve
serve:
	$(PYTHON) serve.py

.PHONY: simulate
simulate:
	$(PYTHON) run.py --mode simulate --duration 30

# ── MLflow ───────────────────────────────────────────────────
.PHONY: mlflow-ui
mlflow-ui:
	$(MLFLOW) ui --backend-store-uri experiments/mlruns --port 5000
	@echo "MLflow UI: http://localhost:5000"

# ── Training ─────────────────────────────────────────────────
.PHONY: train-gpr
train-gpr:
	$(PYTHON) -m src.training.train_gpr --data Joy_Data/Ethanol

.PHONY: train-cnn
train-cnn:
	$(PYTHON) -m src.training.train_cnn --data Joy_Data

.PHONY: cross-eval
cross-eval:
	$(PYTHON) -m src.training.cross_gas_eval --data-dir Joy_Data

.PHONY: ablation
ablation:
	$(PYTHON) -m src.training.ablation --data-dir Joy_Data/Ethanol

# ── Maintenance ──────────────────────────────────────────────
.PHONY: clean
clean:
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true
	rm -rf output/coverage .coverage
	@echo "Clean complete."
