# ============================================================
#  SpectraAgent — Universal Agentic Spectroscopy Platform
#  Developer Makefile  |  Usage: make <target>
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
	@echo "  SpectraAgent — Universal Agentic Spectroscopy Platform"
	@echo ""
	@echo "  Setup"
	@echo "    make install          Install all dependencies into .venv"
	@echo "    make install-ml       Install + PyTorch (CNN support)"
	@echo "    make build-frontend   Build React frontend into static/dist"
	@echo ""
	@echo "  Quality"
	@echo "    make lint             Run ruff linter"
	@echo "    make check-workflows  Validate workflow syntax and guard rules"
	@echo "    make format           Auto-format with ruff"
	@echo "    make test             Run full test suite (1187 tests)"
	@echo "    make test-fast        Run fast lane tests (excludes reliability)"
	@echo "    make test-reliability Run reliability/integration lifecycle tests"
	@echo "    make test-reliability-report Run reliability tests with JUnit report"
	@echo "    make test-reliability-summary Run reliability tests + markdown summary"
	@echo "    make test-reliability-budget Run reliability tests + budget check"
	@echo "    make quality-gate     Run local fast + reliability lanes with reports"
	@echo "    make coverage         Run tests with coverage report"
	@echo "    make check            lint + test (CI equivalent)"
	@echo ""
	@echo "  Run"
	@echo "    make spectraagent     Start SpectraAgent server (simulation mode)"
	@echo "    make spectraagent-hw  Start SpectraAgent server (real hardware)"
	@echo "    make dashboard        Launch Streamlit scientific dashboard"
	@echo "    make serve            Launch legacy FastAPI inference server"
	@echo "    make simulate         Run legacy pipeline in simulation mode"
	@echo ""
	@echo "  MLflow"
	@echo "    make mlflow-ui        Open MLflow experiment tracking UI"
	@echo ""
	@echo "  Training"
	@echo "    make train-gpr        Train GPR calibration model"
	@echo "    make train-cnn        Train CNN gas classifier"
	@echo "    make cross-eval       Leave-one-gas-out cross-validation"
	@echo "    make ablation         Preprocessing ablation study"
	@echo ""
	@echo "  Maintenance"
	@echo "    make clean            Remove build artefacts and caches"
	@echo "    make release-checksums Generate SHA-256 checksums for dist artifacts"
	@echo "    make verify-release-manifest Verify checksum manifest coverage for dist artifacts"
	@echo "    make check-artifact-hygiene Validate no forbidden files in release artifacts"
	@echo "    make evidence-pack    Build research evidence pack (benchmark + blinded + qualification)"
	@echo "    make ci-diagnostics   Collect local CI-style diagnostics markdown"
	@echo "    make detect-flaky-tests Analyze test history for flaky tests"
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

.PHONY: build-frontend
build-frontend:
	cd spectraagent/webapp/frontend && npm install && npm run build
	@echo "Frontend built → spectraagent/webapp/static/dist/"

# ── Quality ──────────────────────────────────────────────────
.PHONY: lint
lint:
	$(RUFF) check src/

.PHONY: check-workflows
check-workflows:
	$(PYTHON) scripts/validate_workflows.py

.PHONY: format
format:
	$(RUFF) format src/ tests/
	$(RUFF) check src/ --fix --unsafe-fixes

.PHONY: test
test:
	$(PYTEST) tests/ -q

.PHONY: test-fast
test-fast:
	$(PYTEST) tests/ -q -x --tb=short -m "not reliability"

.PHONY: test-reliability
test-reliability:
	$(PYTEST) tests/ -q --tb=short -m "reliability"

.PHONY: test-reliability-report
test-reliability-report:
	mkdir -p output/test-results
	$(PYTEST) tests/ -q --tb=short -m "reliability" --durations=20 --junitxml=output/test-results/reliability-junit.xml
	@echo "JUnit report: output/test-results/reliability-junit.xml"

.PHONY: test-reliability-summary
test-reliability-summary: test-reliability-report
	$(PYTHON) scripts/summarize_junit.py --junit output/test-results/reliability-junit.xml --output output/test-results/reliability-summary.md --title "Reliability Local Summary" --top-n 10
	@echo "Markdown summary: output/test-results/reliability-summary.md"

.PHONY: test-reliability-budget
test-reliability-budget: test-reliability-summary
	$(PYTHON) scripts/check_junit_budget.py --junit output/test-results/reliability-junit.xml --output output/test-results/reliability-budget.md --title "Reliability Local Budget" --max-total-seconds 45 --max-case-seconds 12
	@echo "Budget summary: output/test-results/reliability-budget.md"

.PHONY: coverage
coverage:
	$(PYTEST) tests/ --cov=src --cov-report=term-missing --cov-report=html:output/coverage -q
	@echo "HTML report: output/coverage/index.html"

.PHONY: quality-gate
quality-gate:
	$(PYTHON) scripts/quality_gate.py --lane all --reliability-report --enforce-reliability-budget

.PHONY: check
check: lint test

# ── Run ──────────────────────────────────────────────────────
.PHONY: spectraagent
spectraagent:
	$(PYTHON) -m spectraagent start --simulate
	@echo "SpectraAgent → http://localhost:8765"

.PHONY: spectraagent-hw
spectraagent-hw:
	$(PYTHON) -m spectraagent start
	@echo "SpectraAgent (hardware) → http://localhost:8765"

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

.PHONY: release-checksums
release-checksums:
	$(PYTHON) scripts/generate_checksums.py --dist-dir dist --output sha256sums.txt

.PHONY: verify-release-manifest
verify-release-manifest:
	$(PYTHON) scripts/verify_release_manifest.py --dist-dir dist --manifest sha256sums.txt

.PHONY: check-artifact-hygiene
check-artifact-hygiene:
	$(PYTHON) scripts/check_artifact_hygiene.py --dist-dir dist

.PHONY: evidence-pack
evidence-pack:
	$(PYTHON) scripts/build_research_evidence_pack.py --output-dir output/qualification/local --session-id local-manual

.PHONY: ci-diagnostics
ci-diagnostics:
	$(PYTHON) scripts/collect_ci_diagnostics.py --output output/test-results/ci-diagnostics-local.md

.PHONY: detect-flaky-tests
detect-flaky-tests:
	$(PYTHON) scripts/detect_flaky_tests.py \
		--history-dir output/test-history \
		--output output/test-results/flaky-report.md \
		--advisory
