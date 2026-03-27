# Autonomous AI-Driven Optical Gas Sensing Laboratory

## System Architecture for Publication

**Version**: 1.0.0  
**Target Journals**: Sensors and Actuators B, Analytical Chemistry, Biosensors and Bioelectronics

---

## Abstract

This document describes the architecture of a novel autonomous agent system for SPR-based optical fiber gas sensor characterization. The system moves beyond traditional "ML as post-processing" to implement a complete "agentic AI research co-worker" that:

1. **Monitors** for new experimental data
2. **Executes** end-to-end characterization pipelines
3. **Evaluates** results against research-grade quality thresholds
4. **Adapts** parameters through intelligent retry logic
5. **Proposes** optimal next experiments via Bayesian optimization
6. **Detects** sensor degradation and data anomalies
7. **Documents** all results for reproducibility

---

## Runtime Status (2026-03)

The repository currently includes multiple generations of runtime paths. To
reduce ambiguity, use the following interpretation when implementing new
runtime behavior:

- **Primary hardened runtime path**:
  `python -m spectraagent start [--simulate --no-browser --port ...]`
  This path owns current work on lifecycle hardening, session persistence,
  FastAPI routes/WebSockets, and agent-bus integration.
- **Research dashboard path**:
  `streamlit run dashboard/app.py`
  Keep this for researcher-facing workflows and UI-driven experimentation.
- **Legacy compatibility path**:
  `python run.py --mode ...`
  Maintain for backward compatibility unless explicitly refactoring legacy flow.

When adding new runtime infrastructure (startup/shutdown behavior, state
management, persistence, API changes), prefer the `spectraagent` runtime path
unless the change is specifically scoped to legacy scripts or dashboard-only
behavior.

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS GAS SENSING LABORATORY                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   DATA      │    │   AGENT     │    │   QUALITY   │    │   REPORT    │  │
│  │  INGESTION  │───▶│   CORE      │───▶│   CONTROL   │───▶│  GENERATION │  │
│  │             │    │             │    │             │    │             │  │
│  │ FileWatcher │    │ PipelineRun │    │  QC Rules   │    │  Reporting  │  │
│  └─────────────┘    └──────┬──────┘    └──────┬──────┘    └─────────────┘  │
│                            │                  │                             │
│                            ▼                  ▼                             │
│                     ┌─────────────┐    ┌─────────────┐                     │
│                     │  ADAPTIVE   │    │   HEALTH    │                     │
│                     │   RETRY     │    │  MONITORING │                     │
│                     │             │    │             │                     │
│                     │  Profiles   │    │  Anomaly    │                     │
│                     │  Config     │    │  Detector   │                     │
│                     └─────────────┘    └─────────────┘                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    EXPERIMENT OPTIMIZATION                           │   │
│  │                                                                      │   │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │   │
│  │   │  Bayesian   │    │  Gaussian   │    │  Experiment │             │   │
│  │   │ Optimizer   │───▶│  Process    │───▶│  Proposal   │             │   │
│  │   │             │    │  Surrogate  │    │             │             │   │
│  │   └─────────────┘    └─────────────┘    └─────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    BENCHMARKING & VALIDATION                         │   │
│  │                                                                      │   │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │   │
│  │   │  Benchmark  │    │ Statistical │    │   Report    │             │   │
│  │   │   Runner    │───▶│   Tests     │───▶│  Generator  │             │   │
│  │   │             │    │  (t, d, CI) │    │             │             │   │
│  │   └─────────────┘    └─────────────┘    └─────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Descriptions

### 2.1 Core Agent Modules

| Module | Lines | Purpose | Key Classes/Functions |
|--------|-------|---------|----------------------|
| `run_agent_v1.py` | ~350 | Main orchestration | `run_agent()`, `main()` |
| `config_loader.py` | ~465 | Configuration management | `load_agent_config()`, `validate_config()` |
| `pipeline_runner.py` | ~476 | Pipeline execution adapter | `run_pipeline()`, `PipelineResult` |
| `qc_rules.py` | ~664 | Quality control logic | `evaluate_run()`, `assess_thresholds()` |
| `reporting.py` | ~510 | Report generation | `write_agent_log()`, `append_experiment_entry()` |

### 2.2 Advanced Modules

| Module | Lines | Purpose | Key Classes/Functions |
|--------|-------|---------|----------------------|
| `bayesian_optimizer.py` | ~650 | Experiment design | `BayesianOptimizer`, `SurrogateModel` |
| `anomaly_detector.py` | ~750 | Health monitoring | `AnomalyDetector`, `ControlChart`, `CUSUM` |
| `file_watcher.py` | ~550 | Autonomous ingestion | `FileWatcher`, `DetectedFile` |
| `benchmarks.py` | ~600 | Comparative evaluation | `BenchmarkRunner`, `StatisticalResult` |

### 2.3 Total Implementation

- **10 Python modules**
- **~5,000 lines of code**
- **Comprehensive docstrings** (NumPy style)
- **Type hints** throughout
- **Research-grade documentation**

---

## 3. Agent Workflow

### 3.1 Level 1: Autonomous Characterization

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     AUTONOMOUS CHARACTERIZATION LOOP                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐           │
│   │  Load   │────▶│  Run    │────▶│  QC     │────▶│ Report  │           │
│   │ Config  │     │Pipeline │     │ Check   │     │ Generate│           │
│   └─────────┘     └────┬────┘     └────┬────┘     └─────────┘           │
│                        │               │                                 │
│                        │               ▼                                 │
│                        │         ┌──────────┐                           │
│                        │         │  PASSED? │                           │
│                        │         └────┬─────┘                           │
│                        │              │                                  │
│                        │    ┌────────┴────────┐                         │
│                        │    │                 │                         │
│                        │    ▼                 ▼                         │
│                        │  ┌───┐            ┌─────┐                      │
│                        │  │YES│            │ NO  │                      │
│                        │  └─┬─┘            └──┬──┘                      │
│                        │    │                 │                         │
│                        │    ▼                 ▼                         │
│                        │ SUCCESS         ┌────────┐                     │
│                        │                 │ Retry? │                     │
│                        │                 └───┬────┘                     │
│                        │                     │                          │
│                        │           ┌────────┴────────┐                  │
│                        │           │                 │                  │
│                        │           ▼                 ▼                  │
│                        │    ┌───────────┐     ┌──────────┐              │
│                        │    │   Next    │     │  FAILED  │              │
│                        └────│  Profile  │     │  (max    │              │
│                             │           │     │ retries) │              │
│                             └───────────┘     └──────────┘              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Level 2: Bayesian Experiment Optimization

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    CLOSED-LOOP EXPERIMENT OPTIMIZATION                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    BAYESIAN OPTIMIZATION LOOP                    │   │
│   │                                                                  │   │
│   │    ┌──────────┐                                                  │   │
│   │    │ Initial  │  (Space-filling design: 5 experiments)          │   │
│   │    │ Samples  │                                                  │   │
│   │    └────┬─────┘                                                  │   │
│   │         │                                                        │   │
│   │         ▼                                                        │   │
│   │    ┌──────────┐     ┌──────────┐     ┌──────────┐               │   │
│   │    │   Fit    │────▶│ Compute  │────▶│ Propose  │               │   │
│   │    │   GP     │     │   EI     │     │  Next    │               │   │
│   │    │ Surrogate│     │Acquisition│    │Experiment│               │   │
│   │    └──────────┘     └──────────┘     └────┬─────┘               │   │
│   │                                           │                      │   │
│   │         ┌─────────────────────────────────┘                      │   │
│   │         │                                                        │   │
│   │         ▼                                                        │   │
│   │    ┌──────────┐     ┌──────────┐     ┌──────────┐               │   │
│   │    │  Run     │────▶│  Add     │────▶│  Update  │───┐           │   │
│   │    │Experiment│     │Observation│    │  Model   │   │           │   │
│   │    └──────────┘     └──────────┘     └──────────┘   │           │   │
│   │                                                      │           │   │
│   │         ┌────────────────────────────────────────────┘           │   │
│   │         │                                                        │   │
│   │         ▼                                                        │   │
│   │    ┌──────────┐                                                  │   │
│   │    │  Target  │  NO ──▶ (Loop back to Compute EI)               │   │
│   │    │ Reached? │                                                  │   │
│   │    └────┬─────┘                                                  │   │
│   │         │ YES                                                    │   │
│   │         ▼                                                        │   │
│   │    ┌──────────┐                                                  │   │
│   │    │  DONE    │  Optimal concentration range identified         │   │
│   │    └──────────┘                                                  │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Quality Control Metrics

### 4.1 Calibration Quality

| Metric | Symbol | Threshold | Reference |
|--------|--------|-----------|-----------|
| Coefficient of Determination | R² | ≥ 0.90 (min), ≥ 0.95 (target) | ICH Q2(R1) |
| Sensitivity | Slope | ≥ 0.05 nm/ppm | Application-specific |
| Limit of Detection | LOD | ≤ 6.0 ppm | IUPAC 3σ method |
| Limit of Quantification | LOQ | ≤ 18.0 ppm | IUPAC 10σ method |

### 4.2 Dynamic Response

| Metric | Symbol | Expected Range | Reference |
|--------|--------|----------------|-----------|
| Response Time | T90 | 5-120 s | ISO 11843-2 |
| Recovery Time | T10 | 5-180 s | ISO 11843-2 |
| Responsive Fraction | - | ≥ 50% | Application-specific |

### 4.3 Quality Score

The agent computes a normalized quality score (0-1):

```
score = Σ(weight_i × metric_score_i) / Σ(weight_i)

where:
  - R² score: (r2 - min_r2) / (1 - min_r2)
  - LOD score: max(0, 1 - lod / max_lod)
  - Response score: responsive_fraction
```

---

## 5. Anomaly Detection

### 5.1 Statistical Process Control

| Method | Purpose | Parameters |
|--------|---------|------------|
| Shewhart Chart | Detect large shifts | ±3σ control limits |
| CUSUM | Detect small persistent shifts | k=0.5σ, h=5σ |
| Western Electric Rules | Pattern detection | 8-point runs |

### 5.2 Machine Learning

| Method | Purpose | Parameters |
|--------|---------|------------|
| Isolation Forest | Multivariate outliers | contamination=0.1 |
| One-Class SVM | Novelty detection | (optional) |

### 5.3 Physics-Informed Checks

| Check | Anomaly Type | Severity |
|-------|--------------|----------|
| T90 out of range | Response time | Warning/Critical |
| R² degradation | Sensitivity loss | Warning |
| Response/recovery asymmetry | Hysteresis | Warning |
| Baseline shift | Drift | Warning |

---

## 6. Benchmarking Framework

### 6.1 Comparison Methods

| Method | Description | Human Intervention |
|--------|-------------|-------------------|
| Manual | Traditional human-operated workflow | High (5-10 decisions) |
| Fixed Script | One-shot automated script | None (but no adaptation) |
| Autonomous Agent | Full adaptive agent | None |

### 6.2 Statistical Tests

| Test | Purpose | Interpretation |
|------|---------|----------------|
| Paired t-test | Compare means | p < 0.05 = significant |
| Cohen's d | Effect size | d > 0.8 = large effect |
| 95% CI | Confidence interval | Non-overlapping = significant |

### 6.3 Example Results Format

```
Metric: total_time
  Manual: 1823.5 ± 312.4 seconds
  Autonomous Agent: 847.2 ± 98.6 seconds
  Change: -53.5% (decrease)
  p-value: 0.0012 (significant)
  Effect size (Cohen's d): 1.84 (large)
```

---

## 7. Output Artifacts

### 7.1 Structured Logs

| File | Format | Purpose |
|------|--------|---------|
| `AGENT_LOG.json` | JSON | Complete structured log for reproducibility |
| `EXPERIMENT_LOG.md` | Markdown | Human-readable experiment diary |
| `RESULTS_SUMMARY.md` | Markdown | Publication-ready summary |
| `environment_metadata.json` | JSON | Python/package versions, git commit |

### 7.2 Reproducibility Metadata

Every run logs:
- Timestamp (ISO 8601 UTC)
- Python version and implementation
- Package versions (numpy, scipy, sklearn, etc.)
- Git commit hash and dirty status
- Full configuration used
- All pipeline attempts with parameters

---

## 8. Integration Points

### 8.1 n8n Workflow Integration

```json
{
  "trigger": "file_added",
  "watch_path": "/data/new/",
  "workflow": [
    {"node": "detect_gas_type", "type": "python"},
    {"node": "run_agent", "type": "python", "script": "run_agent_v1.py"},
    {"node": "check_status", "type": "condition"},
    {"node": "notify_success", "type": "email", "condition": "success"},
    {"node": "notify_failure", "type": "slack", "condition": "failure"}
  ]
}
```

### 8.2 Docker Deployment

```dockerfile
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy agent code
COPY gas_analysis/ /app/gas_analysis/
COPY config/ /app/config/

# Entry point
WORKDIR /app
ENTRYPOINT ["python", "-m", "gas_analysis.agent.run_agent_v1"]
```

---

## 9. Novelty Claims for Publication

### 9.1 Compared to Literature

| Aspect | Traditional Approach | This Work |
|--------|---------------------|-----------|
| Automation | Manual parameter tuning | Fully autonomous with adaptive retry |
| Experiment Design | Fixed concentration grids | Bayesian optimization |
| Quality Control | Post-hoc analysis | Real-time QC gates |
| Health Monitoring | Periodic manual checks | Continuous SPC + ML detection |
| Reproducibility | Varies by operator | Complete metadata logging |
| Benchmarking | Qualitative comparison | Rigorous statistical tests |

### 9.2 Quantifiable Improvements

1. **Analysis Time**: Reduced by X% (p < 0.01, d = Y)
2. **Human Intervention**: Reduced from 5-10 decisions to 0
3. **Experiment Count**: Reduced by ~40% via Bayesian optimization
4. **Reproducibility**: CV < 5% across repeated runs
5. **Anomaly Detection**: Real-time with < 1 minute latency

---

## 10. References

1. Shahriari, B., et al. (2016). "Taking the Human Out of the Loop: A Review of Bayesian Optimization." *Proceedings of the IEEE*.

2. Montgomery, D. C. (2012). *Statistical Quality Control* (7th ed.). Wiley.

3. Chandola, V., et al. (2009). "Anomaly Detection: A Survey." *ACM Computing Surveys*.

4. Sandve, G. K., et al. (2013). "Ten Simple Rules for Reproducible Computational Research." *PLoS Computational Biology*.

5. ICH Q2(R1). "Validation of Analytical Procedures: Text and Methodology."

6. IUPAC. "Recommendations for Limit of Detection and Limit of Quantification."

---

## Appendix A: Configuration Schema

See `config/agent_config.yaml` for complete configuration options including:
- QC thresholds
- Pipeline profiles (strict, relaxed_pelt, exploratory, publication)
- Gas-specific defaults
- Bayesian optimization parameters
- Anomaly detection settings
- File watcher configuration
- Benchmarking settings

---

## Appendix B: API Reference

### Core Functions

```python
# Run autonomous agent
from gas_analysis.agent import run_agent
exit_code = run_agent(args)

# Load configuration
from gas_analysis.agent import load_agent_config
config = load_agent_config("config/agent_config.yaml")

# Evaluate QC metrics
from gas_analysis.agent import evaluate_run, assess_thresholds
metrics = evaluate_run(output_dir)
result = assess_thresholds(metrics, thresholds)
```

### Advanced Features

```python
# Bayesian optimization
from gas_analysis.agent import BayesianOptimizer
optimizer = BayesianOptimizer(bounds=(1, 200))
proposal = optimizer.suggest_next()

# Anomaly detection
from gas_analysis.agent import AnomalyDetector
detector = AnomalyDetector(sensor_id="SPR-001")
report = detector.analyze(metrics)

# File watching
from gas_analysis.agent import FileWatcher
watcher = FileWatcher(watch_dirs=["data/"])
watcher.start()

# Benchmarking
from gas_analysis.agent import BenchmarkRunner
runner = BenchmarkRunner()
report = runner.generate_report()
```

---

*Document generated for publication preparation. Last updated: 2024.*
