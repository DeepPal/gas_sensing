"""
tests.test_training_agent
=========================
Unit tests for src.agents.training.TrainingAgent.

Tests cover:
  - Initial state / status
  - Volume trigger firing
  - Drift trigger via notify_drift()
  - Performance degradation trigger
  - Manual trigger
  - push() counter increments
  - RetrainResult dataclass helpers
  - No retraining when already retraining (lock guard)
"""

from __future__ import annotations

from datetime import datetime, timezone
import time

from src.agents.training import RetrainResult, RetrainTrigger, TrainingAgent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _agent(**kwargs) -> TrainingAgent:
    """Create a TrainingAgent with MLflow disabled and fast trigger thresholds."""
    defaults = dict(
        model_dir="output/models",
        sessions_dir="output/sessions",
        min_r2_threshold=0.90,
        retrain_every_n_samples=10,  # low so tests can trigger it fast
        min_samples_for_retrain=1,
        cnn_epochs=1,
        mlflow_uri=None,  # disable MLflow in tests
        retrain_cooldown_s=0,  # no cooldown so triggers fire immediately
    )
    defaults.update(kwargs)
    return TrainingAgent(**defaults)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestTrainingAgentInit:
    def test_initial_status_zero_samples(self):
        agent = _agent()
        status = agent.get_status()
        assert status["total_samples"] == 0
        assert status["samples_since_retrain"] == 0

    def test_not_retraining_at_start(self):
        agent = _agent()
        assert not agent.get_status()["is_retraining"]

    def test_no_drift_pending_at_start(self):
        agent = _agent()
        assert not agent.get_status()["drift_pending"]

    def test_empty_results_history(self):
        agent = _agent()
        assert agent.get_results() == []


# ---------------------------------------------------------------------------
# push() counter
# ---------------------------------------------------------------------------


class TestPushCounter:
    def test_sample_count_increments(self):
        agent = _agent(retrain_every_n_samples=1000)
        for _ in range(5):
            agent.push()
        assert agent.get_status()["total_samples"] == 5

    def test_r2_window_populated(self):
        agent = _agent(retrain_every_n_samples=1000)
        for r2 in [0.95, 0.96, 0.97]:
            agent.push(gpr_r2=r2)
        status = agent.get_status()
        assert status["avg_r2_last_window"] is not None
        assert 0.95 <= status["avg_r2_last_window"] <= 0.97

    def test_push_returns_none_below_threshold(self):
        agent = _agent(retrain_every_n_samples=1000)
        result = agent.push(gpr_r2=0.99)
        assert result is None


# ---------------------------------------------------------------------------
# Volume trigger
# ---------------------------------------------------------------------------


class TestVolumeTrigger:
    def test_volume_trigger_fires_at_threshold(self):
        agent = _agent(retrain_every_n_samples=5)
        trigger = None
        for _ in range(5):
            trigger = agent.push()
        assert trigger == RetrainTrigger.VOLUME_THRESHOLD

    def test_counter_resets_after_trigger(self, tmp_path):
        # Use an empty sessions_dir so _collect_session_data finds no CSVs and
        # the cycle exits via the "insufficient data" fast-path — deterministic
        # and independent of real output/sessions/ content on the developer's machine.
        agent = _agent(
            retrain_every_n_samples=3,
            retrain_cooldown_s=0,
            sessions_dir=str(tmp_path),
        )
        for _ in range(3):
            agent.push()
        # wait_for_retrain() blocks until the background thread finishes
        assert agent.wait_for_retrain(timeout=5.0), "Retrain cycle did not finish in time"
        assert agent.get_status()["samples_since_retrain"] == 0


# ---------------------------------------------------------------------------
# Drift trigger
# ---------------------------------------------------------------------------


class TestDriftTrigger:
    def test_notify_drift_sets_flag(self):
        agent = _agent(retrain_every_n_samples=1000)
        agent.notify_drift()
        assert agent.get_status()["drift_pending"]

    def test_drift_trigger_fires_on_next_push(self):
        agent = _agent(retrain_every_n_samples=1000)
        agent.notify_drift()
        trigger = agent.push()
        assert trigger == RetrainTrigger.DRIFT_ALERT

    def test_drift_flag_cleared_after_trigger(self):
        agent = _agent(retrain_every_n_samples=1000)
        agent.notify_drift()
        agent.push()
        assert not agent.get_status()["drift_pending"]


# ---------------------------------------------------------------------------
# Performance degradation trigger
# ---------------------------------------------------------------------------


class TestPerformanceTrigger:
    def test_performance_trigger_fires_when_r2_low(self):
        agent = _agent(
            retrain_every_n_samples=1000,
            min_r2_threshold=0.90,
        )
        # Fill the R² window; the trigger fires on the push that makes the
        # window reach _r2_window_size, then a background thread starts.
        # Capture any trigger raised during filling.
        triggered = None
        for _ in range(agent._r2_window_size + 1):
            t = agent.push(gpr_r2=0.70)
            if t is not None:
                triggered = t
                break

        assert triggered == RetrainTrigger.PERFORMANCE_DEGRADATION

    def test_no_trigger_when_r2_ok(self):
        agent = _agent(
            retrain_every_n_samples=1000,
            min_r2_threshold=0.90,
        )
        for _ in range(agent._r2_window_size + 1):
            trigger = agent.push(gpr_r2=0.99)
        assert trigger is None


# ---------------------------------------------------------------------------
# Manual trigger
# ---------------------------------------------------------------------------


class TestManualTrigger:
    def test_manual_trigger_launches_thread(self, tmp_path):
        # Empty sessions_dir → "insufficient data" fast-path keeps the cycle
        # deterministically fast regardless of output/sessions/ on disk.
        agent = _agent(sessions_dir=str(tmp_path))
        agent.trigger_manual_retrain()
        assert agent.wait_for_retrain(timeout=5.0), "Manual retrain did not finish in time"
        results = agent.get_results()
        assert len(results) >= 1
        assert results[-1].trigger == RetrainTrigger.MANUAL

    def test_manual_trigger_when_retraining_is_noop(self):
        """Calling manual trigger while already retraining should not start a second cycle."""
        agent = _agent()
        agent._is_retraining = True  # simulate in-progress retrain
        agent.trigger_manual_retrain()
        time.sleep(0.05)
        assert len(agent.get_results()) == 0  # no new cycle started
        agent._is_retraining = False


# ---------------------------------------------------------------------------
# RetrainResult helpers
# ---------------------------------------------------------------------------


class TestRetrainResult:
    def _make_result(self, **kwargs) -> RetrainResult:
        return RetrainResult(
            trigger=RetrainTrigger.MANUAL,
            timestamp=datetime.now(timezone.utc),
            **kwargs,
        )

    def test_improved_true_when_gpr_r2_increases(self):
        r = self._make_result(gpr_trained=True, gpr_r2_before=0.85, gpr_r2_after=0.92)
        assert r.improved()

    def test_improved_false_when_r2_decreases(self):
        r = self._make_result(gpr_trained=True, gpr_r2_before=0.95, gpr_r2_after=0.88)
        assert not r.improved()

    def test_improved_true_when_cnn_acc_increases(self):
        r = self._make_result(cnn_trained=True, cnn_acc_before=0.80, cnn_acc_after=0.92)
        assert r.improved()

    def test_improved_false_when_no_models_trained(self):
        r = self._make_result()
        assert not r.improved()

    def test_model_promoted_starts_false(self):
        r = self._make_result()
        assert not r.model_promoted


# ---------------------------------------------------------------------------
# get_status completeness
# ---------------------------------------------------------------------------


class TestGetStatus:
    def test_status_has_all_expected_keys(self):
        agent = _agent()
        status = agent.get_status()
        expected = {
            "is_retraining",
            "total_samples",
            "samples_since_retrain",
            "avg_r2_last_window",
            "drift_pending",
            "n_retrain_cycles",
            "last_retrain",
        }
        assert expected.issubset(status.keys())

    def test_last_retrain_none_initially(self):
        agent = _agent()
        assert agent.get_status()["last_retrain"] is None
