from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / 'scripts'


def _load_script_module(name: str):
    module_path = SCRIPTS_DIR / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmark_evidence = _load_script_module('generate_benchmark_evidence')


def test_build_evidence_payload_uses_synthetic_when_no_data_dir() -> None:
    payload = benchmark_evidence.build_evidence_payload(data_dir=None)

    assert payload['status'] == 'ok'
    assert payload['benchmark']['mode'] == 'synthetic'
    assert 'summary' in payload
    assert payload['summary']['baseline_r2'] > 0.9


def test_build_evidence_payload_includes_novelty_signal_boolean() -> None:
    payload = benchmark_evidence.build_evidence_payload(data_dir=None)

    assert isinstance(payload['summary']['novelty_signal'], bool)
