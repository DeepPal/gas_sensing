from __future__ import annotations

import importlib.util
import json
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


qualification_artifacts = _load_script_module('generate_qualification_artifacts')


def test_load_latest_benchmark_evidence_returns_latest_json(tmp_path: Path) -> None:
    older = tmp_path / 'benchmark_evidence_20260101_000000.json'
    newer = tmp_path / 'benchmark_evidence_20260102_000000.json'

    older.write_text(json.dumps({'summary': {'novelty_signal': False}}), encoding='utf-8')
    newer.write_text(json.dumps({'summary': {'novelty_signal': True}}), encoding='utf-8')

    payload = qualification_artifacts._load_latest_benchmark_evidence(tmp_path)

    assert payload is not None
    assert payload['summary']['novelty_signal'] is True
