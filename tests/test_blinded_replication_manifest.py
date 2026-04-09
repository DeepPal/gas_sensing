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


manifest_script = _load_script_module('generate_blinded_replication_manifest')


def test_build_manifest_contains_protocol_and_steps() -> None:
    profile = {
        'profile_id': 'external_blinded_v1',
        'external_dataset': {
            'dataset_id': 'ds-v1',
            'expected_path': 'data/external/partner_dataset_v1',
            'required_min_trials': 20,
        },
        'metrics': {'min_r2': 0.9},
        'blinding': {'seed': 42, 'holdout_fraction': 0.2},
    }

    manifest = manifest_script.build_manifest(profile=profile, run_id='r1', git_sha='abc')

    assert manifest['status'] == 'ok'
    assert manifest['protocol_id'] == 'blinded_replication_v1'
    assert len(manifest['steps']) >= 5
    assert manifest['dataset']['dataset_id'] == 'ds-v1'
