#!/usr/bin/env python3
"""Generate a blinded replication protocol manifest for external validation.

This script creates machine-readable and markdown protocol artifacts for
external collaborators. It does not require private external data to exist in CI.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _load_profile(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def build_manifest(*, profile: dict[str, Any], run_id: str, git_sha: str) -> dict[str, Any]:
    dataset_cfg = profile.get('external_dataset', {})
    expected_path = Path(str(dataset_cfg.get('expected_path', 'data/external/partner_dataset_v1')))
    dataset_available = expected_path.exists()

    return {
        'status': 'ok',
        'protocol_id': 'blinded_replication_v1',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'profile_id': profile.get('profile_id', 'external_blinded_v1'),
        'dataset': {
            'dataset_id': dataset_cfg.get('dataset_id', 'external_partner_dataset_v1'),
            'expected_path': str(expected_path),
            'available_in_run': dataset_available,
            'required_min_trials': int(dataset_cfg.get('required_min_trials', 20)),
        },
        'blinding': profile.get('blinding', {}),
        'steps': [
            'Partner acquires raw spectra and generates immutable checksum list.',
            'Partner applies blinded split with fixed seed and holdout fraction from profile.',
            'Team receives only blinded sample IDs for training and threshold tuning.',
            'Final model is frozen and checksum-signed before label reveal.',
            'Partner reveals holdout labels and computes final metrics without retraining.',
            'Both parties sign the final replication report and attach artifact hashes.',
        ],
        'success_criteria': profile.get('metrics', {}),
        'ci': {
            'run_id': run_id,
            'git_sha': git_sha,
        },
    }


def _render_markdown(manifest: dict[str, Any]) -> str:
    dataset = manifest['dataset']
    criteria = manifest['success_criteria']
    steps = '\n'.join(f"{idx}. {step}" for idx, step in enumerate(manifest['steps'], start=1))

    return (
        '# Blinded Replication Protocol\n\n'
        f"- Protocol ID: {manifest['protocol_id']}\n"
        f"- Profile: {manifest['profile_id']}\n"
        f"- Dataset ID: {dataset['dataset_id']}\n"
        f"- Dataset path: {dataset['expected_path']}\n"
        f"- Dataset available in this run: {dataset['available_in_run']}\n\n"
        '## Success Criteria\n\n'
        f"- Min R2: {criteria.get('min_r2', 'n/a')}\n"
        f"- Max RMSE (ppm): {criteria.get('max_rmse_ppm', 'n/a')}\n"
        f"- Max LOD RSD (%): {criteria.get('max_lod_rsd_pct', 'n/a')}\n"
        f"- Max LOQ RSD (%): {criteria.get('max_loq_rsd_pct', 'n/a')}\n\n"
        '## Protocol Steps\n\n'
        f"{steps}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='output/qualification/ci')
    parser.add_argument('--profile', default='config/benchmark_profiles/external_blinded_profile.json')
    parser.add_argument('--run-id', default='local')
    parser.add_argument('--git-sha', default='unknown')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = _load_profile(Path(args.profile))
    manifest = build_manifest(profile=profile, run_id=args.run_id, git_sha=args.git_sha)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    json_path = out_dir / f'blinded_replication_manifest_{stamp}.json'
    md_path = out_dir / f'blinded_replication_protocol_{stamp}.md'

    json_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    md_path.write_text(_render_markdown(manifest), encoding='utf-8')

    print(json.dumps({
        'status': 'ok',
        'json': str(json_path),
        'markdown': str(md_path),
        'dataset_available': manifest['dataset']['available_in_run'],
    }))


if __name__ == '__main__':
    main()
