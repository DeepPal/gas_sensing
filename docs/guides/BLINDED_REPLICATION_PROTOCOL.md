# Blinded Replication Protocol

This protocol defines how external collaborators can validate model performance
without leaking labels during tuning.

## Scope

- Use profile: config/benchmark_profiles/external_blinded_profile.json
- Objective: reproduce concentration metrics on independent external data.
- Evidence artifacts are generated in output/qualification/ci.

## Protocol

1. External partner acquires dataset and computes per-file SHA256 checksums.
2. Partner creates blinded split using profile seed and holdout fraction.
3. Development team receives only blinded IDs and trains/finalizes model.
4. Final model checkpoint is hash-locked before any holdout labels are revealed.
5. Partner evaluates holdout and reports R2, RMSE, LOD RSD, LOQ RSD.
6. Both parties sign and archive report with artifact hashes.

## Success Criteria

- R2 >= 0.90
- RMSE <= 1.5 ppm
- LOD RSD <= 20%
- LOQ RSD <= 20%

## Automation

Run:

python scripts/generate_blinded_replication_manifest.py --output-dir output/qualification/ci

The qualification artifact workflow automatically includes this evidence package.
