from pathlib import Path
from config.config_loader import load_config
from gas_analysis.core import pipeline as pl

BASE = Path("data/JOY_Data")
JOBS = {
    "EtOH": (
        BASE / "Multi mix vary-EtOH",
        BASE / "ref MutiAuMIP-EtOH.csv",
        Path("output") / "etoh_topavg",
    ),
    "IPA": (
        BASE / "Multi mix vary-IPA",
        BASE / "ref AuMutiMIP-IPA.csv",
        Path("output") / "ipa_topavg",
    ),
    "MeOH": (
        BASE / "Multi mix vary-MeOH",
        BASE / "ref AuMutiMIP-MeOH.csv",
        Path("output") / "meoh_topavg",
    ),
    "MIX": (
        BASE / "Mixed gas",
        BASE / "ref AuMutiMIP-MIX.csv",
        Path("output") / "mix_topavg",
    ),
}

def run_job(label, data_dir, ref_path, out_dir, avg_top_n=10, scan_full=True, top_k=10):
    pl.CONFIG = load_config()
    print(f"\n=== {label} ===")
    result = pl.run_full_pipeline(
        root_dir=str(data_dir.resolve()),
        ref_path=str(ref_path.resolve()),
        out_root=str(out_dir.resolve()),
        diff_threshold=0.01,
        avg_top_n=avg_top_n,
        scan_full=scan_full,
        top_k_candidates=top_k,
    )
    print(f"Outputs → {out_dir}")
    print("Top full-scan candidates (main):", result.get("fullscan_concentration_response_metrics"))
    for mode, payload in (result.get("top_avg_results") or {}).items():
        print(f"Top full-scan candidates ({mode}): {payload.get('fullscan_metrics_path')}")

if __name__ == "__main__":
    for label, (data_dir, ref_path, out_dir) in JOBS.items():
        run_job(label, data_dir, ref_path, out_dir)