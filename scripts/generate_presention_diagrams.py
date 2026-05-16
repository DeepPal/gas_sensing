import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
SCIENTIFIC_DIR = ROOT / "output" / "scientific"
FIG4_CSV = ROOT / "output" / "publication_figures" / "Figure4_performance_table.csv"
ASSETS_DIR = ROOT / "Kevin_acetone_ppt" / "generated_assets"
GASES = ["Acetone", "Ethanol", "Methanol", "Isopropanol", "Toluene", "Xylene"]

CLINICAL_LIMITS = {
    "Acetone": 2.5,      # ppm (upper end of diabetic breath)
    "Ethanol": 0.1,
    "Methanol": 0.01,
    "Isopropanol": 0.05,
    "Toluene": 0.01,
    "Xylene": 0.01,
}


def _load_figure4_table() -> pd.DataFrame:
    df = pd.read_csv(FIG4_CSV)
    df.columns = [col.replace("\n", " ").strip() for col in df.columns]
    df["Gas"] = df["Gas"].str.strip()
    return df


def _load_calibration_metrics() -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for gas in GASES:
        metrics_path = SCIENTIFIC_DIR / gas / "metrics" / "calibration_metrics.json"
        if not metrics_path.is_file():
            continue
        data = json.loads(metrics_path.read_text())
        wl_shift = data.get("calibration_wavelength_shift", {}).get("centroid", {})
        loocv = data.get("loocv_validation", {}).get("wavelength_shift", {})
        results[gas] = {
            "r2": wl_shift.get("r2"),
            "r2_cv": loocv.get("r2_cv"),
        }
    return results


def _ensure_assets_dir() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def _plot_pipeline_flowchart(fig4_df: pd.DataFrame) -> None:
    steps = [
        "Raw Spectra",
        "Frame Selection",
        "Canonical Spectra",
        "ROI Discovery",
        "Calibration",
        "Validation",
        "Reporting",
        "Presentation",
    ]
    colors = [
        "#5D6D7E",
        "#1F77B4",
        "#2E86C1",
        "#C0392B",
        "#28B463",
        "#AF7AC5",
        "#F0B27A",
        "#17A589",
    ]
    best_lod = fig4_df.loc[fig4_df["LoD (ppm)"].astype(float).idxmin()]
    best_r2 = fig4_df.loc[fig4_df["R²"].astype(float).idxmax()]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    start_x = 0.1
    box_width = 0.9
    padding = 0.25

    for idx, (label, color) in enumerate(zip(steps, colors), start=1):
        x = start_x + (box_width + padding) * (idx - 1)
        rect = FancyBboxPatch(
            (x, 0.45),
            box_width,
            0.7,
            boxstyle="round,pad=0.12",
            linewidth=1.2,
            facecolor=color,
            edgecolor="#2C3E50",
        )
        ax.add_patch(rect)
        ax.text(
            x + box_width / 2,
            0.95,
            f"{idx}",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            x + box_width / 2,
            0.7,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="semibold",
            wrap=True,
        )
        if idx < len(steps):
            ax.annotate(
                "",
                xy=(x + box_width + 0.02, 0.8),
                xytext=(x + box_width + padding - 0.02, 0.8),
                arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.2),
            )

    text = (
        f"Best LoD: {float(best_lod['LoD (ppm)']):.2f} ppm ({best_lod['Gas']})\n"
        f"Best R²: {float(best_r2['R²']):.4f} ({best_r2['Gas']})"
    )
    ax.text(
        start_x + (box_width + padding) * (len(steps) - 1) / 2,
        0.15,
        text,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F8F5", edgecolor="#1ABC9C"),
    )
    x_extent = start_x + (box_width + padding) * len(steps)
    ax.set_xlim(0, x_extent)
    ax.set_ylim(0, 1.4)

    for ext in ("png", "pdf"):
        fig.savefig(ASSETS_DIR / f"pipeline_flowchart.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_calibration_robustness(metrics: Dict[str, Dict[str, float]]) -> None:
    gases = [g for g in GASES if g in metrics]
    r2 = [metrics[g]["r2"] for g in gases]
    r2_cv = [metrics[g]["r2_cv"] for g in gases]

    fig, ax = plt.subplots(figsize=(10, 4))
    indices = range(len(gases))
    width = 0.35

    ax.bar([i - width / 2 for i in indices], r2, width=width, color="#1F618D", label="R²")
    ax.bar([i + width / 2 for i in indices], r2_cv, width=width, color="#E67E22", label="LOOCV R²")

    for i, (train, cv) in enumerate(zip(r2, r2_cv)):
        delta = cv - train if train is not None and cv is not None else None
        if delta is not None:
            ax.text(
                i,
                max(train, cv) + 0.02,
                f"Δ {delta:+.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#2C3E50",
            )

    ax.axhline(0.8, color="#7DCEA0", linestyle="--", linewidth=1, label="Target R² ≥ 0.8")
    ax.set_xticks(list(indices))
    ax.set_xticklabels(gases)
    ax.set_ylabel("Coefficient of Determination")
    ax.set_title("Calibration Stability (Train vs LOOCV)")
    ax.set_ylim(min(-0.2, min(r2_cv) - 0.1), 1.05)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for ext in ("png", "pdf"):
        fig.savefig(ASSETS_DIR / f"calibration_robustness.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_lod_vs_r2(fig4_df: pd.DataFrame) -> None:
    lod = fig4_df["LoD (ppm)"].astype(float)
    r2 = fig4_df["R²"].astype(float)
    gases = fig4_df["Gas"].tolist()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    scatter = ax.scatter(
        r2,
        lod,
        s=120,
        c=["#1F77B4" if g in {"Acetone", "Xylene"} else "#E74C3C" for g in gases],
        edgecolor="#2C3E50",
        linewidths=0.8,
    )
    for gas, x, y in zip(gases, r2, lod):
        ax.text(x + 0.002, y + 0.1, gas, fontsize=9)

    ax.axvline(0.95, color="#7DCEA0", linestyle="--", linewidth=1, label="Target R² ≥ 0.95")
    ax.axhline(3.26, color="#F5B041", linestyle=":", linewidth=1, label="Reference LoD 3.26 ppm")
    ax.set_xlabel("R²")
    ax.set_ylabel("LoD (ppm)")
    ax.set_title("LoD vs. Linearity Across Gases")
    ax.set_ylim(0, max(lod.max() + 2, 10))
    ax.grid(alpha=0.4, linestyle=":")
    ax.legend(loc="upper right", frameon=False)

    for ext in ("png", "pdf"):
        fig.savefig(ASSETS_DIR / f"lod_tradeoff_diagram.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_clinical_comparison(fig4_df: pd.DataFrame) -> None:
    gases = []
    lod = []
    limits = []
    for _, row in fig4_df.iterrows():
        gas = row["Gas"].strip()
        if gas not in CLINICAL_LIMITS:
            continue
        gases.append(gas)
        lod.append(float(row["LoD (ppm)"]))
        limits.append(CLINICAL_LIMITS[gas])

    x = range(len(gases))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, lod, width=0.4, label="LoD (ppm)", color="#5DADE2")
    ax.bar([i + 0.4 for i in x], limits, width=0.4, label="Clinical Limit", color="#52BE80")

    for idx, (gas, lod_val, limit) in enumerate(zip(gases, lod, limits)):
        delta = lod_val - limit
        ax.text(idx + 0.2, max(lod_val, limit) + 0.3, f"Δ {delta:+.2f}", ha="center", fontsize=9)

    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(gases)
    ax.set_ylabel("ppm")
    ax.set_title("LoD vs. Clinical Requirement")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for ext in ("png", "pdf"):
        fig.savefig(ASSETS_DIR / f"clinical_thresholds.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    _ensure_assets_dir()
    fig4_df = _load_figure4_table()
    metrics = _load_calibration_metrics()
    _plot_pipeline_flowchart(fig4_df)
    _plot_calibration_robustness(metrics)
    _plot_lod_vs_r2(fig4_df)
    _plot_clinical_comparison(fig4_df)


if __name__ == "__main__":
    main()
