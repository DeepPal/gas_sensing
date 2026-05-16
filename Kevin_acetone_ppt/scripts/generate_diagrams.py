"""
Enhanced Scientific Diagrams for Presentation - Version 2
Professional publication-quality figures with improved styling
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

import seaborn as sns
import numpy as np
from scipy.signal import savgol_filter
from scipy import stats

import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).parent.parent / "generated_assets"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_DIR = Path(__file__).parent.parent / "data"
SCIENTIFIC_DIR = PROJECT_ROOT / "output" / "scientific"
PUBLICATION_DIR = PROJECT_ROOT / "output" / "publication_figures"
FIGURE4_TABLE = PUBLICATION_DIR / "Figure4_performance_table.csv"
DEFAULT_REFERENCE = {
    "lod": 3.26,
    "sensitivity": 0.116,
    "r2": 0.95,
}


LOD_FACTOR = 3.0


def _parse_roi_nm(text: str) -> tuple[float, float] | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    import re

    nums = [float(m.group(0)) for m in re.finditer(r"[-+]?\d*\.?\d+", raw)]
    if len(nums) < 2:
        return None
    lo, hi = nums[0], nums[1]
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _load_multigas_results() -> pd.DataFrame:
    primary = DATA_DIR / "multigas_results.csv"
    df: pd.DataFrame | None = None
    if primary.exists():
        try:
            df = pd.read_csv(primary)
        except Exception:
            df = None
    if (df is None or df.empty) and FIGURE4_TABLE.exists():
        try:
            df = pd.read_csv(FIGURE4_TABLE)
        except Exception:
            df = None
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    def _find_col(*candidates: str) -> str | None:
        for cand in candidates:
            for col in df.columns:
                if col.lower().strip() == cand.lower().strip():
                    return col
        return None

    rename_map: dict[str, str] = {}

    for target, candidates in (
        ("VOC", ("VOC", "Gas", "gas")),
        ("Optimal ROI (nm)", ("Optimal ROI (nm)", "ROI (nm)")),
        ("LoD (ppm)", ("LoD (ppm)", "lod_ppm", "lod")),
        ("Sensitivity (nm/ppm)", ("Sensitivity (nm/ppm)", "sensitivity")),
        ("R²", ("R²", "r_squared", "r2")),
        ("Spearman ρ", ("Spearman ρ", "spearman", "spearman_rho")),
    ):
        col = _find_col(*candidates)
        if col and col != target:
            rename_map[col] = target

    if rename_map:
        df = df.rename(columns=rename_map)

    roi_start_col = _find_col("roi_start_nm", "roi_start")
    roi_end_col = _find_col("roi_end_nm", "roi_end")
    if roi_start_col and roi_end_col:
        def _fmt_roi(row) -> str:
            try:
                lo = float(row[roi_start_col])
                hi = float(row[roi_end_col])
            except (ValueError, TypeError):
                return ""
            if not np.isfinite(lo) or not np.isfinite(hi):
                return ""
            lo, hi = (lo, hi) if lo <= hi else (hi, lo)
            return f"{lo:.0f}-{hi:.0f}"

        df["Optimal ROI (nm)"] = df.apply(_fmt_roi, axis=1)

    # If ROI string missing but Figure4 ROI column exists, seed from there
    roi_col = _find_col("ROI (nm)")
    if "Optimal ROI (nm)" not in df.columns and roi_col:
        df["Optimal ROI (nm)"] = df[roi_col]

    return df


def _load_acetone_canonical_spectra(conc_ppm: float) -> pd.DataFrame:
    path = SCIENTIFIC_DIR / "Acetone" / "canonical_spectra" / f"{conc_ppm:.1f}ppm.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_voc_canonical_spectra(voc: str, conc_ppm: float) -> pd.DataFrame:
    path = SCIENTIFIC_DIR / voc.capitalize() / "canonical_spectra" / f"{conc_ppm:.1f}ppm.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _save_figure(stem: str, *, dpi: int = 300) -> None:
    plt.savefig(OUTPUT_DIR / f"{stem}.png", dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches='tight', facecolor='white')


def _first_float(value) -> float | None:
    try:
        text = str(value)
    except Exception:
        return None
    import re

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _load_reference_metrics() -> dict:
    path = DATA_DIR / "reference_metrics.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if not {"Metric", "Value"}.issubset(set(df.columns)):
        return {}
    return {
        str(row["Metric"]).strip(): str(row.get("Value", "")).strip()
        for _, row in df.iterrows()
        if str(row.get("Metric", "")).strip()
    }


def _load_comparison_table() -> dict:
    path = DATA_DIR / "comparison_table.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if not {"Metric", "Reference Paper", "This Work"}.issubset(set(df.columns)):
        return {}
    by_metric = {
        str(row["Metric"]).strip(): row
        for _, row in df.iterrows()
        if str(row.get("Metric", "")).strip()
    }
    out: dict = {}
    for key, row in by_metric.items():
        out[key] = {
            "paper": str(row.get("Reference Paper", "")).strip(),
            "this": str(row.get("This Work", "")).strip(),
        }
    return out


def _load_acetone_calibration_metrics() -> dict:
    path = SCIENTIFIC_DIR / "Acetone" / "metrics" / "calibration_metrics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_figure4_row(gas: str) -> dict:
    if not FIGURE4_TABLE.exists():
        return {}
    try:
        df = pd.read_csv(FIGURE4_TABLE)
    except Exception:
        return {}
    if "Gas" not in df.columns:
        return {}
    mask = df["Gas"].astype(str).str.strip().str.lower() == gas.lower()
    if not mask.any():
        return {}
    row = df[mask].iloc[0].to_dict()
    return row


def _get_key_numbers() -> dict:
    reference_metrics = _load_reference_metrics()
    acetone_metrics = _load_acetone_calibration_metrics()
    figure4_row = _load_figure4_row("Acetone")

    centroid = (
        acetone_metrics.get("calibration_wavelength_shift", {})
        .get("centroid", {})
        if isinstance(acetone_metrics, dict)
        else {}
    )

    lod_this = centroid.get("lod_ppm")
    r2_this = centroid.get("r2")
    sens_this = centroid.get("slope")
    noise_this = centroid.get("noise_std")
    r2_cv = centroid.get("r2_cv")

    if not isinstance(lod_this, (int, float)) and figure4_row:
        lod_this = _first_float(figure4_row.get("LoD (ppm)"))
    if not isinstance(sens_this, (int, float)) and figure4_row:
        sens_this = _first_float(figure4_row.get("Sensitivity (nm/ppm)"))
    if not isinstance(r2_this, (int, float)) and figure4_row:
        r2_this = _first_float(figure4_row.get("R²"))
    if not isinstance(r2_cv, (int, float)) and figure4_row:
        r2_cv = _first_float(figure4_row.get("LOOCV R²"))

    lod_paper = DEFAULT_REFERENCE["lod"]
    sens_paper = DEFAULT_REFERENCE["sensitivity"]
    r2_paper = DEFAULT_REFERENCE["r2"]

    improvement_factor = None
    if isinstance(lod_paper, (int, float)) and isinstance(lod_this, (int, float)) and lod_this > 0:
        improvement_factor = lod_paper / lod_this

    if not isinstance(noise_this, (int, float)):
        if isinstance(lod_this, (int, float)) and isinstance(sens_this, (int, float)):
            noise_this = (lod_this * abs(sens_this)) / LOD_FACTOR

    noise_paper = (lod_paper * abs(sens_paper)) / LOD_FACTOR if lod_paper and sens_paper else None

    zno_thickness = reference_metrics.get("ZnO Thickness")

    return {
        "lod_this": lod_this,
        "lod_paper": lod_paper,
        "sens_this": sens_this,
        "sens_paper": sens_paper,
        "r2_this": r2_this,
        "r2_paper": r2_paper,
        "r2_cv": r2_cv,
        "noise_this": noise_this,
        "noise_paper": noise_paper,
        "improvement_factor": improvement_factor,
        "zno_thickness": zno_thickness,
    }

# Professional color palette
COLORS = {
    'primary': '#1E88E5', 'secondary': '#D81B60', 'accent': '#FFC107',
    'success': '#43A047', 'warning': '#FB8C00', 'error': '#E53935',
    'dark': '#212121', 'gray': '#757575', 'light_gray': '#E0E0E0',
    'white': '#FFFFFF', 'purple': '#7B1FA2', 'teal': '#00897B', 'indigo': '#3949AB',
}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11, 'axes.titleweight': 'bold',
    'axes.labelweight': 'bold', 'axes.spines.top': False, 'axes.spines.right': False,
})

def add_box(ax, x, y, w, h, color, text, fs=10):
    shadow = FancyBboxPatch((x+0.05, y-0.05), w, h, boxstyle="round,rounding_size=0.15",
                            facecolor='#00000022', edgecolor='none')
    ax.add_patch(shadow)
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,rounding_size=0.15",
                         facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2, text, fontsize=fs, fontweight='bold', ha='center', va='center', color='white')

def create_pipeline():
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')
    ax.text(7, 7.5, 'Data Processing Pipeline', fontsize=20, fontweight='bold', ha='center', color=COLORS['dark'])
    ax.plot([3, 11], [7.1, 7.1], color=COLORS['primary'], linewidth=3)

    key = _get_key_numbers()
    lod_this = key.get('lod_this')
    r2_this = key.get('r2_this')
    factor = key.get('improvement_factor')
    lod_text = f"{lod_this:.2f} ppm" if isinstance(lod_this, (int, float)) else "N/A"
    r2_text = f"{r2_this:.4f}" if isinstance(r2_this, (int, float)) else "N/A"
    factor_text = f"{factor:.0f}× improvement" if isinstance(factor, (int, float)) else "improvement"
    
    stages = [('Raw\nSpectra', 1.5, 5.5, COLORS['gray']), ('Preprocessing', 4, 5.5, COLORS['primary']),
              ('Temporal\nGating', 6.5, 5.5, COLORS['primary']), ('ROI\nDiscovery', 9, 5.5, COLORS['secondary']),
              ('Calibration', 11.5, 5.5, COLORS['success'])]
    for i, (t, x, y, c) in enumerate(stages):
        add_box(ax, x-0.9, y-0.6, 1.8, 1.2, c, t)
        circ = Circle((x-0.7, y+0.75), 0.2, facecolor=COLORS['dark'], edgecolor='white', lw=2)
        ax.add_patch(circ)
        ax.text(x-0.7, y+0.75, str(i+1), fontsize=9, fontweight='bold', ha='center', va='center', color='white')
    for i in range(len(stages)-1):
        ax.annotate('', xy=(stages[i+1][1]-0.9, 5.5), xytext=(stages[i][1]+0.9, 5.5),
                   arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    # Result box
    res = FancyBboxPatch((5, 2), 4, 1.5, boxstyle="round,rounding_size=0.2",
                         facecolor='#E8F5E9', edgecolor=COLORS['success'], linewidth=2)
    ax.add_patch(res)
    ax.text(7, 3.1, '✓ Key Achievement', fontsize=12, fontweight='bold', ha='center', color=COLORS['success'])
    ax.text(7, 2.5, f'LoD: {lod_text} ({factor_text}) | R² = {r2_text}', fontsize=10, ha='center', color=COLORS['dark'])
    ax.annotate('', xy=(7, 3.5), xytext=(7, 4.3), arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    plt.tight_layout()
    _save_figure('pipeline_flowchart')
    plt.close()
    print("✓ pipeline_flowchart.png")

def create_cnn():
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis('off')
    ax.text(7, 5.5, '1D-CNN Architecture', fontsize=18, fontweight='bold', ha='center', color=COLORS['dark'])
    
    # Layers
    layers = [(1.5, 2.2, COLORS['primary'], 'Input\n1000×1'), (3.5, 1.8, COLORS['secondary'], 'Conv\n32×7'),
              (5.5, 1.5, COLORS['purple'], 'Conv\n64×5'), (7.5, 1.2, COLORS['indigo'], 'Conv\n128×3'),
              (9.5, 0.8, COLORS['warning'], 'GAP'), (11, 0.6, COLORS['teal'], 'Dense\n64'),
              (12.5, 0.4, COLORS['success'], 'Out')]
    for x, h, c, t in layers:
        rect = Rectangle((x-0.4, 3-h/2), 0.8, h, facecolor=c, edgecolor='white', lw=2)
        ax.add_patch(rect)
        ax.text(x, 3-h/2-0.4, t, fontsize=8, ha='center', va='top', color=COLORS['dark'], fontweight='bold')
    for i in range(len(layers)-1):
        ax.annotate('', xy=(layers[i+1][0]-0.4, 3), xytext=(layers[i][0]+0.4, 3),
                   arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.5))
    
    ax.text(7, 1, 'Parameters: 44,033 | Loss: MSE | Optimizer: Adam | Early Stopping', fontsize=10,
           ha='center', bbox=dict(boxstyle='round', facecolor=COLORS['light_gray'], edgecolor=COLORS['gray']))
    plt.tight_layout()
    _save_figure('cnn_architecture')
    plt.close()
    print("✓ cnn_architecture.png")

def create_sensor():
    fig, ax = plt.subplots(figsize=(13, 7), facecolor='white')
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis('off')
    ax.text(6.5, 6.5, 'Optical Fiber Sensor System', fontsize=18, fontweight='bold', ha='center', color=COLORS['dark'])
    
    # Components
    add_box(ax, 0.3, 3.5, 2, 1.4, COLORS['warning'], 'Light Source\nHL-2000')
    add_box(ax, 10, 3.5, 2.2, 1.4, COLORS['success'], 'Spectrometer\nCCS200/M')
    add_box(ax, 10, 1.5, 2.2, 1.4, COLORS['purple'], 'Processing\nPython')
    
    # Fiber
    fiber_y = 4.2
    ax.add_patch(Rectangle((2.8, fiber_y-0.1), 1.5, 0.2, facecolor=COLORS['primary'], edgecolor='white'))
    ax.text(3.55, fiber_y+0.4, 'SMF', fontsize=8, ha='center', color=COLORS['primary'], fontweight='bold')
    ax.add_patch(Rectangle((4.5, fiber_y-0.2), 3, 0.4, facecolor=COLORS['secondary'], edgecolor='white'))
    ax.text(6, fiber_y, 'NCF+ZnO', fontsize=9, ha='center', va='center', color='white', fontweight='bold')
    ax.add_patch(Rectangle((7.7, fiber_y-0.1), 1.5, 0.2, facecolor=COLORS['primary'], edgecolor='white'))
    ax.text(8.45, fiber_y+0.4, 'SMF', fontsize=8, ha='center', color=COLORS['primary'], fontweight='bold')
    
    # Chamber
    chamber = FancyBboxPatch((4.2, 2.5), 3.6, 3, boxstyle="round,rounding_size=0.1",
                             facecolor='none', edgecolor=COLORS['gray'], linewidth=2, linestyle='--')
    ax.add_patch(chamber)
    ax.text(6, 2.2, 'Gas Chamber', fontsize=9, ha='center', color=COLORS['gray'], style='italic')
    ax.annotate('VOC+N₂', xy=(6, 5.5), xytext=(6, 6.1), fontsize=9, ha='center', color=COLORS['success'],
               fontweight='bold', arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))
    
    # Arrows
    ax.annotate('', xy=(2.8, fiber_y), xytext=(2.3, fiber_y), arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))
    ax.annotate('', xy=(10, fiber_y), xytext=(9.2, fiber_y), arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))
    ax.annotate('', xy=(11.1, 3.5), xytext=(11.1, 2.9), arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    ax.text(1.3, 1.8, 'T: 23.5±1.5°C\nRH: 55%', fontsize=9, ha='center', color=COLORS['gray'],
           bbox=dict(boxstyle='round', facecolor=COLORS['light_gray'], edgecolor=COLORS['gray']))
    plt.tight_layout()
    _save_figure('sensor_system_diagram')
    plt.close()
    print("✓ sensor_system_diagram.png")

def create_clinical():
    fig, ax = plt.subplots(figsize=(11, 6), facecolor='white')
    cats = ['Healthy', 'Diabetic', 'Ketoacidosis']
    ranges = [(0.2, 1.8), (1.25, 2.5), (2.5, 5.0)]
    colors = [COLORS['success'], COLORS['warning'], COLORS['error']]
    
    for i, (cat, (lo, hi), c) in enumerate(zip(cats, ranges, colors)):
        ax.barh(i, hi-lo, left=lo, height=0.5, color=c, alpha=0.85, edgecolor='white', lw=2)
        ax.text((lo+hi)/2, i, f'{lo}–{hi} ppm', fontsize=12, fontweight='bold', ha='center', va='center', color='white')
        ax.text(-0.2, i, cat, fontsize=11, fontweight='bold', ha='right', va='center', color=COLORS['dark'])
    
    key = _get_key_numbers()
    lod_this = key.get('lod_this')
    lod_paper = key.get('lod_paper')
    if not (isinstance(lod_this, (int, float)) and isinstance(lod_paper, (int, float))):
        print("! Skipping clinical_thresholds: missing LoD metrics")
        plt.close(fig)
        return
    lod_this_val = float(lod_this)
    lod_paper_val = float(lod_paper)

    ax.axvline(lod_this_val, color=COLORS['primary'], lw=4, label=f'This Work: {lod_this_val:.2f} ppm')
    ax.axvline(lod_paper_val, color=COLORS['gray'], lw=3, linestyle='--', alpha=0.7, label=f'Reference: {lod_paper_val:.2f} ppm')
    ax.annotate(f'Our LoD\n{lod_this_val:.2f} ppm', xy=(lod_this_val, 2.3), xytext=(0.7, 2.5), fontsize=10, fontweight='bold',
               color=COLORS['primary'], arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.8, 3)
    ax.set_xlabel('Breath Acetone (ppm)', fontsize=12, fontweight='bold')
    ax.set_title('Clinical Thresholds vs Detection Capability', fontsize=14, fontweight='bold')
    ax.set_yticks([]); ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--'); ax.spines['left'].set_visible(False)
    plt.tight_layout()
    _save_figure('clinical_thresholds')
    plt.close()
    print("✓ clinical_thresholds.png")


def _load_cross_sensitivity_matrix() -> tuple[list[str], list[str], np.ndarray] | None:
    path = DATA_DIR / "cross_sensitivity_matrix.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df = df.dropna(how="all")
    if df.empty:
        return None

    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    if {"sensor for", "target voc", "response (%)"}.issubset(set(cols_lower.keys())):
        df2 = df.rename(
            columns={
                cols_lower["sensor for"]: "Sensor For",
                cols_lower["target voc"]: "Target VOC",
                cols_lower["response (%)"]: "Response (%)",
            }
        )
        df2["Sensor For"] = df2["Sensor For"].astype(str).str.strip()
        df2["Target VOC"] = df2["Target VOC"].astype(str).str.strip()

        mat = (
            df2.pivot_table(index="Sensor For", columns="Target VOC", values="Response (%)", aggfunc="mean")
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        row_labels = [str(v) for v in mat.index.tolist()]
        col_labels = [str(v) for v in mat.columns.tolist()]
        data = mat.to_numpy(dtype=float)
        if data.size == 0:
            return None
        return row_labels, col_labels, data

    if df.shape[1] >= 2:
        first = df.columns[0]
        df3 = df.copy()
        df3[first] = df3[first].astype(str).str.strip()
        df3 = df3.set_index(first)
        df3.columns = [str(c).strip() for c in df3.columns]
        row_labels = [str(v) for v in df3.index.tolist()]
        col_labels = [str(v) for v in df3.columns.tolist()]
        data = df3.to_numpy(dtype=float)
        if data.size == 0:
            return None
        return row_labels, col_labels, data

    return None


def create_heatmap():
    fig, ax = plt.subplots(figsize=(9, 7), facecolor='white')
    loaded = _load_cross_sensitivity_matrix()
    if loaded is None:
        print("! Skipping cross_sensitivity_heatmap: missing data/cross_sensitivity_matrix.csv")
        plt.close(fig)
        return

    row_labels, col_labels, data = loaded
    im = ax.imshow(data, cmap='YlOrRd', vmin=0, vmax=100)
    n = int(data.shape[0])
    m = int(data.shape[1])

    for i in range(n):
        for j in range(m):
            val = float(data[i, j]) if np.isfinite(data[i, j]) else float('nan')
            c = 'white' if np.isfinite(val) and val > 50 else COLORS['dark']
            text = f'{val:.1f}%' if np.isfinite(val) else "n/a"
            ax.text(j, i, text, ha='center', va='center', color=c, fontsize=9,
                    fontweight='bold' if i == j else 'normal')

    ax.set_xticks(range(m)); ax.set_yticks(range(n))
    ax.set_xticklabels(col_labels[:m], fontsize=10, fontweight='bold', rotation=45, ha='right')
    ax.set_yticklabels(row_labels[:n], fontsize=10, fontweight='bold')
    ax.set_xlabel('Target VOC', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sensor For', fontsize=11, fontweight='bold')
    ax.set_title('Cross-Sensitivity Matrix (%)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Response (%)')
    ax.add_patch(Rectangle((-0.5, -0.5), m, 1, fill=False, edgecolor=COLORS['primary'], lw=3))
    plt.tight_layout()
    _save_figure('cross_sensitivity_heatmap')
    plt.close()
    print("✓ cross_sensitivity_heatmap.png")

def create_tradeoff():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor='white')
    rois = ['Literature\n675-689nm', 'This Work\n580-590nm']

    key = _get_key_numbers()
    sens_paper = key.get('sens_paper')
    sens_this = key.get('sens_this')
    noise_paper = key.get('noise_paper')
    noise_this = key.get('noise_this')
    lod_paper = key.get('lod_paper')
    lod_this = key.get('lod_this')
    factor = key.get('improvement_factor')

    required = [sens_paper, sens_this, noise_paper, noise_this, lod_paper, lod_this]
    if not all(isinstance(v, (int, float)) for v in required):
        print("! Skipping lod_tradeoff_diagram: missing metrics")
        plt.close(fig)
        return

    sens_vals = [float(sens_paper), float(sens_this)]
    noise_vals = [float(noise_paper), float(noise_this)]
    lod_vals = [float(lod_paper), float(lod_this)]
    
    ax1 = axes[0]
    ax1.bar(rois, sens_vals, color=[COLORS['gray'], COLORS['primary']], edgecolor='white', lw=2, width=0.5)
    ax1.set_ylabel('Sensitivity (|S|, nm/ppm)'); ax1.set_title('① Sensitivity')
    ax1.set_ylim(0, max(sens_vals) * 1.25 if max(sens_vals) > 0 else 0.1)
    for i, v in enumerate(sens_vals):
        ax1.text(i, v+0.003, f'{v:.4f}', ha='center', fontweight='bold')
    
    ax2 = axes[1]
    ax2.bar(rois, noise_vals, color=[COLORS['gray'], COLORS['primary']], edgecolor='white', lw=2, width=0.5)
    ax2.set_ylabel('Noise σ (nm)'); ax2.set_title('② Noise')
    ax2.set_ylim(0, max(noise_vals) * 1.25 if max(noise_vals) > 0 else 0.1)
    for i, v in enumerate(noise_vals):
        ax2.text(i, v+0.003, f'{v:.4f}', ha='center', fontweight='bold')
    if noise_vals[0] and noise_vals[1] and noise_vals[0] > 0:
        red = (1.0 - (noise_vals[1] / noise_vals[0])) * 100.0
        ax2.annotate(f'{red:.0f}%↓', xy=(1, noise_vals[1]), xytext=(1.2, 0.04), fontsize=10, color=COLORS['success'],
                    fontweight='bold', arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))
    
    ax3 = axes[2]
    ax3.bar(rois, lod_vals, color=[COLORS['gray'], COLORS['success']], edgecolor='white', lw=2, width=0.5)
    ax3.set_ylabel('LoD (ppm)'); ax3.set_title('③ Detection Limit'); ax3.set_ylim(0, 4)
    for i, v in enumerate(lod_vals):
        ax3.text(i, v+0.1, f'{v:.2f}', ha='center', fontsize=12, fontweight='bold')
    ax3.axhline(1.8, color=COLORS['warning'], ls='--', lw=2)
    if isinstance(factor, (int, float)):
        ax3.annotate(f'{factor:.0f}× better!', xy=(1, lod_vals[1]), xytext=(1.2, 1.5), fontsize=11, color=COLORS['success'],
                    fontweight='bold', arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))
    
    fig.suptitle('Sensitivity-Noise Trade-off: Why Lower Noise Wins', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure('lod_tradeoff_diagram')
    plt.close()
    print("✓ lod_tradeoff_diagram.png")

def create_baseline_vs_optimized_roi():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), facecolor='white')

    key = _get_key_numbers()
    sens_paper = key.get('sens_paper')
    sens_this = key.get('sens_this')
    r2_paper = key.get('r2_paper')
    r2_this = key.get('r2_this')
    lod_paper = key.get('lod_paper')
    lod_this = key.get('lod_this')

    required = [sens_paper, sens_this, r2_paper, r2_this, lod_paper, lod_this]
    if not all(isinstance(v, (int, float)) for v in required):
        print("! Skipping baseline_vs_optimized_roi: missing metrics")
        plt.close(fig)
        return

    labels = ['Baseline\n(Literature ROI)', 'Optimized\n(Data-driven ROI)']

    sens_vals = [float(sens_paper), float(sens_this)]
    r2_vals = [float(r2_paper), float(r2_this)]
    lod_vals = [float(lod_paper), float(lod_this)]

    ax = axes[0]
    ax.bar(labels, sens_vals, color=[COLORS['gray'], COLORS['primary']], edgecolor='white', lw=2, width=0.55)
    ax.set_title('Sensitivity (|Δλ slope|)')
    ax.set_ylabel('nm/ppm')
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    for i, v in enumerate(sens_vals):
        ax.text(i, v + max(sens_vals) * 0.04, f'{v:.4f}', ha='center', fontweight='bold')

    ax = axes[1]
    ax.bar(labels, r2_vals, color=[COLORS['gray'], COLORS['success']], edgecolor='white', lw=2, width=0.55)
    ax.set_title('Goodness of Fit (R²)')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    for i, v in enumerate(r2_vals):
        ax.text(i, v + 0.03, f'{v:.4f}', ha='center', fontweight='bold')

    ax = axes[2]
    ax.bar(labels, lod_vals, color=[COLORS['gray'], COLORS['warning']], edgecolor='white', lw=2, width=0.55)
    ax.set_title('Analytical LoD (kσ/|S|)')
    ax.set_ylabel('ppm')
    ax.set_ylim(0, max(lod_vals) * 1.2)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    for i, v in enumerate(lod_vals):
        ax.text(i, v + max(lod_vals) * 0.04, f'{v:.2f}', ha='center', fontweight='bold')

    fig.suptitle('Acetone Performance: Baseline vs Optimized ROI', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure('baseline_vs_optimized_roi')
    plt.close()
    print("✓ baseline_vs_optimized_roi.png")


def create_roi_discovery_annotated():
    base_img_path = SCIENTIFIC_DIR / "Acetone" / "plots" / "roi_scan_results.png"
    if not base_img_path.exists():
        return

    df_10 = _load_acetone_canonical_spectra(10.0)
    if df_10.empty or not {"wavelength", "absorbance"}.issubset(set(df_10.columns)):
        return

    x = df_10["wavelength"].to_numpy(dtype=float)
    y = df_10["absorbance"].to_numpy(dtype=float)

    img = plt.imread(base_img_path)

    fig = plt.figure(figsize=(13.5, 6.2), facecolor='white')
    gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1.55, 1.0], height_ratios=[1, 1], wspace=0.15, hspace=0.25)

    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(img)
    ax_img.axis('off')
    ax_img.set_title('ROI Discovery (annotated)', fontsize=13, fontweight='bold')

    ax_img.text(
        0.02,
        0.98,
        'Boxes mark ROIs discussed in this work:\n- Traditional manual ROI (675–689 nm)\n- Discovered optimal ROI (580–590 nm)',
        transform=ax_img.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=COLORS['light_gray']),
    )

    ax_a = fig.add_subplot(gs[0, 1])
    m = (x >= 575) & (x <= 595)
    ax_a.plot(x[m], y[m], color=COLORS['primary'], lw=2)
    ax_a.axvspan(580, 590, color=COLORS['primary'], alpha=0.15)
    ax_a.set_title('Local morphology near 580–590 nm', fontsize=11, fontweight='bold')
    ax_a.set_xlabel('Wavelength (nm)')
    ax_a.set_ylabel('Absorbance (a.u.)')
    ax_a.grid(alpha=0.25, linestyle='--')

    ax_b = fig.add_subplot(gs[1, 1])
    m = (x >= 665) & (x <= 700)
    ax_b.plot(x[m], y[m], color=COLORS['gray'], lw=2)
    ax_b.axvspan(675, 689, color=COLORS['gray'], alpha=0.15)
    ax_b.set_title('Local morphology near 675–689 nm', fontsize=11, fontweight='bold')
    ax_b.set_xlabel('Wavelength (nm)')
    ax_b.set_ylabel('Absorbance (a.u.)')
    ax_b.grid(alpha=0.25, linestyle='--')

    fig.suptitle('Data-Driven ROI Discovery: Traditional vs Discovered Window', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure('roi_discovery_annotated')
    plt.close()
    print("✓ roi_discovery_annotated.png")


def create_virtual_sensor_array():
    df = _load_multigas_results()
    if df.empty:
        return

    if "Gas" in df.columns and "VOC" not in df.columns:
        df = df.rename(columns={"Gas": "VOC"})

    if "ROI (nm)" in df.columns and "Optimal ROI (nm)" not in df.columns:
        df = df.rename(columns={"ROI (nm)": "Optimal ROI (nm)"})

    if not {"VOC", "Optimal ROI (nm)", "Sensitivity (nm/ppm)", "LoD (ppm)"}.issubset(set(df.columns)):
        return

    rows = []
    for _, row in df.iterrows():
        roi = _parse_roi_nm(str(row.get("Optimal ROI (nm)", "")))
        if not roi:
            continue
        lo, hi = roi
        center = 0.5 * (lo + hi)
        width = hi - lo
        rows.append(
            {
                "VOC": str(row.get("VOC", "")).strip(),
                "roi_center_nm": center,
                "roi_width_nm": width,
                "lod_ppm": _first_float(row.get("LoD (ppm)")),
                "sensitivity": _first_float(row.get("Sensitivity (nm/ppm)")),
                "r2": _first_float(row.get("R²")),
                "spearman": _first_float(row.get("Spearman ρ")),
            }
        )

    tidy = pd.DataFrame(rows)
    tidy = tidy.dropna(subset=["roi_center_nm", "lod_ppm", "sensitivity"]) if not tidy.empty else tidy
    if tidy.empty:
        return

    tidy = tidy.sort_values(by=["roi_center_nm", "VOC"], ascending=[True, True])

    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor='white')

    voc_order = tidy["VOC"].astype(str).tolist()
    unique_vocs = list(dict.fromkeys(voc_order))
    y_positions = {voc: i for i, voc in enumerate(unique_vocs)}
    y = tidy["VOC"].astype(str).map(y_positions).astype(float).values

    x = tidy["roi_center_nm"].values
    lod = tidy["lod_ppm"].values
    sens = tidy["sensitivity"].values

    size = 400 * (sens / max(sens)) if np.isfinite(sens).all() and max(sens) > 0 else 180

    sc = ax.scatter(
        x,
        y,
        s=size,
        c=lod,
        cmap="viridis_r",
        edgecolor="white",
        linewidth=1.5,
        alpha=0.9,
    )

    for _, row in tidy.iterrows():
        voc = str(row["VOC"])
        y0 = y_positions.get(voc)
        if y0 is None:
            continue
        ax.text(
            float(row["roi_center_nm"]) + 6,
            float(y0) + 0.05,
            f"{float(row['lod_ppm']):.2f} ppm",
            fontsize=9,
            color=COLORS['dark'],
        )

    ax.set_yticks(range(len(unique_vocs)))
    ax.set_yticklabels(unique_vocs, fontweight='bold')
    ax.set_xlabel("Optimal ROI center (nm)", fontweight='bold')
    ax.set_title("Virtual Sensor Array: VOC-Specific Optimal Spectral Windows", fontweight='bold')
    ax.grid(axis='x', alpha=0.25, linestyle='--')

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Analytical LoD (ppm)", fontweight='bold')

    plt.tight_layout()
    _save_figure('virtual_sensor_array')
    plt.close()
    print("✓ virtual_sensor_array.png")


def _centroid_from_absorbance(wavelength: np.ndarray, absorbance: np.ndarray, roi: tuple[float, float]) -> float:
    wl = np.asarray(wavelength, dtype=float)
    a = np.asarray(absorbance, dtype=float)
    m = (wl >= roi[0]) & (wl <= roi[1])
    if not np.any(m):
        return float('nan')
    wl_roi = wl[m]
    a_roi = a[m]
    a_min = float(np.min(a_roi))
    if np.isfinite(a_min):
        a_roi = a_roi - a_min
    a_roi = np.where(np.isfinite(a_roi), a_roi, 0.0)
    s = float(np.sum(a_roi))
    if s <= 0:
        return float('nan')
    return float(np.sum(wl_roi * a_roi) / s)


def create_virtual_sensor_array_radar():
    df = _load_multigas_results()
    if df.empty:
        return

    if "Gas" in df.columns and "VOC" not in df.columns:
        df = df.rename(columns={"Gas": "VOC"})
    if "ROI (nm)" in df.columns and "Optimal ROI (nm)" not in df.columns:
        df = df.rename(columns={"ROI (nm)": "Optimal ROI (nm)"})

    roi_map: dict[str, tuple[float, float]] = {
        "Acetone": (580.0, 590.0),
        "Ethanol": (515.0, 525.0),
        "Toluene": (830.0, 850.0),
    }
    for voc in list(roi_map.keys()):
        match = df[df["VOC"].astype(str).str.lower() == voc.lower()]
        if not match.empty:
            parsed = _parse_roi_nm(str(match.iloc[0].get("Optimal ROI (nm)", "")))
            if parsed:
                roi_map[voc] = parsed

    axes = [
        (f"{roi_map['Acetone'][0]:.0f}–{roi_map['Acetone'][1]:.0f} nm", roi_map["Acetone"]),
        (f"{roi_map['Ethanol'][0]:.0f}–{roi_map['Ethanol'][1]:.0f} nm", roi_map["Ethanol"]),
        (f"{roi_map['Toluene'][0]:.0f}–{roi_map['Toluene'][1]:.0f} nm", roi_map["Toluene"]),
    ]

    vocs = ["Acetone", "Ethanol", "Toluene"]
    concs = np.array([1.0, 3.0, 5.0, 10.0], dtype=float)

    sens: dict[str, list[float]] = {}
    for voc in vocs:
        sens_vals = []
        spectra = {}
        for c in concs:
            d = _load_voc_canonical_spectra(voc, float(c))
            if d.empty or not {"wavelength", "absorbance"}.issubset(set(d.columns)):
                return
            spectra[float(c)] = d

        wl = spectra[1.0]["wavelength"].to_numpy(dtype=float)
        for _, roi in axes:
            centroids = []
            for c in concs:
                a = spectra[float(c)]["absorbance"].to_numpy(dtype=float)
                feat = _apply_feature_engineering(a)
                centroids.append(_centroid_from_weights(wl, feat, roi))
            centroids = np.asarray(centroids, dtype=float)
            if not np.isfinite(centroids).all():
                sens_vals.append(float('nan'))
                continue
            delta = centroids - float(centroids[0])
            fit = np.polyfit(concs, delta, 1)
            sens_vals.append(abs(float(fit[0])))
        sens[voc] = sens_vals

    mat = np.array([sens[v] for v in vocs], dtype=float)
    if not np.isfinite(mat).all():
        return
    denom = np.max(mat, axis=0)
    denom = np.where(denom > 0, denom, 1.0)
    mat_n = mat / denom

    labels = [a[0] for a in axes]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(8.6, 6.6), facecolor='white')
    ax = plt.subplot(111, polar=True)

    color_map = {
        "Acetone": COLORS['primary'],
        "Ethanol": COLORS['secondary'],
        "Toluene": COLORS['warning'],
    }
    for i, voc in enumerate(vocs):
        vals = mat_n[i]
        vals = np.concatenate([vals, vals[:1]])
        ax.plot(angles, vals, lw=2.5, color=color_map.get(voc, COLORS['dark']), label=voc)
        ax.fill(angles, vals, color=color_map.get(voc, COLORS['dark']), alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title('Wavelength-Resolved Selectivity: Virtual Sensor Fingerprints', fontsize=13, fontweight='bold', pad=18)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.10), fontsize=10)

    plt.tight_layout()
    _save_figure('virtual_sensor_array_radar')
    plt.close()
    print("✓ virtual_sensor_array_radar.png")


def create_calibration_robustness_tests():
    acetone = _load_acetone_calibration_metrics()
    centroid = (
        acetone.get("calibration_wavelength_shift", {}).get("centroid", {})
        if isinstance(acetone, dict)
        else {}
    )
    if not centroid:
        return

    x = np.array(centroid.get("concentrations", []), dtype=float)
    y = np.array(centroid.get("delta_lambda", []), dtype=float)
    if x.size < 3 or y.size != x.size:
        return

    slope = float(centroid.get("slope")) if isinstance(centroid.get("slope"), (int, float)) else None
    intercept = float(centroid.get("intercept")) if isinstance(centroid.get("intercept"), (int, float)) else None
    noise_std = float(centroid.get("noise_std")) if isinstance(centroid.get("noise_std"), (int, float)) else None

    if slope is None or intercept is None:
        fit = np.polyfit(x, y, 1)
        slope, intercept = float(fit[0]), float(fit[1])

    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2_obs = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse_obs = float(np.sqrt(ss_res / max(1, (len(x) - 2))))

    rng = np.random.default_rng(7)
    n_perm = 5000
    r2_perm = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        x_perm = rng.permutation(x)
        fit_p = np.polyfit(x_perm, y, 1)
        y_p = fit_p[0] * x_perm + fit_p[1]
        ss_res_p = float(np.sum((y - y_p) ** 2))
        ss_tot_p = ss_tot
        r2_perm[i] = 1.0 - (ss_res_p / ss_tot_p) if ss_tot_p > 0 else 0.0

    p_perm = float(np.mean(r2_perm >= r2_obs))

    n_boot = 5000
    slope_boot = np.empty(n_boot, dtype=float)
    lod_boot = np.empty(n_boot, dtype=float)

    sigma = noise_std if isinstance(noise_std, (int, float)) and noise_std > 0 else rmse_obs
    for i in range(n_boot):
        eps = rng.normal(0.0, sigma, size=len(y))
        y_star = y_hat + eps
        fit_b = np.polyfit(x, y_star, 1)
        slope_b = float(fit_b[0])
        intercept_b = float(fit_b[1])
        y_b = slope_b * x + intercept_b
        rmse_b = float(np.sqrt(np.mean((y_star - y_b) ** 2)))
        slope_boot[i] = slope_b
        lod_boot[i] = (LOD_FACTOR * rmse_b / abs(slope_b)) if slope_b != 0 else np.nan

    lod_boot = lod_boot[np.isfinite(lod_boot)]
    if lod_boot.size == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')

    ax = axes[0]
    ax.hist(r2_perm, bins=35, color=COLORS['light_gray'], edgecolor='white')
    ax.axvline(r2_obs, color=COLORS['primary'], lw=3, label=f"Observed R²={r2_obs:.4f}")
    ax.set_title("Permutation Test (null R² distribution)")
    ax.set_xlabel("R² under shuffled concentrations")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25, linestyle='--')
    ax.legend(fontsize=10)
    ax.text(
        0.02,
        0.95,
        f"Permutation fraction ≥ observed: {p_perm:.4g} (heuristic)",
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['light_gray']),
    )

    ax = axes[1]
    ax.hist(lod_boot, bins=35, color=COLORS['warning'], alpha=0.8, edgecolor='white')
    lod_obs = (LOD_FACTOR * (noise_std if noise_std else rmse_obs) / abs(slope)) if slope != 0 else np.nan
    if np.isfinite(lod_obs):
        ax.axvline(lod_obs, color=COLORS['dark'], lw=3, label=f"LoD={lod_obs:.3f} ppm")
    ci_lo, ci_hi = np.quantile(lod_boot, [0.025, 0.975])
    ax.set_title("Bootstrap LoD distribution (noise-resampled)")
    ax.set_xlabel("LoD (ppm)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25, linestyle='--')
    ax.legend(fontsize=10)
    ax.text(
        0.02,
        0.95,
        f"95% CI: [{ci_lo:.3f}, {ci_hi:.3f}] ppm",
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['light_gray']),
    )

    fig.suptitle("Calibration Robustness Checks (Acetone, optimized ROI)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure('calibration_robustness')
    plt.close()
    print("✓ calibration_robustness.png")


def create_detectability_analysis():
    acetone = _load_acetone_calibration_metrics()
    centroid = (
        acetone.get("calibration_wavelength_shift", {}).get("centroid", {})
        if isinstance(acetone, dict)
        else {}
    )
    if not centroid:
        return

    slope = centroid.get("slope")
    intercept = centroid.get("intercept")
    sigma = centroid.get("noise_std")
    if not (isinstance(slope, (int, float)) and isinstance(intercept, (int, float)) and isinstance(sigma, (int, float))):
        return
    if sigma <= 0 or slope == 0:
        return

    rng = np.random.default_rng(11)
    n_mc = 20000

    c0 = np.zeros(n_mc, dtype=float)
    y0 = slope * c0 + intercept + rng.normal(0.0, sigma, size=n_mc)
    c_hat0 = (y0 - intercept) / slope
    thr = float(np.quantile(c_hat0, 0.95))

    c_grid = np.linspace(0.0, 1.0, 21)
    p_det = []
    for c in c_grid:
        y = slope * c + intercept + rng.normal(0.0, sigma, size=n_mc)
        c_hat = (y - intercept) / slope
        p_det.append(float(np.mean(c_hat > thr)))
    p_det = np.array(p_det)

    target = 0.95
    c_95 = None
    above = np.where(p_det >= target)[0]
    if above.size:
        c_95 = float(c_grid[int(above[0])])

    lod_est = (LOD_FACTOR * sigma / abs(slope))

    fig, ax = plt.subplots(figsize=(10.5, 5.5), facecolor='white')
    ax.plot(c_grid, p_det, color=COLORS['primary'], lw=3, marker='o', ms=5)
    ax.axhline(0.95, color=COLORS['gray'], lw=2, linestyle='--', alpha=0.8)
    ax.axvline(lod_est, color=COLORS['warning'], lw=3, label=f"LoD (kσ/|S|) = {lod_est:.2f} ppm")
    if c_95 is not None:
        ax.axvline(c_95, color=COLORS['success'], lw=3, linestyle='--', label=f"95% detectability ≈ {c_95:.2f} ppm")

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, float(c_grid.max()))
    ax.set_xlabel("Injected concentration (ppm)", fontweight='bold')
    ax.set_ylabel("Detection probability (FPR fixed at 5%)", fontweight='bold')
    ax.set_title("Signal-in-Noise Detectability Using Measured Residual Noise", fontweight='bold')
    ax.grid(alpha=0.25, linestyle='--')
    ax.legend(fontsize=10, loc='lower right')

    ax.text(
        0.02,
        0.98,
        f"Threshold (95% blank): {thr:.3f} ppm\nNoise σ: {sigma:.4f} nm",
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['light_gray']),
    )

    plt.tight_layout()
    _save_figure('detectability_analysis')
    plt.close()
    print("✓ detectability_analysis.png")


def _apply_feature_engineering(absorbance: np.ndarray) -> np.ndarray:
    y = np.asarray(absorbance, dtype=float)
    if y.size < 7:
        return y

    win = 11
    if win >= y.size:
        win = y.size - 1 if (y.size % 2 == 0) else y.size
    if win < 3:
        return y
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return y

    deriv = savgol_filter(y, window_length=win, polyorder=2, deriv=1)
    conv = np.convolve(y, deriv, mode='same')
    mu = float(np.mean(conv))
    sd = float(np.std(conv))
    return (conv - mu) / (sd + 1e-12)


def _centroid_from_weights(wavelength: np.ndarray, weights: np.ndarray, roi: tuple[float, float]) -> float:
    wl = np.asarray(wavelength, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = (wl >= roi[0]) & (wl <= roi[1])
    if not np.any(m):
        return float('nan')
    wl_roi = wl[m]
    w_roi = w[m].copy()
    w_min = float(np.min(w_roi))
    if np.isfinite(w_min):
        w_roi = w_roi - w_min
    w_roi = np.where(np.isfinite(w_roi), w_roi, 0.0)
    s = float(np.sum(w_roi))
    if s <= 0:
        w_roi = np.abs(w[m])
        w_roi = np.where(np.isfinite(w_roi), w_roi, 0.0)
        s = float(np.sum(w_roi))
    if s <= 0:
        return float('nan')
    return float(np.sum(wl_roi * w_roi) / s)


def _shift_spectrum(wavelength: np.ndarray, values: np.ndarray, delta_lambda_nm: float) -> np.ndarray:
    wl = np.asarray(wavelength, dtype=float)
    y = np.asarray(values, dtype=float)
    xq = wl - float(delta_lambda_nm)
    return np.interp(xq, wl, y, left=y[0], right=y[-1])


def create_synthetic_injection_validation():
    key = _load_acetone_calibration_metrics()
    centroid = (
        key.get("calibration_wavelength_shift", {}).get("centroid", {})
        if isinstance(key, dict)
        else {}
    )
    if not centroid:
        return

    roi = centroid.get("roi_range")
    if not (isinstance(roi, list) and len(roi) == 2):
        return
    roi_tuple = (float(roi[0]), float(roi[1]))

    slope = centroid.get("slope")
    sigma_nm = centroid.get("noise_std")
    if not (isinstance(slope, (int, float)) and isinstance(sigma_nm, (int, float))):
        return
    if sigma_nm <= 0 or slope == 0:
        return

    delta_lambda_inject = LOD_FACTOR * float(sigma_nm)
    c_equiv = float(delta_lambda_inject) / float(slope)

    rng = np.random.default_rng(19)
    n = 20000
    d0 = rng.normal(0.0, float(sigma_nm), size=n)
    d1 = rng.normal(float(delta_lambda_inject), float(sigma_nm), size=n)

    thr = float(np.quantile(d0, 0.95))
    p_det = float(np.mean(d1 > thr))
    snr = float(delta_lambda_inject / float(sigma_nm))

    fig, ax = plt.subplots(figsize=(10.8, 5.5), facecolor='white')
    ax.hist(d0, bins=55, color=COLORS['light_gray'], alpha=0.9, edgecolor='white', label='Reference (noise-only)')
    ax.hist(d1, bins=55, color=COLORS['primary'], alpha=0.55, edgecolor='white', label=f'Injected: Δλ = {LOD_FACTOR:.1f}σ (≈{c_equiv:.2f} ppm)')
    ax.axvline(thr, color=COLORS['dark'], lw=2.5, linestyle='--', label='5% FPR threshold')
    ax.set_title('Synthetic Noise-Injection Validation (Δλ Domain)', fontweight='bold')
    ax.set_xlabel('Simulated centroid shift Δλ (nm) relative to reference')
    ax.set_ylabel('Count')
    ax.grid(alpha=0.25, linestyle='--')
    ax.legend(fontsize=10)

    ax.text(
        0.02,
        0.98,
        f"Injected Δλ = {LOD_FACTOR:.1f}σ = {delta_lambda_inject:.4f} nm\n" \
        f"σ (target) = {sigma_nm:.4f} nm\n" \
        f"Estimated SNR = {snr:.2f}\n" \
        f"Detection @5% FPR: {p_det:.2f}",
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['light_gray']),
    )

    plt.tight_layout()
    _save_figure('synthetic_injection_validation')
    plt.close()
    print("✓ synthetic_injection_validation.png")

def create_feature_eng():
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')
    ax.text(7, 7.5, 'Spectral Feature Engineering', fontsize=18, fontweight='bold', ha='center', color=COLORS['dark'])
    
    steps = [('① Raw I(λ)', 1.5, 5.5, COLORS['gray']), ('② T=I/I₀', 4, 5.5, COLORS['primary']),
             ('③ A=-log(T)', 6.5, 5.5, COLORS['primary']), ('④ dA/dλ', 9, 5.5, COLORS['secondary']),
             ('⑤ Convolution', 11.5, 5.5, COLORS['purple'])]
    for t, x, y, c in steps:
        add_box(ax, x-1, y-0.5, 2, 1, c, t, fs=10)
    for i in range(len(steps)-1):
        ax.annotate('', xy=(steps[i+1][1]-1, 5.5), xytext=(steps[i][1]+1, 5.5),
                   arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    # Benefits
    benefits = [('Baseline Drift', 'Suppressed', COLORS['primary']),
                ('Dynamic Range', 'Compressed', COLORS['secondary']),
                ('SNR', 'Improved', COLORS['success'])]
    for i, (t, v, c) in enumerate(benefits):
        x = 2.5 + i*4
        box = FancyBboxPatch((x-1.2, 2.5), 2.4, 1.5, boxstyle="round,rounding_size=0.15",
                             facecolor='white', edgecolor=c, linewidth=2)
        ax.add_patch(box)
        ax.text(x, 3.5, t, fontsize=10, fontweight='bold', ha='center', color=c)
        ax.text(x, 2.9, v, fontsize=9, ha='center', color=COLORS['dark'])
    
    plt.tight_layout()
    _save_figure('feature_engineering_workflow')
    plt.close()
    print("✓ feature_engineering_workflow.png")

def create_noise():
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    path = DATA_DIR / "noise_robustness.csv"
    if not path.exists():
        print("! Skipping noise_robustness_chart: missing data/noise_robustness.csv")
        plt.close(fig)
        return

    df = pd.read_csv(path)
    df = df.dropna(how="all")
    if df.empty:
        plt.close(fig)
        return

    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    required = {"snr", "standard", "ml"}
    if not required.issubset(set(cols_lower.keys())):
        print("! Skipping noise_robustness_chart: noise_robustness.csv must have columns SNR, Standard, ML")
        plt.close(fig)
        return

    snr = df[cols_lower["snr"]].astype(str).tolist()
    std = df[cols_lower["standard"]].astype(float).to_list()
    ml = df[cols_lower["ml"]].astype(float).to_list()
    if not (len(snr) == len(std) == len(ml)):
        print("! Skipping noise_robustness_chart: inconsistent column lengths")
        plt.close(fig)
        return
    x = np.arange(len(snr)); w = 0.35
    ax.bar(x-w/2, std, w, label='Standard', color=COLORS['gray'], edgecolor='white', lw=2)
    ax.bar(x+w/2, ml, w, label='ML-Enhanced', color=COLORS['success'], edgecolor='white', lw=2)
    for i, (s, m) in enumerate(zip(std, ml)):
        imp = (1-m/s)*100
        ax.annotate(f'{imp:.0f}%↓', xy=(i+w/2, m), xytext=(i+w/2+0.1, m*0.5), fontsize=8,
                   color=COLORS['success'], fontweight='bold', arrowprops=dict(arrowstyle='->', color=COLORS['success']))
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(snr, fontweight='bold')
    ax.set_ylabel('MSE', fontweight='bold'); ax.set_xlabel('SNR', fontweight='bold')
    ax.set_title('Noise Robustness Comparison', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3, ls='--')
    plt.tight_layout()
    _save_figure('noise_robustness_chart')
    plt.close()
    print("✓ noise_robustness_chart.png")

def create_roi():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    ax1 = axes[0]
    wl = np.linspace(500, 900, 500)
    np.random.seed(42)
    sp = 0.5 + 0.35*np.exp(-((wl-585)**2)/(2*12**2)) + 0.45*np.exp(-((wl-680)**2)/(2*18**2)) + 0.02*np.random.randn(500)
    ax1.plot(wl, sp, color=COLORS['dark'], lw=1.5)
    ax1.axvspan(580, 590, alpha=0.4, color=COLORS['success'], label='Our ROI')
    ax1.axvspan(675, 689, alpha=0.3, color=COLORS['gray'], label='Literature ROI')
    ax1.set_xlabel('Wavelength (nm)'); ax1.set_ylabel('Signal'); ax1.set_title('ROI Selection')
    ax1.legend(); ax1.grid(alpha=0.3, ls='--')
    
    ax2 = axes[1]
    X, Y = np.meshgrid(np.linspace(520, 880, 50), np.linspace(5, 30, 25))
    np.random.seed(42)
    Z = np.clip(3.5 - 3*np.exp(-((X-585)**2)/(2*35**2) - ((Y-10)**2)/(2*10**2)) + 0.2*np.random.randn(*X.shape), 0.15, 4)
    cf = ax2.contourf(X, Y, Z, levels=15, cmap='RdYlGn_r')
    ax2.plot(585, 10, 'w*', ms=20, mec='black', mew=2)
    ax2.annotate('Optimal\n0.17 ppm', xy=(585, 10), xytext=(650, 22), fontsize=9, color='white', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white', lw=2),
                bbox=dict(boxstyle='round', facecolor=COLORS['success'], edgecolor='white'))
    plt.colorbar(cf, ax=ax2, label='LoD (ppm)')
    ax2.set_xlabel('ROI Center (nm)'); ax2.set_ylabel('ROI Width (nm)')
    ax2.set_title('ROI Optimization (385 candidates)')
    plt.tight_layout()
    _save_figure('roi_optimization_concept')
    plt.close()
    print("✓ roi_optimization_concept.png")

def create_abstract():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')

    key = _get_key_numbers()
    lod_this = key.get('lod_this')
    lod_paper = key.get('lod_paper')
    r2_this = key.get('r2_this')
    factor = key.get('improvement_factor')
    lod_text = f"{lod_this:.2f} ppm" if isinstance(lod_this, (int, float)) else "N/A"
    lod_prev_text = f"{lod_paper:.2f} ppm" if isinstance(lod_paper, (int, float)) else "N/A"
    r2_text = f"{r2_this:.4f}" if isinstance(r2_this, (int, float)) else "N/A"
    factor_text = f"{factor:.0f}×" if isinstance(factor, (int, float)) else "N/A"
    
    # Banner
    ban = FancyBboxPatch((0.5, 6), 13, 0.7, boxstyle="round,rounding_size=0.1",
                         facecolor=COLORS['primary'], edgecolor='white', lw=2)
    ax.add_patch(ban)
    ax.text(7, 6.35, 'Data-Driven Spectral Feature Engineering for Sub-ppm Acetone Detection',
           fontsize=13, fontweight='bold', ha='center', color='white')
    
    # Sections
    sections = [('Challenge', 1.8, '#FFEBEE', COLORS['error'], f'Breath Acetone\nfor Diabetes\nPrev LoD: {lod_prev_text}'),
                ('Approach', 5.2, '#E3F2FD', COLORS['primary'], 'ZnO-NCF Sensor\nFeature Engineering\nROI Discovery'),
                ('Results', 8.6, '#E8F5E9', COLORS['success'], f'Analytical LoD*: {lod_text}\n{factor_text} improvement\nR² = {r2_text}'),
                ('Impact', 12, '#FFF3E0', COLORS['warning'], 'Clinical Context\nRoom Temp\nNon-invasive')]
    for t, x, bg, ec, txt in sections:
        box = FancyBboxPatch((x-1.3, 3), 2.6, 2.5, boxstyle="round,rounding_size=0.15",
                             facecolor=bg, edgecolor=ec, linewidth=2)
        ax.add_patch(box)
        ax.text(x, 5.2, t, fontsize=12, fontweight='bold', ha='center', color=ec)
        ax.text(x, 4.2, txt, fontsize=9, ha='center', va='center', color=COLORS['dark'])
    
    # Arrows
    for x in [3.0, 6.4, 9.8]:
        ax.annotate('', xy=(x+0.5, 4.25), xytext=(x, 4.25),
                   arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    # Metrics
    metrics = [('LoD', lod_text, COLORS['success']), ('Improvement', factor_text, COLORS['secondary']),
               ('R²', r2_text, COLORS['primary']), ('Response', '26 s', COLORS['warning'])]
    for i, (l, v, c) in enumerate(metrics):
        x = 2 + i*3.3
        box = FancyBboxPatch((x-1, 0.8), 2, 1.5, boxstyle="round,rounding_size=0.1",
                             facecolor='white', edgecolor=c, linewidth=2)
        ax.add_patch(box)
        ax.text(x, 1.9, l, fontsize=9, ha='center', color=COLORS['gray'])
        ax.text(x, 1.3, v, fontsize=13, fontweight='bold', ha='center', color=c)
    
    plt.tight_layout()
    _save_figure('graphical_abstract')
    plt.close()
    print("✓ graphical_abstract.png")


def create_loocv_parity_acetone():
    key = _load_acetone_calibration_metrics()
    centroid = (
        key.get("calibration_wavelength_shift", {})
        .get("centroid", {})
        if isinstance(key, dict)
        else {}
    )

    conc = centroid.get("concentrations") or centroid.get("concentrations_ppm")
    delta = centroid.get("delta_lambda") or centroid.get("wavelength_shifts_nm")
    if not isinstance(conc, list) or not isinstance(delta, list) or len(conc) != len(delta) or len(conc) < 3:
        return

    x = np.array([float(v) for v in conc], dtype=float)
    y = np.array([float(v) for v in delta], dtype=float)

    # LOOCV predictions
    preds = []
    for i in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        m, b = np.polyfit(x[mask], y[mask], 1)
        preds.append(m * x[i] + b)
    preds = np.array(preds, dtype=float)

    rmse_cv = float(np.sqrt(np.mean((preds - y) ** 2)))
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2_cv = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')

    fig, ax = plt.subplots(figsize=(6.8, 5.6), facecolor='white')
    ax.scatter(y, preds, s=70, color=COLORS['primary'], edgecolor='white', linewidth=1.5, zorder=3)
    min_v = float(min(np.min(y), np.min(preds)))
    max_v = float(max(np.max(y), np.max(preds)))
    pad = (max_v - min_v) * 0.1 if max_v > min_v else 0.05
    lo = min_v - pad
    hi = max_v + pad
    ax.plot([lo, hi], [lo, hi], linestyle='--', color=COLORS['gray'], linewidth=2, label='Ideal')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Observed Δλ (nm)', fontweight='bold')
    ax.set_ylabel('LOOCV Predicted Δλ (nm)', fontweight='bold')
    ax.set_title('Acetone LOOCV Parity (Centroid Δλ)', fontweight='bold')
    ax.grid(alpha=0.25, linestyle='--')
    ax.legend(loc='upper left')

    ax.text(
        0.98,
        0.02,
        f'LOOCV R² = {r2_cv:.4f}\nLOOCV RMSE = {rmse_cv:.4f} nm',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['light_gray']),
    )

    plt.tight_layout()
    _save_figure('acetone_loocv_parity')
    plt.close()
    print("✓ acetone_loocv_parity.png")

def create_measurement_timeline():
    fig, ax = plt.subplots(figsize=(13, 4), facecolor='white')
    ax.set_xlim(0, 13); ax.set_ylim(0, 3); ax.axis('off')
    ax.text(6.5, 2.7, 'Measurement Cycle Timeline', fontsize=16, fontweight='bold', ha='center', color=COLORS['dark'])
    phases = [
        ('Baseline\n(~200 frames N₂)', 1.5, COLORS['primary']),
        ('Exposure\n(~500 frames VOC)', 5.5, COLORS['secondary']),
        ('Recovery\n(~200 frames N₂)', 9.5, COLORS['success'])
    ]
    for label, center, color in phases:
        rect = FancyBboxPatch((center-2, 1), 4, 1, boxstyle="round,rounding_size=0.2",
                              facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(center, 1.5, label, fontsize=12, fontweight='bold', ha='center', va='center', color='white')
    ax.annotate('', xy=(4, 1.5), xytext=(3, 1.5), arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.annotate('', xy=(8, 1.5), xytext=(7, 1.5), arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.text(6.5, 0.5, 'Acquisition rate: 10 ms per spectrum | Purge between runs: 120 s', fontsize=11,
            ha='center', color=COLORS['gray'])
    plt.tight_layout()
    _save_figure('measurement_cycle')
    plt.close()
    print("✓ measurement_cycle.png")

def create_fabrication_flow():
    fig, ax = plt.subplots(figsize=(13, 5), facecolor='white')
    ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis('off')
    ax.text(6.5, 4.5, 'ZnO Nanostructure Fabrication Workflow', fontsize=16, fontweight='bold',
            ha='center', color=COLORS['dark'])
    steps = [
        ('Sol-Gel\nSynthesis\n60°C, 10 nm', 1.5),
        ('Aging &\nPurification', 3.8),
        ('Spray Coating\n250°C, 0.3 mL', 6.1),
        ('Annealing\n250°C, 2 h', 8.4),
        ('FESEM /\nThickness Check', 10.7),
    ]
    colors = [COLORS['primary'], COLORS['primary'], COLORS['secondary'], COLORS['secondary'], COLORS['success']]
    for (label, x), color in zip(steps, colors):
        add_box(ax, x-1.1, 2.4, 2.2, 1.6, color, label, fs=11)
    for i in range(len(steps)-1):
        ax.annotate('', xy=(steps[i+1][1]-1.1, 3.2), xytext=(steps[i][1]+1.1, 3.2),
                    arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    key = _get_key_numbers()
    thickness = key.get('zno_thickness')
    thickness_text = str(thickness).strip() if thickness else '85 nm'
    ax.text(6.5, 1.0, f'Outcome: Uniform {thickness_text} coating (wurtzite phase)', fontsize=11,
            ha='center', color=COLORS['gray'])
    plt.tight_layout()
    _save_figure('fabrication_flow')
    plt.close()
    print("✓ fabrication_flow.png")

def create_voc_concentration_chart():
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    path = DATA_DIR / "voc_protocol.csv"
    if not path.exists():
        plt.close(fig)
        return

    df = pd.read_csv(path)
    df = df.dropna(how="all")
    if df.empty or "VOC" not in df.columns or "Concentrations" not in df.columns:
        plt.close(fig)
        return

    vocs = df["VOC"].astype(str).str.strip().tolist()
    conc_rows: list[list[float]] = []
    import re

    for _, row in df.iterrows():
        text = str(row.get("Concentrations", ""))
        vals = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", text)]
        if not vals:
            plt.close(fig)
            return
        conc_rows.append(vals)

    max_len = max(len(v) for v in conc_rows)
    if max_len == 0:
        plt.close(fig)
        return

    concentrations = np.full((len(conc_rows), max_len), np.nan, dtype=float)
    for i, vals in enumerate(conc_rows):
        concentrations[i, : len(vals)] = np.asarray(vals, dtype=float)

    if not np.isfinite(concentrations).all():
        plt.close(fig)
        return

    colors = sns.color_palette('Blues', int(concentrations.shape[1]))
    bottom = np.zeros(len(vocs))
    for i in range(concentrations.shape[1]):
        ax.bar(vocs, concentrations[:, i], bottom=bottom, color=colors[i], edgecolor='white', linewidth=1.5,
               label=f'{concentrations[0, i]} ppm')
        bottom += concentrations[:, i]
    ax.set_ylabel('Cumulative Concentration (ppm)', fontsize=12, fontweight='bold')
    ax.set_title('VOC Test Panel Concentrations', fontsize=14, fontweight='bold')
    ax.legend(title='Levels', loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    _save_figure('voc_concentrations')
    plt.close()
    print("✓ voc_concentrations.png")

def create_sg_demo():
    fig, ax = plt.subplots(figsize=(11, 5), facecolor='white')
    df = _load_acetone_canonical_spectra(10.0)
    if df.empty or not {"wavelength", "absorbance"}.issubset(set(df.columns)):
        return

    wl = df["wavelength"].to_numpy(dtype=float)
    base = df["absorbance"].to_numpy(dtype=float)
    rng = np.random.default_rng(1)
    noisy = base + rng.normal(0.0, 0.03 * float(np.std(base) if np.std(base) > 0 else 1.0), size=base.shape)

    smoothed = savgol_filter(noisy, 11, 2)
    derivative = savgol_filter(noisy, 11, 2, deriv=1)

    ax.plot(wl, noisy, color=COLORS['gray'], alpha=0.6, label='Raw Absorbance (noisy)')
    ax.plot(wl, smoothed, color=COLORS['primary'], linewidth=2, label='Savitzky–Golay Smoothed')
    ax2 = ax.twinx()
    ax2.plot(wl, derivative, color=COLORS['secondary'], linewidth=1.5, label='First Derivative (right axis)')
    ax.set_xlabel('Wavelength (nm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Absorbance (a.u.)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('dA/dλ', fontsize=11, fontweight='bold', color=COLORS['secondary'])
    ax.set_title('Savitzky–Golay Processing Effect (example spectrum)', fontsize=14, fontweight='bold')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper right')
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    _save_figure('sg_processing_demo')
    plt.close()
    print("✓ sg_processing_demo.png")


def create_feature_engineering_progression():
    concentrations = [1.0, 5.0, 10.0]
    spectra = []
    required_cols = {"wavelength", "intensity", "transmittance"}
    for conc in concentrations:
        df = _load_acetone_canonical_spectra(conc)
        if df.empty or not required_cols.issubset(set(df.columns)):
            continue
        wl = df["wavelength"].to_numpy(dtype=float)
        intensity = df["intensity"].to_numpy(dtype=float)
        trans = np.clip(df["transmittance"].to_numpy(dtype=float), 1e-9, None)
        absorbance = -np.log10(trans)
        spectra.append({
            "conc": conc,
            "wl": wl,
            "intensity": intensity,
            "trans": trans,
            "absorbance": absorbance,
        })
    if not spectra:
        return

    wl = spectra[0]["wl"]
    ref_abs = spectra[0]["absorbance"]

    win = 31
    if win >= wl.size:
        win = wl.size - 1 if wl.size % 2 == 0 else wl.size
    if win < 5:
        win = 5 if wl.size >= 5 else wl.size - (1 - wl.size % 2)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return

    for spec in spectra:
        spec["derivative"] = savgol_filter(spec["absorbance"], window_length=win, polyorder=2, deriv=1)
        conv = np.convolve(spec["absorbance"], spec["derivative"], mode='same')
        spec["convolution"] = (conv - np.mean(conv)) / (np.std(conv) + 1e-9)
        spec["delta_abs"] = spec["absorbance"] - ref_abs
        spec["roi_metric"] = None

    metrics = _load_acetone_calibration_metrics()
    centroid = (
        metrics.get("calibration_wavelength_shift", {}).get("centroid", {})
        if isinstance(metrics, dict)
        else {}
    )
    roi_opt = centroid.get("roi_range") if isinstance(centroid, dict) else None
    try:
        roi_opt_tuple = (float(roi_opt[0]), float(roi_opt[1])) if roi_opt and len(roi_opt) == 2 else None
    except (TypeError, ValueError):
        roi_opt_tuple = None
    literature_roi = (675.0, 689.0)

    panels = [
        ("Raw Intensity I(λ)", "intensity", 'Counts (a.u.)', 'Lamp ripple + drift dominate'),
        ("Transmittance T = I/I₀", "trans", 'Unitless', 'Normalization removes source spectrum'),
        ("Absorbance A = -log₁₀(T)", "absorbance", 'Absorbance (a.u.)', 'Dynamic range linearized'),
        ("ΔAbsorbance relative to 1 ppm", "delta_abs", 'ΔAbs', 'Sub-ppm change emerges after preprocessing'),
        ("Derivative dA/dλ (Savitzky–Golay)", "derivative", 'a.u./nm', 'Baseline removed → transitions only'),
        ("Convolution A ⊗ dA/dλ (weighted feature)", "convolution", 'Normalized', 'ROI weighting highlights stable shift'),
    ]
    n_w_panels = len(panels)
    fig, axes = plt.subplots(
        n_w_panels + 1,
        1,
        figsize=(12.5, 17),
        gridspec_kw={'height_ratios': [1] * n_w_panels + [0.65]},
        sharex=False,
        facecolor='white',
    )
    for ax in axes[:-1]:
        ax.sharex(axes[0])
    theme_palette = [COLORS['primary'], COLORS['secondary'], COLORS['warning'], COLORS['purple']]
    colors_map = [theme_palette[i % len(theme_palette)] for i in range(len(spectra))]
    line_handles: list = []
    line_labels: list[str] = []
    for idx, (ax, (title, key, ylabel, annotation)) in enumerate(zip(axes[:-1], panels)):
        for spec, color in zip(spectra, colors_map):
            (line,) = ax.plot(
                spec["wl"],
                spec[key],
                color=color,
                lw=2.2 if key in {"delta_abs", "derivative", "convolution"} else 1.8,
                label=f"{spec['conc']:.0f} ppm" if idx == 0 else None,
            )
            if idx == 0:
                line_handles.append(line)
                line_labels.append(f"{spec['conc']:.0f} ppm")
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', color=COLORS['dark'])
        ax.grid(alpha=0.25, linestyle='--')
        if key in {"derivative", "convolution", "delta_abs"}:
            vals = np.concatenate([spec[key] for spec in spectra])
            lo, hi = np.percentile(vals, [2, 98])
            pad = (hi - lo) * 0.3 if hi > lo else 0.1
            ax.set_ylim(lo - pad, hi + pad)
        ax.text(
            0.01,
            0.9,
            annotation,
            transform=ax.transAxes,
            fontsize=9,
            color=COLORS['dark'],
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['gray'], alpha=0.85),
        )
        if literature_roi:
            ax.axvspan(*literature_roi, color=COLORS['gray'], alpha=0.12, label='Literature ROI' if idx == 0 else None)
        if roi_opt_tuple:
            ax.axvspan(*roi_opt_tuple, color=COLORS['success'], alpha=0.18, label='Optimized ROI' if idx == 0 else None)
        if roi_opt_tuple and spec.get("roi_metric") is None:
            mask = (spec["wl"] >= roi_opt_tuple[0]) & (spec["wl"] <= roi_opt_tuple[1])
        else:
            mask = (spec["wl"] >= literature_roi[0]) & (spec["wl"] <= literature_roi[1])
        # store roi metric per spec once
        for spec in spectra:
            if spec["roi_metric"] is None:
                span = roi_opt_tuple if roi_opt_tuple else literature_roi
                if span:
                    mask_metric = (spec["wl"] >= span[0]) & (spec["wl"] <= span[1])
                    if np.any(mask_metric):
                        spec["roi_metric"] = float(np.mean(spec["convolution"][mask_metric]))
                    else:
                        spec["roi_metric"] = 0.0

    roi_handles: list = []
    if literature_roi:
        roi_handles.append(mpatches.Patch(color=COLORS['gray'], alpha=0.2, label='Literature ROI'))
    if roi_opt_tuple:
        roi_handles.append(mpatches.Patch(color=COLORS['success'], alpha=0.25, label='Optimized ROI'))

    legend = fig.legend(
        line_handles + roi_handles,
        line_labels + [h.get_label() for h in roi_handles],
        loc='upper center',
        ncol=max(1, len(line_handles) + len(roi_handles)),
        frameon=True,
        bbox_to_anchor=(0.5, 0.985),
        columnspacing=1.4,
        handlelength=2.5,
    )
    if legend and legend.get_frame():
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)

    roi_ax = inset_axes(axes[-2], width="35%", height="45%", loc='upper left', borderpad=2)
    span = roi_opt_tuple if roi_opt_tuple else literature_roi
    if span:
        mask = (wl >= span[0] - 10) & (wl <= span[1] + 10)
        for spec, color in zip(spectra, colors_map):
            roi_ax.plot(spec["wl"][mask], spec["convolution"][mask], color=color, lw=2)
        roi_ax.set_title('Zoom on Optimized ROI', fontsize=9)
        roi_ax.set_xlabel('λ (nm)', fontsize=8)
        roi_ax.set_ylabel('Weight', fontsize=8)
        roi_ax.grid(alpha=0.3, linestyle='--')

    axes[-2].set_xlabel('Wavelength (nm)', fontweight='bold')

    trend_ax = axes[-1]
    trend_ax.set_title('ROI Feature vs Concentration (scalar trend)', fontsize=12, fontweight='bold', color=COLORS['dark'])
    conc_vals = np.array([spec['conc'] for spec in spectra], dtype=float)
    roi_vals = np.array([spec.get('roi_metric', 0.0) for spec in spectra], dtype=float)
    order = np.argsort(conc_vals)
    conc_sorted = conc_vals[order]
    roi_sorted = roi_vals[order]
    trend_ax.plot(conc_sorted, roi_sorted, marker='o', color=COLORS['primary'], lw=2.5)
    if conc_sorted.size >= 2:
        coeffs = np.polyfit(conc_sorted, roi_sorted, 1)
        fit = np.polyval(coeffs, conc_sorted)
        trend_ax.plot(conc_sorted, fit, linestyle='--', color=COLORS['secondary'], label=f"Fit slope = {coeffs[0]:.3f}")
    trend_ax.set_xlabel('Concentration (ppm)', fontweight='bold')
    trend_ax.set_ylabel('Mean ROI feature (a.u.)', fontweight='bold')
    trend_ax.grid(alpha=0.3, linestyle='--')
    if conc_sorted.size >= 2:
        trend_ax.legend(loc='upper left')

    fig.suptitle('Feature Engineering Progression (Multi-Concentration Acetone)', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_figure('feature_engineering_progression')
    plt.close()
    print("✓ feature_engineering_progression.png")


def create_classification_metrics():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    path = DATA_DIR / "clinical_classification_metrics.json"
    if not path.exists():
        print("! Skipping roc_confusion: missing data/clinical_classification_metrics.json")
        plt.close(fig)
        return

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        plt.close(fig)
        return

    roc = payload.get("roc") if isinstance(payload, dict) else None
    cm = payload.get("confusion_matrix") if isinstance(payload, dict) else None
    if not isinstance(roc, dict) or not isinstance(cm, list):
        plt.close(fig)
        return

    fpr = np.array(roc.get("fpr", []), dtype=float)
    tpr = np.array(roc.get("tpr", []), dtype=float)
    auc = roc.get("auc")
    if fpr.size < 2 or tpr.size != fpr.size:
        plt.close(fig)
        return

    ax = axes[0]
    auc_text = f"{float(auc):.3f}" if isinstance(auc, (int, float)) else "N/A"
    ax.plot(fpr, tpr, color=COLORS['primary'], linewidth=2, label=f'ROC-AUC = {auc_text}')
    ax.plot([0, 1], [0, 1], color=COLORS['gray'], linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (Healthy vs Diabetic)')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')

    ax = axes[1]
    matrix = np.array(cm, dtype=float)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        plt.close(fig)
        return
    matrix_i = np.rint(matrix).astype(int)
    sns.heatmap(matrix_i, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Pred Healthy', 'Pred Diabetic'],
                yticklabels=['Actual Healthy', 'Actual Diabetic'],
                ax=ax, annot_kws={"fontsize": 12, "fontweight": "bold"})
    ax.set_title('Confusion Matrix (Threshold 1.2 ppm)')
    plt.suptitle('Clinical Classification Metrics', fontsize=15, fontweight='bold')
    plt.tight_layout()
    _save_figure('roc_confusion')
    plt.close()
    print("✓ roc_confusion.png")

def create_performance_metrics_card():
    key = _get_key_numbers()
    lod_this = key.get('lod_this')
    r2_this = key.get('r2_this')
    r2_cv = key.get('r2_cv')
    factor = key.get('improvement_factor')
    lod_text = f"{lod_this:.2f} ppm" if isinstance(lod_this, (int, float)) else "N/A"
    r2_text = f"{r2_this:.4f}" if isinstance(r2_this, (int, float)) else "N/A"
    r2cv_text = f"{r2_cv:.4f}" if isinstance(r2_cv, (int, float)) else "N/A"
    factor_text = f"{factor:.0f}×" if isinstance(factor, (int, float)) else "N/A"

    metrics = [
        ('Detection Limit', lod_text, 'Breath-range sensitivity', COLORS['success']),
        ('Improvement', factor_text, 'vs reference analysis', COLORS['secondary']),
        ('R²', r2_text, 'calibration fit', COLORS['primary']),
        ('LOOCV R²', r2cv_text, 'generalization', COLORS['warning']),
    ]

    fig, ax = plt.subplots(figsize=(12.5, 4.5), facecolor='white')
    ax.axis('off')
    ax.text(0.02, 0.92, 'Performance Summary', fontsize=17, fontweight='bold', color=COLORS['dark'])
    ax.text(0.02, 0.83, 'ZnO–NCF + data-driven ROI (controlled chamber)', fontsize=11, color=COLORS['gray'])

    for i, (label, value, subtitle, color) in enumerate(metrics):
        left = 0.02 + i * 0.24
        width = 0.22
        # drop shadow
        shadow = FancyBboxPatch(
            (left + 0.005, 0.085), width, 0.6,
            boxstyle="round,rounding_size=0.025",
            transform=ax.transAxes,
            facecolor='#D0D5DD',
            edgecolor='none',
            alpha=0.3,
        )
        ax.add_patch(shadow)
        card = FancyBboxPatch(
            (left, 0.1), width, 0.6,
            boxstyle="round,rounding_size=0.025",
            transform=ax.transAxes,
            facecolor='white',
            edgecolor='none',
        )
        ax.add_patch(card)
        ax.plot([left + 0.015, left + 0.015], [0.13, 0.65], transform=ax.transAxes, color=color, linewidth=3)
        ax.text(left + 0.04, 0.6, label, fontsize=12, fontweight='bold', color=COLORS['dark'], transform=ax.transAxes)
        ax.text(left + 0.04, 0.48, value, fontsize=21, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(left + 0.04, 0.36, subtitle, fontsize=10.5, color=COLORS['gray'], transform=ax.transAxes)

    ax.text(0.02, 0.04, '* Analytical LoD and R² reported for controlled chamber measurements', fontsize=9.5, color=COLORS['gray'])
    plt.tight_layout()
    _save_figure('performance_metrics')
    plt.close()
    print("✓ performance_metrics.png")

def create_limitations_matrix():
    fig, ax = plt.subplots(figsize=(11, 6), facecolor='white')
    ax.axis('off')
    limitations = [
        ('Validation Scope', 'Synthetic gas only', 'Plan real-breath study (N>100)'),
        ('Long-term Stability', 'New ROI drift uncharacterized', '30-day drift campaign underway'),
        ('Environment', 'No T/RH compensation', 'Integrate humidity sensor + software correction'),
        ('Regulatory Path', 'Clinical trial pending', 'Map FDA/CE pathway, collect pilot data'),
    ]
    table = ax.table(cellText=[[l, g, m] for l, g, m in limitations],
                     colLabels=['Limitation', 'Gap', 'Mitigation'], loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS['light_gray'])
        if row == 0:
            cell.set_facecolor(COLORS['primary'])
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
        else:
            cell.set_facecolor('#FDFDFD')
    ax.set_title('Limitations and Mitigation Plan', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    _save_figure('limitations_matrix')
    plt.close()
    print("✓ limitations_matrix.png")

def main():
    print("\n" + "="*50)
    print("Generating Enhanced Diagrams v2")
    print("="*50 + "\n")
    create_pipeline()
    create_sensor()
    create_clinical()
    create_heatmap()
    create_tradeoff()
    create_baseline_vs_optimized_roi()
    create_roi_discovery_annotated()
    create_detectability_analysis()
    create_synthetic_injection_validation()
    create_virtual_sensor_array()
    create_virtual_sensor_array_radar()
    create_calibration_robustness_tests()
    create_feature_eng()
    create_feature_engineering_progression()
    create_abstract()
    create_loocv_parity_acetone()
    create_measurement_timeline()
    create_fabrication_flow()
    create_voc_concentration_chart()
    create_sg_demo()
    create_performance_metrics_card()
    create_limitations_matrix()
    print(f"\n✓ All diagrams saved to: {OUTPUT_DIR}\n")

if __name__ == "__main__":
    main()
