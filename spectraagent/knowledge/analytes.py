"""
spectraagent.knowledge.analytes
================================
Generic chemistry reference for volatile analytes.

This module provides ONLY physical/chemical properties — things that are true
regardless of which sensor, sensor type, or lab is being used.  It does NOT
contain sensor-specific response values (Δλ, LOD, sensitivity).  Those
values are observed from real experiments and stored in SensorMemory.

Adding a new analyte
---------------------
Add an entry to ANALYTE_REGISTRY.  The only required fields are name and
cas_number; fill as many optional fields as are relevant for the lab's
analytes.  The platform will operate correctly with partial entries —
agents gracefully degrade to "unknown analyte" context when data is sparse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class AnalyteProperties:
    """Physical and chemical properties of one analyte.

    These are sensor-independent facts used by agents to:
    - Explain cross-sensitivity patterns (polarity, functional group)
    - Estimate equilibration time (vapour pressure)
    - Flag safety thresholds in experimental context
    - Infer interferent likelihood from chemical similarity

    Attributes
    ----------
    name : str
        Common name (capitalised, e.g. "Ethanol").
    cas_number : str
        CAS registry number for unambiguous identification.
    formula : str
        Molecular formula.
    molecular_weight_g_mol : float
        Molar mass in g/mol.
    boiling_point_c : float
        Atmospheric boiling point in °C.
    polarity_index : float
        Snyder polarity index P' (0 = non-polar, 10.2 = water).
        Higher polarity → stronger interaction with polar MIP/polymer matrices.
    vapor_pressure_kpa_25c : float
        Saturated vapour pressure at 25 °C in kPa.
        Drives equilibration speed and sets maximum ppm ceiling in closed systems.
    functional_group : str
        Primary functional group affecting surface interaction,
        e.g. "alcohol", "ketone", "aromatic", "aldehyde", "ester", "alkene".
    known_interferents : list[str]
        Chemically similar analytes likely to cause cross-sensitivity.
        Ordered roughly by expected severity.
    osha_pel_ppm : float or None
        OSHA permissible exposure limit (TWA, ppm) — useful for safety context.
    idlh_ppm : float or None
        OSHA immediately dangerous to life or health level (ppm).
    notes : str
        Any important experimental or chemical notes relevant to optical sensing.
    """

    name: str
    cas_number: str
    formula: str = ""
    molecular_weight_g_mol: float = 0.0
    boiling_point_c: float = 0.0
    polarity_index: float = 0.0
    vapor_pressure_kpa_25c: float = 0.0
    functional_group: str = "unknown"
    known_interferents: list[str] = field(default_factory=list)
    osha_pel_ppm: Optional[float] = None
    idlh_ppm: Optional[float] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Registry — pure chemistry, no sensor-specific response values
# ---------------------------------------------------------------------------

ANALYTE_REGISTRY: dict[str, AnalyteProperties] = {

    "ethanol": AnalyteProperties(
        name="Ethanol",
        cas_number="64-17-5",
        formula="C₂H₅OH",
        molecular_weight_g_mol=46.07,
        boiling_point_c=78.4,
        polarity_index=5.2,
        vapor_pressure_kpa_25c=7.87,
        functional_group="alcohol",
        known_interferents=["methanol", "isopropanol", "acetone", "water_vapor"],
        osha_pel_ppm=1000.0,
        idlh_ppm=3300.0,
        notes=(
            "Hydrogen-bond donor/acceptor. Strong interaction with hydroxyl-selective "
            "MIP matrices. Rapid equilibration (~5–15 min depending on flow rate). "
            "Water vapour is the most common confounding interferent in humid environments."
        ),
    ),

    "methanol": AnalyteProperties(
        name="Methanol",
        cas_number="67-56-1",
        formula="CH₃OH",
        molecular_weight_g_mol=32.04,
        boiling_point_c=64.7,
        polarity_index=5.1,
        vapor_pressure_kpa_25c=16.9,
        functional_group="alcohol",
        known_interferents=["ethanol", "water_vapor", "formaldehyde"],
        osha_pel_ppm=200.0,
        idlh_ppm=6000.0,
        notes=(
            "Similar polarity to ethanol but lower MW. High vapour pressure means "
            "faster equilibration and desorption than ethanol. Toxic — "
            "check ventilation before high-concentration experiments."
        ),
    ),

    "acetone": AnalyteProperties(
        name="Acetone",
        cas_number="67-64-1",
        formula="(CH₃)₂CO",
        molecular_weight_g_mol=58.08,
        boiling_point_c=56.1,
        polarity_index=5.1,
        vapor_pressure_kpa_25c=30.8,
        functional_group="ketone",
        known_interferents=["isopropanol", "methyl_ethyl_ketone", "ethanol"],
        osha_pel_ppm=1000.0,
        idlh_ppm=2500.0,
        notes=(
            "Hydrogen-bond acceptor only (no donor). Very high vapour pressure — "
            "concentration in closed cells changes quickly if not sealed. "
            "Clinical relevance: breath acetone biomarker for diabetes/ketosis."
        ),
    ),

    "isopropanol": AnalyteProperties(
        name="Isopropanol",
        cas_number="67-63-0",
        formula="(CH₃)₂CHOH",
        molecular_weight_g_mol=60.10,
        boiling_point_c=82.6,
        polarity_index=3.9,
        vapor_pressure_kpa_25c=5.79,
        functional_group="alcohol",
        known_interferents=["acetone", "ethanol", "water_vapor"],
        osha_pel_ppm=400.0,
        idlh_ppm=2000.0,
        notes=(
            "Secondary alcohol, lower polarity than ethanol or methanol. "
            "Common lab disinfectant — ambient IPA in lab air can be 0.01–0.5 ppm, "
            "creating a background that elevates apparent baseline. "
            "Metabolically linked to acetone (oxidation product)."
        ),
    ),

    "toluene": AnalyteProperties(
        name="Toluene",
        cas_number="108-88-3",
        formula="C₆H₅CH₃",
        molecular_weight_g_mol=92.14,
        boiling_point_c=110.6,
        polarity_index=2.4,
        vapor_pressure_kpa_25c=3.79,
        functional_group="aromatic",
        known_interferents=["benzene", "xylene", "ethylbenzene", "styrene"],
        osha_pel_ppm=200.0,
        idlh_ppm=500.0,
        notes=(
            "Non-polar aromatic — interacts via π-π stacking and van der Waals. "
            "Part of the BTEX group (benzene, toluene, ethylbenzene, xylene); "
            "environmental/industrial exposure monitoring application. "
            "Work in fume hood — OSHA ceiling applies."
        ),
    ),

    "benzene": AnalyteProperties(
        name="Benzene",
        cas_number="71-43-2",
        formula="C₆H₆",
        molecular_weight_g_mol=78.11,
        boiling_point_c=80.1,
        polarity_index=2.7,
        vapor_pressure_kpa_25c=12.7,
        functional_group="aromatic",
        known_interferents=["toluene", "xylene", "ethylbenzene"],
        osha_pel_ppm=1.0,    # very low PEL — carcinogen
        idlh_ppm=500.0,
        notes=(
            "KNOWN CARCINOGEN — handle with strict fume-hood procedures. "
            "Very low OSHA PEL (1 ppm). LOD target must be well below 1 ppm "
            "for occupational safety applications. High cross-sensitivity with "
            "toluene and other monoaromatics."
        ),
    ),

    "ethylene": AnalyteProperties(
        name="Ethylene",
        cas_number="74-85-1",
        formula="C₂H₄",
        molecular_weight_g_mol=28.05,
        boiling_point_c=-103.7,
        polarity_index=0.0,
        vapor_pressure_kpa_25c=6000.0,   # gas at RT
        functional_group="alkene",
        known_interferents=["propylene", "acetylene"],
        osha_pel_ppm=None,    # simple asphyxiant, no OSHA PEL
        idlh_ppm=None,
        notes=(
            "Gas at room temperature — handle as compressed gas. "
            "Plant hormone / fruit ripening biomarker. "
            "Application: post-harvest agricultural freshness monitoring. "
            "Low polarity means interaction is primarily size/shape-based in MIP."
        ),
    ),

    "ammonia": AnalyteProperties(
        name="Ammonia",
        cas_number="7664-41-7",
        formula="NH₃",
        molecular_weight_g_mol=17.03,
        boiling_point_c=-33.4,
        polarity_index=8.8,
        vapor_pressure_kpa_25c=1003.0,   # gas at RT
        functional_group="amine",
        known_interferents=["methylamine", "water_vapor"],
        osha_pel_ppm=50.0,
        idlh_ppm=300.0,
        notes=(
            "Strong Lewis base — can interact strongly with Lewis-acid surface sites. "
            "Very high polarity. Gas at RT. Work in fume hood — corrosive. "
            "Food quality and industrial leak-detection application."
        ),
    ),

    "formaldehyde": AnalyteProperties(
        name="Formaldehyde",
        cas_number="50-00-0",
        formula="HCHO",
        molecular_weight_g_mol=30.03,
        boiling_point_c=-19.0,
        polarity_index=6.0,
        vapor_pressure_kpa_25c=446.0,    # gas at RT
        functional_group="aldehyde",
        known_interferents=["acetaldehyde", "methanol"],
        osha_pel_ppm=0.75,   # very low PEL — carcinogen
        idlh_ppm=20.0,
        notes=(
            "KNOWN CARCINOGEN / IRRITANT — very low OSHA PEL (0.75 ppm). "
            "Gas at room temperature. Indoor air quality and occupational health "
            "monitoring application. Highly reactive — can polymerise or react "
            "with sensor surfaces; check sensor compatibility before use."
        ),
    ),

    "co2": AnalyteProperties(
        name="Carbon Dioxide",
        cas_number="124-38-9",
        formula="CO₂",
        molecular_weight_g_mol=44.01,
        boiling_point_c=-78.5,    # sublimation point
        polarity_index=0.0,
        vapor_pressure_kpa_25c=6400.0,   # gas at RT
        functional_group="inorganic_gas",
        known_interferents=["N2O", "SO2"],
        osha_pel_ppm=5000.0,
        idlh_ppm=40000.0,
        notes=(
            "Non-polar linear molecule — typically produces very small optical sensor "
            "response. Used mainly as a reference gas or interferent check. "
            "Ambient CO₂ ~400 ppm — consider as background in sensitive sensors."
        ),
    ),
}

# Common name aliases
ANALYTE_REGISTRY["ethyl_alcohol"] = ANALYTE_REGISTRY["ethanol"]
ANALYTE_REGISTRY["methyl_alcohol"] = ANALYTE_REGISTRY["methanol"]
ANALYTE_REGISTRY["isopropyl_alcohol"] = ANALYTE_REGISTRY["isopropanol"]
ANALYTE_REGISTRY["2-propanol"] = ANALYTE_REGISTRY["isopropanol"]
ANALYTE_REGISTRY["ipa"] = ANALYTE_REGISTRY["isopropanol"]
ANALYTE_REGISTRY["propanone"] = ANALYTE_REGISTRY["acetone"]
ANALYTE_REGISTRY["methylbenzene"] = ANALYTE_REGISTRY["toluene"]
ANALYTE_REGISTRY["ethene"] = ANALYTE_REGISTRY["ethylene"]
ANALYTE_REGISTRY["nh3"] = ANALYTE_REGISTRY["ammonia"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lookup_analyte(gas_label: str) -> Optional[AnalyteProperties]:
    """Return AnalyteProperties for gas_label (case-insensitive), or None."""
    return ANALYTE_REGISTRY.get(gas_label.lower().strip().replace(" ", "_"))


def list_known_analytes() -> list[str]:
    """Return deduplicated list of canonical analyte names."""
    seen: set[str] = set()
    result: list[str] = []
    for ref in ANALYTE_REGISTRY.values():
        if ref.name not in seen:
            seen.add(ref.name)
            result.append(ref.name)
    return sorted(result)


def format_analyte_chemistry_brief(props: AnalyteProperties) -> str:
    """One-paragraph chemistry summary for agent prompt injection."""
    parts = [
        f"**{props.name}** (CAS {props.cas_number}, {props.formula}, "
        f"MW {props.molecular_weight_g_mol:.1f} g/mol, "
        f"BP {props.boiling_point_c:.1f}°C, "
        f"Snyder P'={props.polarity_index:.1f}, "
        f"functional group: {props.functional_group}).",
    ]
    if props.vapor_pressure_kpa_25c > 0:
        vp = props.vapor_pressure_kpa_25c
        eq_hint = "fast (<3 min)" if vp > 20 else "moderate (5–10 min)" if vp > 5 else "slow (10–20 min)"
        parts.append(f"Vapour pressure {vp:.1f} kPa at 25°C → equilibration {eq_hint}.")
    if props.known_interferents:
        parts.append(f"Primary interferents: {', '.join(props.known_interferents[:4])}.")
    if props.osha_pel_ppm is not None:
        parts.append(f"OSHA PEL: {props.osha_pel_ppm} ppm.")
    if props.notes:
        parts.append(props.notes)
    return " ".join(parts)
