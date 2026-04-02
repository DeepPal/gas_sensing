"""Unified multi-dataset registry for SpectraAgent.

Provides a single interface to register, list, and load spectral datasets
from any source — different analytes, sensor configurations, and data formats.

Usage
-----
::

    from src.io.dataset_registry import DatasetRegistry

    registry = DatasetRegistry()

    # Register datasets
    registry.register(
        name="Ethanol_CCS200",
        path="output/batch/Ethanol/stable_selected",
        analyte="Ethanol",
        config_id="CCS200_v1",
        signal_type="intensity",
        normalisation="snv",
        description="Ethanol 0.1–10 ppm, CCS200 spectrometer",
    )

    registry.register(
        name="IPA_CCS200",
        path="data/IPA/stable",
        analyte="IPA",
        config_id="CCS200_v1",
    )

    # Load any registered dataset
    ds = registry.load("Ethanol_CCS200")
    print(ds)   # SpectralDataset(n_samples=5, ...)

    # Load and merge multiple datasets
    merged = registry.load_merged(["Ethanol_CCS200", "IPA_CCS200"])

    # Save/load registry to JSON
    registry.save("output/dataset_registry.json")
    registry2 = DatasetRegistry.from_file("output/dataset_registry.json")

    # List all registered datasets
    print(registry.summary())
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from src.io.universal_loader import (
    SpectralDataset,
    SignalType,
    Normalisation,
    load_dataset,
    merge_datasets,
)

__all__ = [
    "DatasetEntry",
    "DatasetRegistry",
]


# ---------------------------------------------------------------------------
# Dataset entry
# ---------------------------------------------------------------------------

@dataclass
class DatasetEntry:
    """Metadata for one registered dataset.

    Attributes
    ----------
    name :
        Unique identifier string, e.g. ``'Ethanol_CCS200'``.
    path :
        File or directory path (absolute or relative to registry location).
    analyte :
        Analyte name (e.g. ``'Ethanol'``, ``'IPA'``). ``None`` if multi-analyte.
    config_id :
        Sensor configuration identifier (e.g. ``'CCS200_v1'``, ``'SensorB'``).
    signal_type :
        Signal type to extract (``'intensity'``, ``'absorbance'``, ``'auto'``).
    normalisation :
        Preprocessing to apply (``'snv'``, ``'area'``, ``'none'``).
    description :
        Free-text description.
    tags :
        List of arbitrary tags (e.g. ``['calibration', 'real', 'simulated']``).
    metadata :
        Any additional key-value metadata.
    """
    name: str
    path: str
    analyte: str | None = None
    config_id: str | None = None
    signal_type: str = "auto"
    normalisation: str = "none"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class DatasetRegistry:
    """Registry of named spectral datasets.

    Thread-safe for reads; not thread-safe for concurrent writes.
    """

    def __init__(self) -> None:
        self._entries: dict[str, DatasetEntry] = {}
        self._registry_path: Path | None = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        path: str | Path,
        analyte: str | None = None,
        config_id: str | None = None,
        signal_type: SignalType = "auto",
        normalisation: Normalisation = "none",
        description: str = "",
        tags: list[str] | None = None,
        overwrite: bool = False,
        **metadata: Any,
    ) -> "DatasetRegistry":
        """Register a new dataset.

        Parameters
        ----------
        name :
            Unique name for this dataset.
        path :
            Path to the data file or directory.
        analyte :
            Analyte name (inferred from path if not provided).
        config_id :
            Sensor configuration identifier.
        signal_type :
            Signal column to extract.
        normalisation :
            Normalisation to apply on load.
        description :
            Human-readable description.
        tags :
            Arbitrary tags for filtering.
        overwrite :
            If False (default), raises if name already registered.
        **metadata :
            Additional key-value pairs stored in the entry.

        Returns
        -------
        self (for chaining)
        """
        if name in self._entries and not overwrite:
            raise ValueError(
                f"Dataset '{name}' already registered. "
                "Use overwrite=True to replace it."
            )
        entry = DatasetEntry(
            name=name,
            path=str(path),
            analyte=analyte,
            config_id=config_id,
            signal_type=signal_type,
            normalisation=normalisation,
            description=description,
            tags=tags or [],
            metadata=metadata,
        )
        self._entries[name] = entry
        return self

    def unregister(self, name: str) -> None:
        """Remove a dataset from the registry."""
        if name not in self._entries:
            raise KeyError(f"Dataset '{name}' not found in registry.")
        del self._entries[name]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        name: str,
        normalisation: Normalisation | None = None,
        signal_type: SignalType | None = None,
        reference_wavelengths: "np.ndarray | None" = None,  # type: ignore[name-defined]
    ) -> SpectralDataset:
        """Load a registered dataset.

        Parameters
        ----------
        name :
            Registered dataset name.
        normalisation :
            Override the registered normalisation.
        signal_type :
            Override the registered signal type.
        reference_wavelengths :
            If provided, interpolate to this wavelength grid.

        Returns
        -------
        SpectralDataset
        """
        entry = self._get_entry(name)
        path = self._resolve_path(entry.path)

        return load_dataset(
            path,
            signal_type=signal_type or entry.signal_type,  # type: ignore[arg-type]
            normalisation=normalisation or entry.normalisation,  # type: ignore[arg-type]
            analyte=entry.analyte,
            config_id=entry.config_id,
            reference_wavelengths=reference_wavelengths,
        )

    def load_merged(
        self,
        names: list[str],
        normalisation: Normalisation | None = None,
        reference_wavelengths: "np.ndarray | None" = None,  # type: ignore[name-defined]
    ) -> SpectralDataset:
        """Load and merge multiple registered datasets onto a common wavelength grid.

        Parameters
        ----------
        names :
            List of registered dataset names to merge.
        normalisation :
            Normalisation to apply (overrides individual entries).
        reference_wavelengths :
            Common wavelength grid.  If None, uses the grid of the first dataset.

        Returns
        -------
        Merged SpectralDataset
        """
        datasets = [
            self.load(n, normalisation=normalisation,
                      reference_wavelengths=reference_wavelengths)
            for n in names
        ]
        return merge_datasets(*datasets,
                              reference_wavelengths=reference_wavelengths)

    def load_by_analyte(
        self,
        analyte: str,
        normalisation: Normalisation | None = None,
    ) -> SpectralDataset:
        """Load and merge all datasets matching a given analyte name."""
        matches = [name for name, entry in self._entries.items()
                   if entry.analyte and entry.analyte.lower() == analyte.lower()]
        if not matches:
            raise KeyError(
                f"No datasets registered for analyte '{analyte}'. "
                f"Available analytes: {self.list_analytes()}"
            )
        return self.load_merged(matches, normalisation=normalisation)

    def load_by_config(
        self,
        config_id: str,
        normalisation: Normalisation | None = None,
    ) -> SpectralDataset:
        """Load and merge all datasets from a given sensor configuration."""
        matches = [name for name, entry in self._entries.items()
                   if entry.config_id == config_id]
        if not matches:
            raise KeyError(
                f"No datasets registered for config '{config_id}'. "
                f"Available configs: {self.list_configs()}"
            )
        return self.load_merged(matches, normalisation=normalisation)

    def load_by_tag(
        self,
        tag: str,
        normalisation: Normalisation | None = None,
    ) -> SpectralDataset:
        """Load and merge all datasets with a given tag."""
        matches = [name for name, entry in self._entries.items()
                   if tag in entry.tags]
        if not matches:
            raise KeyError(f"No datasets tagged '{tag}'.")
        return self.load_merged(matches, normalisation=normalisation)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def list_names(self) -> list[str]:
        """Return sorted list of all registered dataset names."""
        return sorted(self._entries.keys())

    def list_analytes(self) -> list[str]:
        """Return sorted list of unique analyte names."""
        return sorted({e.analyte for e in self._entries.values()
                       if e.analyte is not None})

    def list_configs(self) -> list[str]:
        """Return sorted list of unique config IDs."""
        return sorted({e.config_id for e in self._entries.values()
                       if e.config_id is not None})

    def get_entry(self, name: str) -> DatasetEntry:
        """Return the DatasetEntry for a registered name."""
        return self._get_entry(name)

    def summary(self) -> str:
        """Human-readable table of all registered datasets."""
        if not self._entries:
            return "DatasetRegistry: (empty)"

        lines = [
            f"DatasetRegistry ({len(self._entries)} datasets):",
            f"{'Name':<30} {'Analyte':<12} {'Config':<14} {'Norm':<8} Description",
            "-" * 80,
        ]
        for name in sorted(self._entries):
            e = self._entries[name]
            lines.append(
                f"{name:<30} {str(e.analyte):<12} {str(e.config_id):<14} "
                f"{e.normalisation:<8} {e.description[:30]}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"DatasetRegistry(n={len(self._entries)}, "
                f"analytes={self.list_analytes()}, "
                f"configs={self.list_configs()})")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save registry to a JSON file.

        Parameters
        ----------
        path :
            Output file path (e.g. ``'output/dataset_registry.json'``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "entries": {name: entry.to_dict()
                        for name, entry in self._entries.items()},
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._registry_path = path

    @classmethod
    def from_file(cls, path: str | Path) -> "DatasetRegistry":
        """Load a registry from a JSON file.

        Parameters
        ----------
        path :
            Path to a JSON file saved by ``DatasetRegistry.save()``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Registry file not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        registry = cls()
        registry._registry_path = path

        for name, entry_dict in data.get("entries", {}).items():
            registry._entries[name] = DatasetEntry.from_dict(entry_dict)

        return registry

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_entry(self, name: str) -> DatasetEntry:
        if name not in self._entries:
            raise KeyError(
                f"Dataset '{name}' not registered. "
                f"Available: {self.list_names()}"
            )
        return self._entries[name]

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve path relative to registry file location if relative."""
        p = Path(path_str)
        if p.is_absolute() or p.exists():
            return p
        if self._registry_path is not None:
            candidate = self._registry_path.parent / p
            if candidate.exists():
                return candidate
        return p
