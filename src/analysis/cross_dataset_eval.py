"""Cross-dataset generalisation benchmark.

Evaluates how well a model trained on one sensor configuration generalises
to data from a different configuration — without retraining.  This is the
core scientific metric for the data-driven, physics-agnostic platform.

Benchmark protocol
------------------
1. Load datasets from N configurations (each a ``SpectralDataset``).
2. For each held-out configuration C:
   - Train on all other configurations.
   - Evaluate on C (zero-shot transfer).
   - Record accuracy / MAE / R² depending on task.
3. Aggregate results → cross-config generalisation scores.

Usage
-----
::

    from src.analysis.cross_dataset_eval import (
        CrossDatasetBenchmark,
        BenchmarkConfig,
        run_benchmark,
    )
    from src.io.universal_loader import SpectralDataset

    bench = CrossDatasetBenchmark(BenchmarkConfig(task="classification"))
    bench.add_config("CCS200_v1", ds_v1)
    bench.add_config("CCS200_v2", ds_v2)
    bench.add_config("Instrument_B", ds_b)

    results = bench.run()
    print(results.summary())
    print(results.mean_accuracy)   # averaged across all leave-one-out folds
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "CrossDatasetBenchmark",
    "run_benchmark",
]


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Configuration for the cross-dataset benchmark.

    Parameters
    ----------
    task :
        ``'classification'`` — analyte identification accuracy.
        ``'regression'``    — concentration MAE and R².
    n_components_pca :
        PCA components to use as features when no pre-trained encoder is
        provided.  Set 0 to use raw spectra (slow, not recommended for
        high-dim data).
    classifier :
        Scikit-learn estimator name for the classification baseline.
        ``'knn'``, ``'svc'``, or ``'rf'`` (random forest).
    min_samples_per_class :
        Minimum labelled samples required per class.  Configs with fewer
        samples are skipped with a warning.
    interpolate_wavelengths :
        If True, interpolate all datasets to a common wavelength grid before
        evaluation (required if configs have different wavelength ranges).
    seed :
        Random seed for reproducibility.
    """
    task: Literal["classification", "regression"] = "classification"
    n_components_pca: int = 32
    classifier: Literal["knn", "svc", "rf"] = "knn"
    min_samples_per_class: int = 2
    interpolate_wavelengths: bool = True
    seed: int = 42


@dataclass
class ConfigResult:
    """Per-configuration benchmark result."""
    config_id: str
    train_configs: list[str]
    test_config: str
    n_train: int
    n_test: int
    # Classification metrics
    accuracy: float | None = None
    balanced_accuracy: float | None = None
    # Regression metrics
    mae: float | None = None
    r2: float | None = None
    # Feature quality
    embedding_separation: float | None = None  # silhouette score on PCA features


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results across all leave-one-out folds."""
    config_results: list[ConfigResult] = field(default_factory=list)
    task: str = "classification"

    @property
    def mean_accuracy(self) -> float:
        accs = [r.accuracy for r in self.config_results if r.accuracy is not None]
        return float(np.mean(accs)) if accs else float("nan")

    @property
    def mean_mae(self) -> float:
        maes = [r.mae for r in self.config_results if r.mae is not None]
        return float(np.mean(maes)) if maes else float("nan")

    @property
    def mean_r2(self) -> float:
        r2s = [r.r2 for r in self.config_results if r.r2 is not None]
        return float(np.mean(r2s)) if r2s else float("nan")

    @property
    def mean_silhouette(self) -> float:
        sils = [r.embedding_separation for r in self.config_results
                if r.embedding_separation is not None]
        return float(np.mean(sils)) if sils else float("nan")

    def summary(self) -> str:
        lines = [
            f"Cross-Dataset Benchmark ({self.task})",
            "=" * 50,
        ]
        for r in self.config_results:
            line = f"  Test: {r.test_config:<20s}  "
            if r.accuracy is not None:
                line += f"acc={r.accuracy:.3f}  "
            if r.mae is not None:
                line += f"MAE={r.mae:.4f}  R²={r.r2:.3f}  "
            if r.embedding_separation is not None:
                line += f"silhouette={r.embedding_separation:.3f}"
            lines.append(line)
        lines.append("-" * 50)
        if self.task == "classification":
            lines.append(f"  Mean accuracy:   {self.mean_accuracy:.4f}")
        else:
            lines.append(
                f"  Mean MAE: {self.mean_mae:.4f}  Mean R²: {self.mean_r2:.4f}")
        lines.append(f"  Mean silhouette: {self.mean_silhouette:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class CrossDatasetBenchmark:
    """Leave-one-configuration-out generalisation benchmark.

    Registers multiple ``SpectralDataset`` objects (one per sensor
    configuration) and evaluates cross-config transfer performance.
    """

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        self.config = config or BenchmarkConfig()
        self._configs: dict[str, "SpectralDataset"] = {}  # type: ignore[name-defined]
        self._encoder: Callable | None = None  # optional pre-trained encoder

    def add_config(
        self,
        name: str,
        dataset: "SpectralDataset",  # type: ignore[name-defined]
    ) -> "CrossDatasetBenchmark":
        """Register a dataset from a specific sensor configuration.

        Parameters
        ----------
        name :
            Unique identifier for this configuration (e.g. ``'CCS200_v1'``).
        dataset :
            Loaded ``SpectralDataset``.
        """
        self._configs[name] = dataset
        return self

    def set_encoder(self, encoder: Callable) -> "CrossDatasetBenchmark":
        """Optionally plug in a pre-trained encoder.

        The encoder must accept an ndarray of shape (N, n_wl) and return
        (N, embed_dim).  If not set, PCA is used as a feature extractor.
        """
        self._encoder = encoder
        return self

    def run(self) -> BenchmarkResult:
        """Run the leave-one-config-out benchmark.

        Returns
        -------
        BenchmarkResult with per-config and aggregate metrics.
        """
        if len(self._configs) < 2:
            raise ValueError(
                "At least 2 configurations required for cross-dataset evaluation. "
                f"Got {len(self._configs)}."
            )

        results = BenchmarkResult(task=self.config.task)
        config_names = list(self._configs.keys())

        # Build a common wavelength grid if needed
        common_wl = self._build_common_wl()

        # Encode + align all configs
        encoded: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
        for name, ds in self._configs.items():
            X, y = self._prepare_dataset(ds, common_wl)
            encoded[name] = (X, y)

        # Leave-one-out loop
        for test_name in config_names:
            train_names = [n for n in config_names if n != test_name]

            # Concatenate training data
            X_train_parts, y_train_parts = [], []
            for n in train_names:
                X_part, y_part = encoded[n]
                X_train_parts.append(X_part)
                if y_part is not None:
                    y_train_parts.append(y_part)

            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = (np.concatenate(y_train_parts, axis=0)
                       if y_train_parts else None)
            X_test, y_test = encoded[test_name]

            result = self._evaluate_fold(
                test_name, train_names, X_train, y_train, X_test, y_test)
            results.config_results.append(result)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_common_wl(self) -> np.ndarray | None:
        if not self.config.interpolate_wavelengths:
            return None
        # Use the wavelength axis with the most points
        datasets = list(self._configs.values())
        return datasets[int(np.argmax([ds.n_wavelengths for ds in datasets]))].wavelengths

    def _prepare_dataset(
        self,
        ds: "SpectralDataset",  # type: ignore[name-defined]
        common_wl: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Align to common grid and extract features."""
        spectra = ds.spectra.copy()

        # Interpolate to common wavelength grid if needed
        same_grid = (common_wl is not None
                     and len(ds.wavelengths) == len(common_wl)
                     and np.allclose(ds.wavelengths, common_wl, atol=0.1))
        if common_wl is not None and not same_grid:
            spectra = np.stack([
                np.interp(common_wl, ds.wavelengths, row)
                for row in spectra
            ])

        # Feature extraction
        if self._encoder is not None:
            X = self._encoder(spectra)
        else:
            X = self._pca_features(spectra)

        return X, ds.labels

    def _pca_features(self, spectra: np.ndarray) -> np.ndarray:
        """Reduce to PCA features (fit on this dataset only for LOO)."""
        try:
            from sklearn.decomposition import PCA
            n_comp = min(self.config.n_components_pca,
                         spectra.shape[0] - 1,
                         spectra.shape[1])
            if n_comp < 1:
                return spectra
            pca = PCA(n_components=n_comp, random_state=self.config.seed)
            return pca.fit_transform(spectra)
        except ImportError:
            warnings.warn("sklearn not available — using raw spectra as features.")
            return spectra

    def _evaluate_fold(
        self,
        test_name: str,
        train_names: list[str],
        X_train: np.ndarray,
        y_train: np.ndarray | None,
        X_test: np.ndarray,
        y_test: np.ndarray | None,
    ) -> ConfigResult:
        """Train on source configs, evaluate on test config."""
        result = ConfigResult(
            config_id=test_name,
            train_configs=train_names,
            test_config=test_name,
            n_train=len(X_train),
            n_test=len(X_test),
        )

        # Compute embedding separation (silhouette) — only for classification
        # (discrete labels; skip for regression with many unique float values)
        if (y_test is not None
                and self.config.task == "classification"
                and len(np.unique(y_test)) > 1):
            result.embedding_separation = self._silhouette(X_test, y_test)

        if y_train is None or y_test is None:
            return result

        if self.config.task == "classification":
            result.accuracy, result.balanced_accuracy = self._classify(
                X_train, y_train, X_test, y_test)
        else:
            result.mae, result.r2 = self._regress(
                X_train, y_train, X_test, y_test)

        return result

    def _classify(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
    ) -> tuple[float, float]:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, balanced_accuracy_score

            scaler = StandardScaler()
            Xt = scaler.fit_transform(X_train)
            Xv = scaler.transform(X_test)

            clf = self._build_classifier()

            # Filter classes with enough samples
            classes, counts = np.unique(y_train, return_counts=True)
            valid = classes[counts >= self.config.min_samples_per_class]
            mask_tr = np.isin(y_train, valid)
            mask_te = np.isin(y_test, valid)

            if mask_tr.sum() < 2 or mask_te.sum() < 1:
                warnings.warn(
                    f"Not enough samples for classification in fold {self.test_name}")
                return float("nan"), float("nan")

            clf.fit(Xt[mask_tr], y_train[mask_tr])
            preds = clf.predict(Xv[mask_te])
            acc = float(accuracy_score(y_test[mask_te], preds))
            bal = float(balanced_accuracy_score(y_test[mask_te], preds))
            return acc, bal

        except ImportError:
            warnings.warn("sklearn required for classification benchmark.")
            return float("nan"), float("nan")

    def _regress(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
    ) -> tuple[float, float]:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_absolute_error, r2_score

            # Drop NaN labels
            tr_mask = np.isfinite(y_train.astype(float))
            te_mask = np.isfinite(y_test.astype(float))
            if tr_mask.sum() < 2 or te_mask.sum() < 1:
                return float("nan"), float("nan")

            scaler = StandardScaler()
            Xt = scaler.fit_transform(X_train[tr_mask])
            Xv = scaler.transform(X_test[te_mask])

            reg = self._build_regressor()
            reg.fit(Xt, y_train[tr_mask].astype(float))
            preds = reg.predict(Xv)
            mae = float(mean_absolute_error(y_test[te_mask].astype(float), preds))
            r2 = float(r2_score(y_test[te_mask].astype(float), preds))
            return mae, r2

        except ImportError:
            warnings.warn("sklearn required for regression benchmark.")
            return float("nan"), float("nan")

    def _silhouette(self, X: np.ndarray, y: np.ndarray) -> float:
        try:
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import StandardScaler
            Xs = StandardScaler().fit_transform(X)
            unique = np.unique(y)
            if len(unique) < 2 or len(X) < 4:
                return float("nan")
            return float(silhouette_score(Xs, y))
        except ImportError:
            return float("nan")

    def _build_classifier(self):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        c = self.config.classifier
        if c == "knn":
            return KNeighborsClassifier(n_neighbors=3, metric="cosine")
        elif c == "svc":
            return SVC(kernel="rbf", probability=True,
                       random_state=self.config.seed)
        elif c == "rf":
            return RandomForestClassifier(
                n_estimators=100, random_state=self.config.seed)
        raise ValueError(f"Unknown classifier: {c!r}")

    def _build_regressor(self):
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=3, metric="cosine")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_benchmark(
    configs: dict[str, "SpectralDataset"],  # type: ignore[name-defined]
    task: Literal["classification", "regression"] = "classification",
    encoder: Callable | None = None,
    **kwargs,
) -> BenchmarkResult:
    """One-shot cross-dataset benchmark.

    Parameters
    ----------
    configs :
        Mapping of config_name → SpectralDataset.
    task :
        ``'classification'`` or ``'regression'``.
    encoder :
        Optional callable: (N, n_wl) ndarray → (N, embed_dim) ndarray.
        If None, PCA features are used.

    Returns
    -------
    BenchmarkResult
    """
    bench_cfg = BenchmarkConfig(task=task, **kwargs)
    bench = CrossDatasetBenchmark(bench_cfg)
    for name, ds in configs.items():
        bench.add_config(name, ds)
    if encoder is not None:
        bench.set_encoder(encoder)
    return bench.run()
