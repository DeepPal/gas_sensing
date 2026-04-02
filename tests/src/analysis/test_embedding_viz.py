"""Tests for src.analysis.embedding_viz — dimensionality reduction and visualisation."""
import numpy as np
import pytest
from src.analysis.embedding_viz import (
    EmbeddingResult,
    reduce_dimensions,
    plot_embedding,
    plot_reconstruction,
    plot_training_curves,
)


@pytest.fixture
def latents():
    rng = np.random.default_rng(42)
    return rng.normal(size=(30, 16)).astype(np.float32)


class TestReduceDimensions:
    def test_pca_shape(self, latents):
        result = reduce_dimensions(latents, method="pca")
        assert result.coords.shape == (30, 2)
        assert result.method == "pca"
        assert result.latent_dim == 16

    def test_tsne_shape(self, latents):
        result = reduce_dimensions(latents, method="tsne", tsne_perplexity=5.0)
        assert result.coords.shape == (30, 2)

    def test_umap_missing_raises(self, latents, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "umap":
                raise ImportError("No module named 'umap'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="umap-learn"):
            reduce_dimensions(latents, method="umap")

    def test_invalid_method_raises(self, latents):
        with pytest.raises(ValueError, match="Unknown method"):
            reduce_dimensions(latents, method="mds")  # type: ignore

    def test_coords_finite(self, latents):
        result = reduce_dimensions(latents, method="pca")
        assert np.isfinite(result.coords).all()

    def test_embedding_result_metadata(self, latents):
        result = reduce_dimensions(latents, method="pca")
        result.metadata["analyte"] = "Ethanol"
        assert result.metadata["analyte"] == "Ethanol"


class TestPlotEmbedding:
    def test_returns_figure(self, latents):
        pytest.importorskip("plotly")
        fig = plot_embedding(latents, method="pca")
        assert fig is not None

    def test_with_numeric_colour(self, latents):
        pytest.importorskip("plotly")
        colours = np.linspace(0.1, 10.0, 30)
        fig = plot_embedding(latents, colour_by=colours,
                             colour_label="Concentration (ppm)", method="pca")
        assert fig is not None

    def test_with_string_colour(self, latents):
        pytest.importorskip("plotly")
        labels = ["Ethanol"] * 15 + ["IPA"] * 15
        fig = plot_embedding(latents, colour_by=labels, method="pca")
        assert fig is not None

    def test_accepts_embedding_result(self, latents):
        pytest.importorskip("plotly")
        result = reduce_dimensions(latents, method="pca")
        fig = plot_embedding(result)
        assert fig is not None


class TestPlotReconstruction:
    def test_basic(self):
        pytest.importorskip("plotly")
        wl = np.linspace(400, 900, 100)
        original = np.random.default_rng(0).normal(0.5, 0.1, (5, 100))
        recon = original + np.random.default_rng(1).normal(0, 0.01, (5, 100))
        fig = plot_reconstruction(original, recon, wavelengths=wl)
        assert fig is not None

    def test_subset_indices(self):
        pytest.importorskip("plotly")
        original = np.random.default_rng(0).normal(size=(10, 50))
        recon = original.copy()
        fig = plot_reconstruction(original, recon, indices=[0, 2, 4])
        assert fig is not None


class TestPlotTrainingCurves:
    def test_basic(self):
        pytest.importorskip("plotly")
        hist = {
            "train_loss": [0.5, 0.4, 0.3, 0.2, 0.15],
            "val_loss": [0.55, 0.45, 0.35, 0.25, 0.20],
        }
        fig = plot_training_curves(hist)
        assert fig is not None

    def test_train_only(self):
        pytest.importorskip("plotly")
        hist = {"train_loss": [1.0, 0.8, 0.6]}
        fig = plot_training_curves(hist)
        assert fig is not None
