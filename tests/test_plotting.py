import numpy as np

from src.plotting import visualize_fits_image


class TestVisualizeFitsImage:
    def test_creates_file(self, synthetic_image_data, tmp_path):
        """visualize_fits_image should create a PNG file."""
        out = tmp_path / "preview.png"
        visualize_fits_image(
            synthetic_image_data,
            file_path=out.name,
            plots_dir=tmp_path,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_with_sources(self, synthetic_image_data, synthetic_sources, tmp_path):
        """visualize_fits_image should work with detected source overlay."""
        out = tmp_path / "preview_sources.png"
        visualize_fits_image(
            synthetic_image_data,
            sources=synthetic_sources,
            gaia_sources=None,
            file_path=out.name,
            plots_dir=tmp_path,
        )
        assert out.exists()

    def test_with_gaia_sources(self, synthetic_image_data, tmp_path):
        """visualize_fits_image should work with Gaia source overlay."""
        gaia = {"x": np.array([30, 60, 90]), "y": np.array([40, 80, 120])}
        out = tmp_path / "preview_gaia.png"
        visualize_fits_image(
            synthetic_image_data,
            sources=None,
            gaia_sources=gaia,
            file_path=out.name,
            plots_dir=tmp_path,
        )
        assert out.exists()

    def test_with_both_overlays(
        self, synthetic_image_data, synthetic_sources, tmp_path
    ):
        """visualize_fits_image should work with both source overlays."""
        gaia = {"x": np.array([30, 60]), "y": np.array([40, 80])}
        out = tmp_path / "preview_both.png"
        visualize_fits_image(
            synthetic_image_data,
            sources=synthetic_sources,
            gaia_sources=gaia,
            file_path=out.name,
            plots_dir=tmp_path,
        )
        assert out.exists()

    def test_with_tiny_image(self, tmp_path):
        """visualize_fits_image should handle very small images."""
        tiny = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        out = tmp_path / "tiny.png"
        visualize_fits_image(tiny, file_path=out.name, plots_dir=tmp_path)
        assert out.exists()

    def test_uses_agg_backend(self):
        """Matplotlib should use the Agg backend (set at module import)."""
        import matplotlib

        assert matplotlib.get_backend().upper() == "AGG"
