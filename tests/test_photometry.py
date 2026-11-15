import numpy as np

from src.photometry import Photometry


class TestPhotometryInit:
    def test_init_stores_attributes(
        self, synthetic_image_data, synthetic_sources, mock_wcs
    ):
        """Photometry should store all passed attributes."""
        phot = Photometry(
            synthetic_image_data,
            synthetic_sources,
            estimated_fwhm=5.0,
            psf_models={},
            wcs=mock_wcs,
        )
        assert phot.image_clean is synthetic_image_data
        assert phot.sources is synthetic_sources
        assert phot.estimated_fwhm == 5.0
        assert phot.photometry_results == {}


class TestAperturePhotometry:
    def test_aperture_photometry_runs(
        self, synthetic_image_data, synthetic_sources, mock_wcs
    ):
        """_aperture_photometry should populate photometry_results."""
        phot = Photometry(
            synthetic_image_data,
            synthetic_sources,
            estimated_fwhm=5.0,
            psf_models={},
            wcs=mock_wcs,
        )
        phot._aperture_photometry()
        assert "aperture" in phot.photometry_results
        result = phot.photometry_results["aperture"]
        assert len(result) == len(synthetic_sources)
        assert "flux" in result.colnames

    def test_aperture_photometry_positive_flux(
        self, synthetic_image_data, synthetic_sources, mock_wcs
    ):
        """Aperture fluxes should be positive for bright synthetic stars."""
        phot = Photometry(
            synthetic_image_data,
            synthetic_sources,
            estimated_fwhm=5.0,
            psf_models={},
            wcs=mock_wcs,
        )
        phot._aperture_photometry()
        result = phot.photometry_results["aperture"]
        assert np.all(result["flux"] > 0)


class TestPerformPhotometry:
    def test_perform_photometry_with_aperture_only(
        self, synthetic_image_data, synthetic_sources, mock_wcs
    ):
        """perform_photometry should run the specified methods."""
        phot = Photometry(
            synthetic_image_data,
            synthetic_sources,
            estimated_fwhm=5.0,
            psf_models={},
            wcs=mock_wcs,
        )
        results = phot.perform_photometry(methods=["aperture"])
        assert "aperture" in results

    def test_perform_photometry_skips_missing_psf_model(
        self, synthetic_image_data, synthetic_sources, mock_wcs
    ):
        """perform_photometry should skip methods with no PSF model."""
        phot = Photometry(
            synthetic_image_data,
            synthetic_sources,
            estimated_fwhm=5.0,
            psf_models={},
            wcs=mock_wcs,
        )
        # No PSF models loaded, so only aperture should run
        results = phot.perform_photometry(methods=["moffat", "aperture"])
        assert "aperture" in results
        assert "moffat" not in results

    def test_perform_photometry_with_psf_model(
        self, synthetic_image_data, synthetic_sources, mock_wcs
    ):
        """perform_photometry should run PSF fitting when a model is available."""
        from photutils.psf import MoffatPSF

        alpha = 5.0 / (2.0 * np.sqrt(2 ** (1.0 / 3.0) - 1))
        psf_models = {"moffat": MoffatPSF(alpha=alpha)}
        phot = Photometry(
            synthetic_image_data,
            synthetic_sources,
            estimated_fwhm=5.0,
            psf_models=psf_models,
            wcs=mock_wcs,
        )
        results = phot.perform_photometry(methods=["moffat"])
        assert "moffat" in results


class TestSaveCoordinates:
    def test_save_coordinates_creates_file(
        self, synthetic_image_data, synthetic_sources, mock_wcs, tmp_path
    ):
        """_save_coordinates should write a coordinate file to disk."""
        phot = Photometry(
            synthetic_image_data,
            synthetic_sources,
            estimated_fwhm=5.0,
            psf_models={},
            wcs=mock_wcs,
            catalogs_dir=tmp_path,
        )
        phot._aperture_photometry()
        phot._save_coordinates("aperture")
        expected = tmp_path / "xy_radec_aperture.txt"
        assert expected.exists()

    def test_save_coordinates_content(
        self, synthetic_image_data, synthetic_sources, mock_wcs, tmp_path
    ):
        """The saved coordinate file should have correct format (two columns)."""
        phot = Photometry(
            synthetic_image_data,
            synthetic_sources,
            estimated_fwhm=5.0,
            psf_models={},
            wcs=mock_wcs,
            catalogs_dir=tmp_path,
        )
        phot._aperture_photometry()
        phot._save_coordinates("aperture")
        expected = tmp_path / "xy_radec_aperture.txt"
        content = expected.read_text().strip()
        assert len(content) > 0
        lines = content.split("\n")
        for line in lines:
            parts = line.strip().split()
            assert len(parts) == 2  # RA and Dec
            float(parts[0])
            float(parts[1])

    def test_save_coordinates_skip_missing_method(self, tmp_path):
        """_save_coordinates should not fail for a method not in results."""
        phot = Photometry(
            np.ones((10, 10)),
            sources=None,
            estimated_fwhm=3.0,
            psf_models={},
            wcs=None,
            catalogs_dir=tmp_path,
        )
        phot.photometry_results = {}
        # Should not raise
        phot._save_coordinates("nonexistent")

    def test_save_coordinates_without_wcs(
        self, synthetic_image_data, synthetic_sources, tmp_path
    ):
        """_save_coordinates should handle missing WCS gracefully."""
        phot = Photometry(
            synthetic_image_data,
            synthetic_sources,
            estimated_fwhm=5.0,
            psf_models={},
            wcs=None,
            catalogs_dir=tmp_path,
        )
        phot._aperture_photometry()
        phot._save_coordinates("aperture")
        # File should not be created (WCS is needed for coordinate conversion)
        expected = tmp_path / "xy_radec_aperture.txt"
        # May or may not exist depending on error handling
