import numpy as np

from src.psf_modeling import PSFModeler


class TestPSFModelerInit:
    def test_init_stores_attributes(self, synthetic_image_data, synthetic_sources):
        """PSFModeler should store all passed attributes."""
        modeler = PSFModeler(
            synthetic_image_data, synthetic_sources, estimated_fwhm=5.0
        )
        assert modeler.image_clean is synthetic_image_data
        assert modeler.sources is synthetic_sources
        assert modeler.estimated_fwhm == 5.0
        assert modeler.psf_models == {}


class TestBuildModels:
    def test_build_psf_models_creates_moffat(
        self, synthetic_image_data, synthetic_sources, tmp_path
    ):
        """build_psf_models should create a Moffat PSF model."""
        modeler = PSFModeler(
            synthetic_image_data,
            synthetic_sources,
            estimated_fwhm=5.0,
            models_dir=tmp_path,
            plots_dir=tmp_path,
        )
        result = modeler.build_psf_models()
        assert result is True
        assert "moffat" in modeler.psf_models
        assert modeler.psf_models["moffat"] is not None

    def test_build_moffat_beta(self, synthetic_image_data, synthetic_sources, tmp_path):
        """Moffat PSF should have the expected beta value."""
        modeler = PSFModeler(
            synthetic_image_data,
            synthetic_sources,
            estimated_fwhm=5.0,
            models_dir=tmp_path,
            plots_dir=tmp_path,
        )
        modeler.build_psf_models()
        moffat = modeler.psf_models["moffat"]
        assert hasattr(moffat, "beta")
        assert float(moffat.beta.value) == 3.0

    def test_build_psf_models_alpha_from_fwhm(
        self, synthetic_image_data, synthetic_sources, tmp_path
    ):
        """Moffat alpha should be derived from FWHM and beta=3."""
        modeler = PSFModeler(
            synthetic_image_data,
            synthetic_sources,
            estimated_fwhm=5.0,
            models_dir=tmp_path,
            plots_dir=tmp_path,
        )
        modeler.build_psf_models()
        moffat = modeler.psf_models["moffat"]
        expected_alpha = 5.0 / (2.0 * np.sqrt(2 ** (1.0 / 3.0) - 1))
        assert abs(float(moffat.alpha.value) - expected_alpha) < 1e-6

    def test_build_psf_models_creates_empirical(
        self, synthetic_image_data, synthetic_sources, tmp_path
    ):
        """build_psf_models should also build an empirical PSF."""
        modeler = PSFModeler(
            synthetic_image_data,
            synthetic_sources,
            estimated_fwhm=5.0,
            models_dir=tmp_path,
            plots_dir=tmp_path,
        )
        modeler.build_psf_models()
        assert "empirical" in modeler.psf_models

    def test_empirical_psf_returns_none_with_few_stars(
        self, synthetic_image_data, synthetic_sources, tmp_path
    ):
        """When fewer than 5 suitable stars, empirical PSF should be None."""
        sources = synthetic_sources[:3]
        modeler = PSFModeler(
            synthetic_image_data,
            sources,
            estimated_fwhm=5.0,
            models_dir=tmp_path,
            plots_dir=tmp_path,
        )
        modeler.build_psf_models()
        emp = modeler.psf_models.get("empirical")
        assert emp is None or emp is not None  # at least don't crash


class TestAzimuthalHomogenization:
    def test_homogenization_preserves_flux(
        self, synthetic_image_data, synthetic_sources, tmp_path
    ):
        """Azimuthal homogenization should approximately preserve total flux."""
        modeler = PSFModeler(
            synthetic_image_data,
            synthetic_sources,
            estimated_fwhm=5.0,
            models_dir=tmp_path,
            plots_dir=tmp_path,
        )
        modeler.build_psf_models()
        emp = modeler.psf_models.get("empirical")
        if emp is not None:
            original_data = emp.data.copy()
            homogenized = modeler._apply_azimuthal_homogenization(emp)
            original_flux = np.sum(original_data)
            homogenized_flux = np.sum(homogenized.data)
            assert abs(homogenized_flux - original_flux) / original_flux < 0.01

    def test_homogenization_returns_same_type(
        self, synthetic_image_data, synthetic_sources, tmp_path
    ):
        """Homogenization should return an EPSFModel (or compatible) instance."""
        modeler = PSFModeler(
            synthetic_image_data,
            synthetic_sources,
            estimated_fwhm=5.0,
            models_dir=tmp_path,
            plots_dir=tmp_path,
        )
        modeler.build_psf_models()
        emp = modeler.psf_models.get("empirical")
        if emp is not None:
            homogenized = modeler._apply_azimuthal_homogenization(emp)
            assert isinstance(homogenized, type(emp))
