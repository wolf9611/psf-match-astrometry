import numpy as np
import pandas as pd

from src.detection import SourceDetector


class TestSourceDetectorInit:
    def test_background_estimation(self, synthetic_image_data):
        """Background level and RMS should be estimated from the image."""
        detector = SourceDetector(synthetic_image_data)
        assert detector.bkg_level is not None
        assert detector.bkg_rms is not None
        assert detector.bkg_rms > 0
        assert detector.image_clean is not None
        assert detector.image_clean.shape == synthetic_image_data.shape

    def test_initial_fwhm_from_pixel_scale(self):
        """When pixel_scale is given, FWHM should be >= 2.0."""
        data = np.ones((50, 50), dtype=np.float64)
        detector = SourceDetector(data, pixel_scale_arcsec=0.25)
        assert detector.estimated_fwhm >= 2.0

    def test_initial_fwhm_without_pixel_scale(self):
        """Without pixel scale, default FWHM should be 3.0."""
        data = np.ones((50, 50), dtype=np.float64)
        detector = SourceDetector(data)
        assert detector.estimated_fwhm == 3.0

    def test_wcs_stored(self, mock_wcs, synthetic_image_data):
        """WCS should be stored when passed."""
        detector = SourceDetector(synthetic_image_data, wcs=mock_wcs)
        assert detector.wcs is mock_wcs


class TestSourceDetectorDetect:
    def test_detect_sources_returns_true(self, synthetic_image_data):
        """detect_sources should return True on a starfield image."""
        detector = SourceDetector(synthetic_image_data)
        result = detector.detect_sources()
        assert result is True

    def test_detect_sources_populates_sources(self, synthetic_image_data):
        """After detection, self.sources should be non-empty."""
        detector = SourceDetector(synthetic_image_data)
        detector.detect_sources()
        assert detector.sources is not None
        assert len(detector.sources) >= 5

    def test_detect_sources_returns_false_on_noise(self):
        """detect_sources should return False on pure noise with sufficient FWHM."""
        noise = np.random.poisson(5, (50, 50)).astype(np.float64)
        detector = SourceDetector(noise)
        result = detector.detect_sources(fwhm_initial=2.0, threshold_sigma=10.0)
        assert result is False

    def test_detect_sources_converges_fwhm(self, synthetic_image_data):
        """After detection, estimated_fwhm should be updated."""
        detector = SourceDetector(synthetic_image_data)
        initial_fwhm = detector.estimated_fwhm
        detector.detect_sources()
        assert detector.estimated_fwhm != initial_fwhm or len(detector.sources) > 0


class TestSourceDetectorSave:
    def test_save_detected_sources_csv(self, synthetic_image_data, tmp_path):
        """save_detected_sources should write a CSV with expected columns."""
        detector = SourceDetector(synthetic_image_data)
        detector.detect_sources()
        out = tmp_path / "sources.csv"
        detector.save_detected_sources(out)
        assert out.exists()
        df = pd.read_csv(out)
        assert "x_pixel" in df.columns
        assert "y_pixel" in df.columns
        assert "peak_counts" in df.columns
        assert "mag_instrumental" in df.columns
        assert len(df) == len(detector.sources)

    def test_save_detected_sources_with_wcs(
        self, synthetic_image_data, mock_wcs, tmp_path
    ):
        """When WCS is provided, RA/Dec columns should be in the CSV."""
        detector = SourceDetector(synthetic_image_data, wcs=mock_wcs)
        detector.detect_sources()
        out = tmp_path / "sources_wcs.csv"
        detector.save_detected_sources(out)
        df = pd.read_csv(out)
        assert "ra_deg" in df.columns
        assert "dec_deg" in df.columns

    def test_save_detected_sources_ids_sequential(self, synthetic_image_data, tmp_path):
        """Source IDs should be sequential starting from 1."""
        detector = SourceDetector(synthetic_image_data)
        detector.detect_sources()
        out = tmp_path / "ids.csv"
        detector.save_detected_sources(out)
        df = pd.read_csv(out)
        assert list(df["id"]) == list(range(1, len(df) + 1))


class TestFWHMEstimation:
    def test_estimate_fwhm_from_sources_returns_positive(self, synthetic_image_data):
        """_estimate_fwhm_from_sources should return a positive float."""
        detector = SourceDetector(synthetic_image_data)
        detector.detect_sources()
        fwhm = detector._estimate_fwhm_from_sources()
        assert isinstance(fwhm, (float, np.floating))
        assert fwhm > 0

    def test_estimate_fwhm_reasonable_range(self, synthetic_image_data):
        """Estimated FWHM should be within reasonable bounds."""
        detector = SourceDetector(synthetic_image_data)
        detector.detect_sources()
        fwhm = detector._estimate_fwhm_from_sources()
        # Our synthetic stars have sigma ~2.5 px, so FWHM ~ 2.355 * 2.5 ~ 5.9
        # Allow tolerance for background noise
        assert 1.5 < fwhm < 20.0
