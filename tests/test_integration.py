"""
Integration tests that exercise the pipeline on synthetic data.
These tests verify that the major components work together correctly
and produce sensible output files.
"""

import numpy as np
import pandas as pd

from src.data_io import DataIO
from src.detection import SourceDetector
from src.photometry import Photometry
from src.psf_modeling import PSFModeler


class TestDetectionToPhotometry:
    """Integration: synthetic image -> detection -> PSF -> photometry."""

    def test_detection_to_aperture_photometry(self, synthetic_fits_file, tmp_path):
        """
        Load a synthetic FITS, run source detection, then aperture photometry.
        All steps should complete without error and produce consistent results.
        """
        # Step 1: Load FITS
        data_io = DataIO()
        data_io.load_fits(synthetic_fits_file)
        assert data_io.image_data is not None
        assert data_io.wcs is not None

        # Step 2: Source detection
        detector = SourceDetector(
            data_io.image_data, data_io.wcs, data_io.pixel_scale_arcsec
        )
        result = detector.detect_sources()
        assert result is True
        assert detector.sources is not None
        assert len(detector.sources) >= 5

        # Step 3: Save detected sources
        sources_file = tmp_path / "detected.csv"
        detector.save_detected_sources(sources_file)
        assert sources_file.exists()
        df = pd.read_csv(sources_file)
        assert len(df) == len(detector.sources)

        # Step 4: Build PSF models
        modeler = PSFModeler(
            data_io.image_data,
            detector.sources,
            estimated_fwhm=detector.estimated_fwhm,
            models_dir=tmp_path,
            plots_dir=tmp_path,
        )
        psf_result = modeler.build_psf_models()
        assert psf_result is True
        assert "moffat" in modeler.psf_models

        # Step 5: Aperture photometry
        phot = Photometry(
            data_io.image_data,
            detector.sources,
            estimated_fwhm=detector.estimated_fwhm,
            psf_models=modeler.psf_models,
            wcs=data_io.wcs,
            catalogs_dir=tmp_path,
        )
        results = phot.perform_photometry(methods=["aperture", "moffat"])
        assert "aperture" in results
        assert "aperture" in phot.photometry_results

        # Verify non-zero fluxes
        ap_result = results["aperture"]
        assert np.all(ap_result["flux"] > 0)

    def test_detection_output_consistency(
        self, synthetic_image_data, star_positions, tmp_path
    ):
        """
        Test that detected source positions are reasonably close
        to the input star positions (within a few pixels).
        """
        data_io = DataIO()
        from astropy.wcs import WCS

        # Create a minimal WCS
        w = WCS(naxis=2)
        w.wcs.crpix = [100.0, 100.0]
        w.wcs.cdelt = [-0.1 / 3600, 0.1 / 3600]
        w.wcs.crval = [10.0, 20.0]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.cunit = ["deg", "deg"]
        data_io.wcs = w

        detector = SourceDetector(synthetic_image_data, wcs=data_io.wcs)
        detector.detect_sources()

        # Most of the 15 stars should be detected (within reason)
        assert len(detector.sources) >= 5

        # Check that detected sources cluster near input positions
        input_x = np.array([p[0] for p in star_positions])
        input_y = np.array([p[1] for p in star_positions])

        detected_x = detector.sources["x_centroid"]
        detected_y = detector.sources["y_centroid"]

        # For each detected source, find nearest input star
        for dx, dy in zip(detected_x, detected_y):
            dists = np.sqrt((input_x - dx) ** 2 + (input_y - dy) ** 2)
            min_dist = np.min(dists)
            # Should be within 5 pixels of an input star
            assert min_dist < 5.0, (
                f"Detected source at ({dx:.1f}, {dy:.1f}) is {min_dist:.1f} "
                f"pixels from nearest input star"
            )


class TestPipelineWithMockData:
    """End-to-end pipeline simulation with mocked external calls."""

    def test_full_pipeline_no_crash(self, synthetic_fits_file, tmp_path):
        """
        Run a full detection -> modeling -> photometry sequence
        with synthetic data. This verifies all components interoperate.
        """
        data_io = DataIO()
        data_io.load_fits(synthetic_fits_file)

        detector = SourceDetector(
            data_io.image_data, data_io.wcs, data_io.pixel_scale_arcsec
        )
        assert detector.detect_sources() is True

        modeler = PSFModeler(
            data_io.image_data,
            detector.sources,
            estimated_fwhm=detector.estimated_fwhm,
            models_dir=tmp_path,
            plots_dir=tmp_path,
        )
        assert modeler.build_psf_models() is True

        phot = Photometry(
            data_io.image_data,
            detector.sources,
            estimated_fwhm=detector.estimated_fwhm,
            psf_models=modeler.psf_models,
            wcs=data_io.wcs,
            catalogs_dir=tmp_path,
        )
        results = phot.perform_photometry(methods=["aperture", "moffat"])
        assert "aperture" in results
        assert len(results["aperture"]) > 0

    def test_pipeline_saves_output_files(self, synthetic_fits_file, tmp_path):
        """
        Verify that pipeline steps correctly write their output files.
        """
        data_io = DataIO()
        data_io.load_fits(synthetic_fits_file)

        detector = SourceDetector(
            data_io.image_data, data_io.wcs, data_io.pixel_scale_arcsec
        )
        detector.detect_sources()

        # Save detected sources
        sources_file = tmp_path / "all_detected_sources.csv"
        detector.save_detected_sources(sources_file)
        assert sources_file.exists()

        # Build PSF models (should save .fits files)
        modeler = PSFModeler(
            data_io.image_data,
            detector.sources,
            estimated_fwhm=detector.estimated_fwhm,
            models_dir=tmp_path,
            plots_dir=tmp_path,
        )
        modeler.build_psf_models()
        # ePSF .fits files may not be saved if <5 suitable stars
        # but the Moffat model should be in memory

        # Aperture photometry and save coordinates
        phot = Photometry(
            data_io.image_data,
            detector.sources,
            estimated_fwhm=detector.estimated_fwhm,
            psf_models=modeler.psf_models,
            wcs=data_io.wcs,
            catalogs_dir=tmp_path,
        )
        phot.perform_photometry(methods=["aperture"])
        phot._save_coordinates("aperture")

        coord_file = tmp_path / "xy_radec_aperture.txt"
        assert coord_file.exists()
        lines = coord_file.read_text().strip().split("\n")
        assert len(lines) == len(detector.sources)
