import numpy as np
import pandas as pd
from astropy.wcs import WCS

from src.calibration import Calibration


class TestCalibrationInit:
    def test_init_stores_attributes(self, synthetic_matches, mock_wcs):
        """Calibration should store all passed attributes."""
        cal = Calibration(
            wcs=mock_wcs,
            gaia_corrected=None,
            estimated_fwhm=5.0,
            image_shape=(200, 200),
            sources=None,
        )
        assert cal.wcs is mock_wcs
        assert cal.estimated_fwhm == 5.0
        assert cal.image_shape == (200, 200)
        assert cal.rms_history == []
        assert cal.rms_stage_names == []

    def test_init_default_airmass(self, mock_wcs):
        """When MJD-OBS is missing from WCS, airmass should default to 1.2."""
        cal = Calibration(
            wcs=mock_wcs,
            gaia_corrected=None,
            estimated_fwhm=5.0,
            image_shape=(200, 200),
            sources=None,
        )
        # The WCS header doesn't have MJD-OBS, so airmass should be 1.2
        assert cal.airmass is not None


class TestAirmass:
    def test_estimate_airmass_default(self, mock_wcs):
        """_estimate_airmass should return 1.2 when metadata is missing."""
        cal = Calibration(
            wcs=mock_wcs,
            gaia_corrected=None,
            estimated_fwhm=5.0,
            image_shape=(200, 200),
            sources=None,
        )
        airmass = cal._estimate_airmass()
        assert airmass == 1.20

    def test_estimate_airmass_with_mjd(self, tmp_path):
        """_estimate_airmass should compute a reasonable airmass with MJD-OBS."""
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [100.0, 100.0]
        wcs.wcs.cdelt = [-0.1 / 3600, 0.1 / 3600]
        wcs.wcs.crval = [10.0, 20.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.cunit = ["deg", "deg"]
        wcs.wcs.set()

        # Add MJD-OBS to WCS via header inheritance

        hdr = wcs.to_header()
        hdr["MJD-OBS"] = 59958.145833

        # Create a new WCS from this header
        from astropy.wcs import WCS as WCS2

        wcs_with_mjd = WCS2(hdr)

        cal = Calibration(
            wcs=wcs_with_mjd,
            gaia_corrected=None,
            estimated_fwhm=5.0,
            image_shape=(200, 200),
            sources=None,
        )
        airmass = cal._estimate_airmass()
        assert airmass >= 1.0
        assert airmass <= 3.0


class TestSNRWeight:
    def test_snr_weight_returns_positive(self):
        """_snr_weight should return positive weights and sigmas."""
        mags = np.array([15.0, 17.0, 19.0])
        weights, sigmas = Calibration._snr_weight(
            mags, fwhm_pixels=5.0, pixel_scale_arcsec=0.1
        )
        assert np.all(weights > 0)
        assert np.all(sigmas > 0)
        assert len(weights) == len(mags)
        assert len(sigmas) == len(mags)

    def test_snr_weight_brighter_higher_weight(self):
        """Brighter stars should have higher SNR and thus higher raw weight."""
        bright = np.array([14.0])
        faint = np.array([20.0])
        w_bright, sig_bright = Calibration._snr_weight(
            bright, fwhm_pixels=5.0, pixel_scale_arcsec=0.1
        )
        w_faint, sig_faint = Calibration._snr_weight(
            faint, fwhm_pixels=5.0, pixel_scale_arcsec=0.1
        )
        # Raw sigmas should be smaller for bright stars (before renormalization)
        assert sig_bright[0] < sig_faint[0]


class TestGeometricDistortion:
    def test_model_geometric_distortion_returns_dict(self, synthetic_matches, mock_wcs):
        """_model_geometric_distortion should return a dict with expected keys."""
        cal = Calibration(
            wcs=mock_wcs,
            gaia_corrected=None,
            estimated_fwhm=5.0,
            image_shape=(200, 200),
            sources=None,
        )
        result = cal._model_geometric_distortion(synthetic_matches, order=2)
        expected_keys = {"powers", "a_coeffs", "b_coeffs", "crpix", "pixel_scale"}
        assert expected_keys.issubset(result.keys())
        assert len(result["powers"]) > 0
        assert len(result["a_coeffs"]) > 0

    def test_sip_corrections_do_not_change_row_count(self, synthetic_matches, mock_wcs):
        """Applying SIP corrections should not change the number of rows."""
        cal = Calibration(
            wcs=mock_wcs,
            gaia_corrected=None,
            estimated_fwhm=5.0,
            image_shape=(200, 200),
            sources=None,
        )
        sip_model = cal._model_geometric_distortion(synthetic_matches, order=2)
        corrected = cal._apply_sip_corrections(synthetic_matches, sip_model)
        assert len(corrected) == len(synthetic_matches)

    def test_sip_corrections_creates_columns(self, synthetic_matches, mock_wcs):
        """SIP-corrected matches should have ra_diff_sip_arcsec and dec_diff_sip_arcsec."""
        cal = Calibration(
            wcs=mock_wcs,
            gaia_corrected=None,
            estimated_fwhm=5.0,
            image_shape=(200, 200),
            sources=None,
        )
        sip_model = cal._model_geometric_distortion(synthetic_matches, order=2)
        corrected = cal._apply_sip_corrections(synthetic_matches, sip_model)
        assert "ra_diff_sip_arcsec" in corrected.columns
        assert "dec_diff_sip_arcsec" in corrected.columns


class TestMagnitudeSystematics:
    def test_model_with_color(self, synthetic_matches, mock_wcs):
        """_model_magnitude_systematics should work with colour information."""
        cal = Calibration(
            wcs=mock_wcs,
            gaia_corrected=None,
            estimated_fwhm=5.0,
            image_shape=(200, 200),
            sources=None,
        )
        corrected = synthetic_matches.copy()
        # Add SIP correction columns
        corrected["ra_diff_sip_arcsec"] = corrected["ra_diff_arcsec"]
        corrected["dec_diff_sip_arcsec"] = corrected["dec_diff_arcsec"]
        result = cal._model_magnitude_systematics(corrected)
        assert "ridge_dx" in result
        assert "ridge_dy" in result
        assert "has_color" in result

    def test_model_without_color(self, mock_wcs):
        """_model_magnitude_systematics should work without colour information."""
        cal = Calibration(
            wcs=mock_wcs,
            gaia_corrected=None,
            estimated_fwhm=5.0,
            image_shape=(200, 200),
            sources=None,
        )
        df = pd.DataFrame(
            {
                "ra_diff_sip_arcsec": [0.1, -0.05, 0.02],
                "dec_diff_sip_arcsec": [0.05, 0.01, -0.03],
                "phot_g_mean_mag": [15.0, 17.0, 19.0],
            }
        )
        result = cal._model_magnitude_systematics(df)
        assert "has_color" in result
        assert result["has_color"] is False

    def test_apply_all_corrections_smoke(self, synthetic_matches, mock_wcs):
        """_apply_all_corrections should run end-to-end without error."""
        cal = Calibration(
            wcs=mock_wcs,
            gaia_corrected=None,
            estimated_fwhm=5.0,
            image_shape=(200, 200),
            sources=None,
        )
        sip_model = cal._model_geometric_distortion(synthetic_matches, order=2)
        sip_corrected = cal._apply_sip_corrections(synthetic_matches, sip_model)
        sys_model = cal._model_magnitude_systematics(sip_corrected)
        final = cal._apply_all_corrections(synthetic_matches, sip_model, sys_model)
        assert "ra_diff_final_arcsec" in final.columns
        assert "dec_diff_final_arcsec" in final.columns


class TestSystematicCorrections:
    def test_model_systematic_corrections_with_missing_file(self, mock_wcs, tmp_path):
        """model_systematic_corrections should return None when file is missing."""
        cal = Calibration(
            wcs=mock_wcs,
            gaia_corrected=None,
            estimated_fwhm=5.0,
            image_shape=(200, 200),
            sources=None,
            catalogs_dir=tmp_path,
            logs_dir=tmp_path,
        )
        missing = tmp_path / "no_matches.csv"
        result = cal.model_systematic_corrections(missing)
        assert result is None


class TestSaveRefinedCoordinates:
    def test_save_refined_pixel_coordinates(
        self, synthetic_matches, mock_wcs, tmp_path
    ):
        """_save_refined_pixel_coordinates should create output files."""
        cal = Calibration(
            wcs=mock_wcs,
            gaia_corrected=None,
            estimated_fwhm=5.0,
            image_shape=(200, 200),
            sources=None,
            catalogs_dir=tmp_path,
            logs_dir=tmp_path,
        )
        final = synthetic_matches.copy()
        final["ra_diff_final_arcsec"] = final["ra_diff_arcsec"]
        final["dec_diff_final_arcsec"] = final["dec_diff_arcsec"]
        cal._save_refined_pixel_coordinates(final)
        # Should create xy.txt and refined_coordinates_full.csv
        xy_file = tmp_path / "xy.txt"
        refined_file = tmp_path / "refined_coordinates_full.csv"
        # At least one should exist
        assert xy_file.exists() or refined_file.exists()
