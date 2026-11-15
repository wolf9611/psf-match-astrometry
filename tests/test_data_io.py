from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from src.data_io import DataIO

# ---------------------------------------------------------------------------
# FITS loading
# ---------------------------------------------------------------------------


class TestLoadFITS:
    def test_load_fits_basic(self, synthetic_fits_file):
        """Basic FITS loading populates image_data and header."""
        data_io = DataIO()
        data_io.load_fits(synthetic_fits_file)
        assert data_io.image_data is not None
        assert data_io.header is not None
        assert len(data_io.image_data.shape) == 2

    def test_load_fits_shape(self, synthetic_fits_file, synthetic_image_data):
        """Loaded image shape should match the written data."""
        data_io = DataIO()
        data_io.load_fits(synthetic_fits_file)
        assert data_io.image_data.shape == synthetic_image_data.shape

    def test_load_fits_parses_instrument(self, synthetic_fits_file):
        """Instrument name should be parsed from the header."""
        data_io = DataIO()
        data_io.load_fits(synthetic_fits_file)
        assert data_io.instrument == "DECam"

    def test_load_fits_parses_pixel_scale(self, synthetic_fits_file):
        """Pixel scale should be computed from WCS."""
        data_io = DataIO()
        data_io.load_fits(synthetic_fits_file)
        # 0.1 arcsec/px = CDELT 2.77778e-05 deg * 3600
        assert data_io.pixel_scale_arcsec is not None
        assert abs(data_io.pixel_scale_arcsec - 0.1) < 1e-4

    def test_load_fits_file_not_found(self, tmp_path):
        """A missing FITS file should raise FileNotFoundError."""
        data_io = DataIO()
        missing = tmp_path / "no_such_file.fits"
        with pytest.raises(FileNotFoundError):
            data_io.load_fits(missing)

    def test_load_fits_eso_header(self, synthetic_fits_eso_file):
        """ESO-style headers should be parsed correctly."""
        data_io = DataIO()
        data_io.load_fits(synthetic_fits_eso_file)
        assert data_io.instrument == "FORS2"
        assert data_io.telescope != "UNKNOWN"
        assert data_io.filter_name == "R_SPECIAL"

    def test_load_fits_eso_coordinate_parsing(self, synthetic_fits_eso_file):
        """RA/Dec in sexagesimal from ESO headers should be parsed."""
        data_io = DataIO()
        data_io.load_fits(synthetic_fits_eso_file)
        assert data_io.ra_center is not None
        assert data_io.dec_center is not None
        # 05:30:00.0 = 82.5 deg, -30:00:00.0 = -30.0 deg
        assert abs(data_io.ra_center - 82.5) < 0.01
        assert abs(data_io.dec_center - (-30.0)) < 0.01


class TestMEF:
    def test_list_mef_extensions(self, synthetic_fits_mef_file):
        """list_mef_extensions should return all image-like extensions."""
        extensions = DataIO.list_mef_extensions(synthetic_fits_mef_file)
        # Primary (no data) + 2 image HDUs
        assert len(extensions) >= 2

    def test_load_fits_mef_auto_detect(self, synthetic_fits_mef_file):
        """Loading a MEF should auto-discover the first 2D extension."""
        data_io = DataIO()
        data_io.load_fits(synthetic_fits_mef_file)
        assert data_io.image_data is not None

    def test_load_fits_mef_specific_extension(self, synthetic_fits_mef_file):
        """Loading a specific extension should return that extension's data."""
        data_io = DataIO()
        data_io.load_fits(synthetic_fits_mef_file, extension=2)
        assert data_io.image_data.shape == (60, 60)


class Test3D:
    def test_load_fits_3d_squeeze(self, synthetic_fits_file):
        """A 3D array with singleton dimension should be squeezed to 2D."""
        data_io = DataIO()
        # Write a 3D array and load it
        with fits.open(synthetic_fits_file, mode="update") as hdul:
            primary = hdul[0]
            original = primary.data
            # Wrap in singleton dim
            primary.data = original[np.newaxis, :, :]
            primary.header["NAXIS"] = 3

        data_io.load_fits(synthetic_fits_file)
        assert len(data_io.image_data.shape) == 2


class TestObservationTime:
    def test_mjd_obs_parsed(self, synthetic_fits_file):
        """MJD-OBS should be parsed correctly."""
        data_io = DataIO()
        data_io.load_fits(synthetic_fits_file)
        assert data_io.obstime is not None
        assert abs(data_io.obstime.value - 59958.145833) < 0.001

    def test_observation_time_from_date_obs(self, tmp_path, synthetic_image_data):
        """DATE-OBS should be used when MJD-OBS is missing."""
        file_path = tmp_path / "no_mjd.fits"
        hdr = fits.Header()
        hdr["DATE-OBS"] = "2023-01-15T03:30:00"
        hdr["CRVAL1"] = 10.0
        hdr["CRVAL2"] = 20.0
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CRPIX1"] = 100.0
        hdr["CRPIX2"] = 100.0
        hdr["CD1_1"] = -2.77778e-05
        hdr["CD1_2"] = 0.0
        hdr["CD2_1"] = 0.0
        hdr["CD2_2"] = 2.77778e-05
        hdu = fits.PrimaryHDU(data=synthetic_image_data, header=hdr)
        hdu.writeto(file_path, overwrite=True)

        data_io = DataIO()
        data_io.load_fits(file_path)
        assert data_io.obstime is not None


# ---------------------------------------------------------------------------
# Gaia catalog loading
# ---------------------------------------------------------------------------


class TestLoadGaiaCatalog:
    def test_load_gaia_catalog_from_csv(self, synthetic_gaia_catalog, tmp_path):
        """load_gaia_catalog should load a CSV with matching columns."""
        catalog_path = tmp_path / "gaia.csv"
        synthetic_gaia_catalog.to_csv(catalog_path, index=False)
        data_io = DataIO()
        data_io.ra_center = synthetic_gaia_catalog["ra"].mean()
        data_io.dec_center = synthetic_gaia_catalog["dec"].mean()
        result = data_io.load_gaia_catalog(catalog_path)
        assert result is not None
        assert len(result["ra"]) == len(synthetic_gaia_catalog)

    def test_cache_validation_triggers_redownload(self, tmp_path):
        """When cached catalog is for a different sky area, redownload should occur."""
        catalog_path = tmp_path / "gaia.csv"
        df = pd.DataFrame(
            {"ra": [100.0], "dec": [100.0], "pmra": [0.0], "pmdec": [0.0]}
        )
        df.to_csv(catalog_path, index=False)

        data_io = DataIO()
        data_io.ra_center = 10.0
        data_io.dec_center = 20.0

        with patch.object(DataIO, "download_gaia_catalog") as mock_dl:

            def mock_download(path):
                new_df = pd.DataFrame(
                    {
                        "ra": [10.0],
                        "dec": [20.0],
                        "pmra": [0.0],
                        "pmdec": [0.0],
                    }
                )
                new_df.to_csv(path, index=False)

            mock_dl.side_effect = mock_download
            result = data_io.load_gaia_catalog(catalog_path)
            mock_dl.assert_called_once()
            assert result["ra"][0] == 10.0

    def test_load_gaia_catalog_skips_download_if_cached_matches(
        self, synthetic_gaia_catalog, tmp_path
    ):
        """When cached catalog matches the sky area, download should be skipped."""
        catalog_path = tmp_path / "gaia_cached.csv"
        synthetic_gaia_catalog.to_csv(catalog_path, index=False)

        data_io = DataIO()
        data_io.ra_center = synthetic_gaia_catalog["ra"].mean()
        data_io.dec_center = synthetic_gaia_catalog["dec"].mean()

        with patch.object(DataIO, "download_gaia_catalog") as mock_dl:
            data_io.load_gaia_catalog(catalog_path)
            mock_dl.assert_not_called()

    def test_load_gaia_catalog_returns_dict_like(self, tmp_path):
        """load_gaia_catalog should return a dict-like object."""
        df = pd.DataFrame(
            {
                "ra": [10.0],
                "dec": [20.0],
                "pmra": [0.0],
                "pmdec": [0.0],
                "pmra_error": [0.5],
                "pmdec_error": [0.5],
                "ra_error": [0.1],
                "dec_error": [0.1],
                "ref_epoch": [2016.0],
                "phot_g_mean_mag": [15.0],
                "bp_rp": [0.5],
                "ruwe": [1.0],
            }
        )
        path = tmp_path / "test_gaia.csv"
        df.to_csv(path, index=False)
        data_io = DataIO()
        data_io.ra_center = 10.0
        data_io.dec_center = 20.0
        result = data_io.load_gaia_catalog(path)
        assert result is not None
        assert "ra" in result
        assert "dec" in result

    def test_load_gaia_catalog_includes_error_columns(
        self, synthetic_gaia_catalog, tmp_path
    ):
        """load_gaia_catalog should include new astrometric error columns."""
        catalog_path = tmp_path / "gaia_with_errors.csv"
        synthetic_gaia_catalog.to_csv(catalog_path, index=False)
        data_io = DataIO()
        data_io.ra_center = synthetic_gaia_catalog["ra"].mean()
        data_io.dec_center = synthetic_gaia_catalog["dec"].mean()
        result = data_io.load_gaia_catalog(catalog_path)
        assert result is not None
        assert "ra_error" in result
        assert "dec_error" in result
        assert "pmra_error" in result
        assert "pmdec_error" in result
        assert len(result["ra_error"]) == len(synthetic_gaia_catalog)


# ---------------------------------------------------------------------------
# Milliquas catalog loading
# ---------------------------------------------------------------------------


class TestMilliquas:
    def test_load_milliquas_catalog(self, tmp_path):
        """load_milliquas_catalog should load a Milliquas CSV."""
        df = pd.DataFrame(
            {
                "ra": [10.0],
                "dec": [20.0],
                "Rmag": [15.0],
                "Name": ["QSO"],
            }
        )
        path = tmp_path / "milliquas.csv"
        df.to_csv(path, index=False)
        data_io = DataIO()
        data_io.ra_center = 10.0
        data_io.dec_center = 20.0
        result = data_io.load_milliquas_catalog(path)
        assert result is not None
        assert len(result["ra"]) > 0

    def test_load_milliquas_raises_on_missing_file(self, tmp_path):
        """load_milliquas_catalog should raise on missing file when auto_download is False."""
        data_io = DataIO()
        missing = tmp_path / "no_milli.csv"
        with patch.object(DataIO, "download_milliquas_catalog") as mock_dl:
            mock_dl.return_value = None
            with pytest.raises((FileNotFoundError, ValueError)):
                data_io.load_milliquas_catalog(missing)


# ---------------------------------------------------------------------------
# Overscan / bias correction
# ---------------------------------------------------------------------------


class TestOverscan:
    def test_overscan_skipped_for_reduced_data(self, tmp_path, synthetic_image_data):
        """_apply_basic_calibration should be a no-op for pre-reduced data."""
        file_path = tmp_path / "reduced.fits"
        hdr = fits.Header()
        hdr["INSTRUME"] = "DECam"
        hdr["CRVAL1"] = 10.0
        hdr["CRVAL2"] = 20.0
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CRPIX1"] = 100.0
        hdr["CRPIX2"] = 100.0
        hdr["CD1_1"] = -2.77778e-05
        hdr["CD1_2"] = 0.0
        hdr["CD2_1"] = 0.0
        hdr["CD2_2"] = 2.77778e-05
        hdu = fits.PrimaryHDU(data=synthetic_image_data, header=hdr)
        hdu.writeto(file_path, overwrite=True)

        data_io = DataIO()
        data_io.load_fits(file_path)
        median_before = float(np.median(synthetic_image_data))
        median_after = float(np.median(data_io.image_data))
        # Should be very similar (clipped noise only)
        assert abs(median_after - median_before) < 10.0


# ---------------------------------------------------------------------------
# Header value helpers
# ---------------------------------------------------------------------------


class TestHeaderHelpers:
    def test_get_header_value_found(self, synthetic_fits_file):
        """_get_header_value should return the value for an existing key."""
        data_io = DataIO()
        data_io.load_fits(synthetic_fits_file)
        val = data_io._get_header_value(["INSTRUME", "INSTRUMENT"])
        assert val == "DECam"

    def test_get_header_value_missing(self, tmp_path, synthetic_image_data):
        """_get_header_value should return the default for a missing key."""
        file_path = tmp_path / "minimal.fits"
        hdr = fits.Header()
        hdr["CRVAL1"] = 10.0
        hdr["CRVAL2"] = 20.0
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CRPIX1"] = 100.0
        hdr["CRPIX2"] = 100.0
        hdr["CD1_1"] = -2.77778e-05
        hdr["CD1_2"] = 0.0
        hdr["CD2_1"] = 0.0
        hdr["CD2_2"] = 2.77778e-05
        hdu = fits.PrimaryHDU(data=synthetic_image_data, header=hdr)
        hdu.writeto(file_path, overwrite=True)

        data_io = DataIO()
        data_io.load_fits(file_path)
        val = data_io._get_header_value(["NONEXISTENT_KEY"], default="fallback")
        assert val == "fallback"
