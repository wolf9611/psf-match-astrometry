from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.table import QTable
from astropy.wcs import WCS

from src.config import config as pipeline_config

# ---------------------------------------------------------------------------
# Synthetic 2D image with Gaussian stars
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_image_data():
    """200x200 image with ~15 Gaussian stars at known pixel positions."""
    np.random.seed(42)
    ny, nx = 200, 200
    image = np.random.poisson(50, (ny, nx)).astype(np.float64)

    star_positions = [
        (30, 40, 800, 2.5, 2.5),
        (60, 80, 1200, 2.8, 2.6),
        (90, 30, 600, 2.4, 2.4),
        (120, 140, 1500, 2.6, 2.7),
        (150, 170, 900, 2.5, 2.5),
        (40, 150, 1100, 2.7, 2.8),
        (70, 110, 700, 2.3, 2.4),
        (100, 60, 1300, 2.6, 2.5),
        (160, 30, 500, 2.4, 2.5),
        (180, 100, 1000, 2.8, 2.6),
        (50, 180, 850, 2.5, 2.5),
        (130, 20, 950, 2.6, 2.7),
        (20, 100, 1100, 2.4, 2.4),
        (170, 150, 750, 2.7, 2.6),
        (110, 90, 650, 2.5, 2.5),
    ]

    for x0, y0, amp, sx, sy in star_positions:
        x = np.arange(nx)
        y = np.arange(ny)
        x_grid, y_grid = np.meshgrid(x, y)
        g = amp * np.exp(
            -((x_grid - x0) ** 2) / (2 * sx**2) - ((y_grid - y0) ** 2) / (2 * sy**2)
        )
        image += g

    return image


@pytest.fixture
def star_positions():
    """The (x, y, amplitude) tuples used in synthetic_image_data."""
    return [
        (30, 40, 800),
        (60, 80, 1200),
        (90, 30, 600),
        (120, 140, 1500),
        (150, 170, 900),
        (40, 150, 1100),
        (70, 110, 700),
        (100, 60, 1300),
        (160, 30, 500),
        (180, 100, 1000),
        (50, 180, 850),
        (130, 20, 950),
        (20, 100, 1100),
        (170, 150, 750),
        (110, 90, 650),
    ]


# ---------------------------------------------------------------------------
# Synthetic FITS file with DES-like headers
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_fits_file(tmp_path, synthetic_image_data):
    """Writes a temporary FITS file with DES/NOIRLab-style header keywords."""
    file_path = tmp_path / "test_image.fits"

    hdr = fits.Header()
    hdr["INSTRUME"] = "DECam"
    hdr["TELESCOP"] = "CTIO 4.0m"
    hdr["FILTER"] = "r"
    hdr["EXPTIME"] = 90.0
    hdr["GAIN"] = 4.0
    hdr["RDNOISE"] = 10.0
    hdr["SATURATE"] = 60000.0
    hdr["DATE-OBS"] = "2023-01-15T03:30:00"
    hdr["MJD-OBS"] = 59958.145833
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
    hdr["AIRMASS"] = 1.2
    hdr["FWHM"] = 1.5

    hdu = fits.PrimaryHDU(data=synthetic_image_data, header=hdr)
    hdu.writeto(file_path, overwrite=True)
    return file_path


@pytest.fixture
def synthetic_fits_eso_file(tmp_path, synthetic_image_data):
    """Writes a temporary FITS file with ESO-style hierarchical headers."""
    file_path = tmp_path / "test_eso.fits"

    hdr = fits.Header()
    hdr["HIERARCH ESO INS ID"] = "FORS2"
    hdr["HIERARCH ESO TEL ID"] = "ESO-VLT-U2"
    hdr["HIERARCH ESO INS FILT1 NAME"] = "R_SPECIAL"
    hdr["HIERARCH ESO DET EXP1 TIME"] = 120.0
    hdr["HIERARCH ESO DET OUT1 GAIN"] = 2.0
    hdr["HIERARCH ESO DET OUT1 SATUR"] = 50000.0
    hdr["HIERARCH ESO OBS RA"] = "05:30:00.0"
    hdr["HIERARCH ESO OBS DEC"] = "-30:00:00.0"
    hdr["DATE-OBS"] = "2022-06-01T08:00:00"
    hdr["CRVAL1"] = 82.5
    hdr["CRVAL2"] = -30.0
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
    return file_path


@pytest.fixture
def synthetic_fits_mef_file(tmp_path):
    """Writes a minimal MEF FITS file with multiple image extensions."""
    file_path = tmp_path / "test_mef.fits"

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "DECam"

    data1 = np.ones((50, 50), dtype=np.float32)
    data2 = np.ones((60, 60), dtype=np.float32)

    ext1 = fits.ImageHDU(data1, name="EXT1")
    ext2 = fits.ImageHDU(data2, name="EXT2")

    hdul = fits.HDUList([primary, ext1, ext2])
    hdul.writeto(file_path, overwrite=True)
    return file_path


# ---------------------------------------------------------------------------
# Synthetic Gaia catalog
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_gaia_catalog():
    """DataFrame mimicking Gaia DR3 columns used by the pipeline."""
    np.random.seed(42)
    n = 50
    # Place Gaia sources in a ~0.3 deg region around (10, 20)
    ra = 10.0 + np.random.uniform(-0.15, 0.15, n)
    dec = 20.0 + np.random.uniform(-0.15, 0.15, n)
    np.random.seed(42)
    return pd.DataFrame(
        {
            "ra": ra,
            "dec": dec,
            "pmra": np.random.normal(0, 2, n),
            "pmdec": np.random.normal(0, 2, n),
            "pmra_error": np.random.uniform(0.1, 1.0, n),
            "pmdec_error": np.random.uniform(0.1, 1.0, n),
            "ra_error": np.random.uniform(0.02, 0.5, n),
            "dec_error": np.random.uniform(0.02, 0.5, n),
            "ref_epoch": np.full(n, 2016.0),
            "phot_g_mean_mag": np.random.uniform(14, 20, n),
            "bp_rp": np.random.uniform(0.3, 1.5, n),
            "ruwe": np.random.uniform(0.8, 1.3, n),
        }
    )


# ---------------------------------------------------------------------------
# Synthetic DAOStarFinder source table
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_sources(star_positions):
    """QTable mimicking DAOStarFinder output."""
    n = len(star_positions)
    x = np.array([p[0] for p in star_positions], dtype=np.float64)
    y = np.array([p[1] for p in star_positions], dtype=np.float64)
    # Add small offsets to simulate centroid differences
    x += np.random.RandomState(42).normal(0, 0.1, n)
    y += np.random.RandomState(42).normal(0, 0.1, n)
    peak = np.array([p[2] for p in star_positions], dtype=np.float64)

    sources = QTable()
    sources["id"] = np.arange(1, n + 1)
    sources["x_centroid"] = x
    sources["y_centroid"] = y
    sources["peak"] = peak
    sources["flux"] = peak * 50.0
    sources["sharpness"] = np.full(n, 0.5)
    sources["roundness"] = np.full(n, 0.0)
    return sources


# ---------------------------------------------------------------------------
# Mock WCS
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_wcs():
    """Simple TAN-projection WCS centred at (10, 20) with 0.1 arcsec/px."""
    w = WCS(naxis=2)
    w.wcs.crpix = [100.0, 100.0]
    w.wcs.cdelt = [-0.1 / 3600, 0.1 / 3600]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]
    return w


@pytest.fixture
def mock_wcs_fors2():
    """WCS for a FORS2-like field centred at (82.5, -30.0)."""
    w = WCS(naxis=2)
    w.wcs.crpix = [100.0, 100.0]
    w.wcs.cdelt = [0.1 / 3600, -0.1 / 3600]
    w.wcs.crval = [82.5, -30.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]
    return w


# ---------------------------------------------------------------------------
# Synthetic matched catalog (as produced by Astrometry)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_matches(synthetic_gaia_catalog, star_positions, mock_wcs):
    """DataFrame structured like the matched catalog produced by Astrometry."""
    np.random.seed(42)
    n = min(len(star_positions), 15)

    ra_gaia = synthetic_gaia_catalog["ra"].values[:n]
    dec_gaia = synthetic_gaia_catalog["dec"].values[:n]

    sep_arcsec = np.random.uniform(0.01, 0.5, n)
    cos_dec = np.cos(np.radians(dec_gaia))
    ra_psf = ra_gaia + sep_arcsec / 3600 / cos_dec
    dec_psf = dec_gaia + sep_arcsec / 3600

    x_pix, y_pix = mock_wcs.all_world2pix(ra_psf, dec_psf, 0)

    return pd.DataFrame(
        {
            "ra_gaia_corr": ra_gaia,
            "dec_gaia_corr": dec_gaia,
            "ra_psf": ra_psf,
            "dec_psf": dec_psf,
            "separation_arcsec": sep_arcsec,
            "phot_g_mean_mag": synthetic_gaia_catalog["phot_g_mean_mag"].values[:n],
            "bp_rp": synthetic_gaia_catalog["bp_rp"].values[:n],
            "ruwe": synthetic_gaia_catalog["ruwe"].values[:n],
            "ra_diff_arcsec": (ra_psf - ra_gaia) * 3600 * cos_dec,
            "dec_diff_arcsec": (dec_psf - dec_gaia) * 3600,
            "x_pixel": x_pix,
            "y_pixel": y_pix,
        }
    )


# ---------------------------------------------------------------------------
# Config context manager
# ---------------------------------------------------------------------------


@contextmanager
def config_overrides(**kwargs):
    """Temporarily set attributes on the global config object."""
    saved = {}
    for key, value in kwargs.items():
        if hasattr(pipeline_config, key):
            saved[key] = getattr(pipeline_config, key)
            object.__setattr__(pipeline_config, key, value)
    try:
        yield
    finally:
        for key, value in saved.items():
            object.__setattr__(pipeline_config, key, value)


@pytest.fixture
def decam_config():
    """Temporarily set config to DECam-like values."""
    with config_overrides(
        image_file=Path("/fake/input/DECam/image.fits"),
        gaia_catalog_file=Path("/fake/input/DECam/gaia_catalog.csv"),
        gain=4.0,
        readnoise=10.0,
        saturation_level=60000.0,
        extension=0,
    ):
        yield pipeline_config
