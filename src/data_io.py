"""
Data Input/Output operations for FITS files and astronomical catalog queries.

Supports both classic (DES/NOIRLab) and ESO hierarchical header conventions
with Multi-Extension FITS (MEF) capability.
"""

import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS

from src.config import config
from src.logger import get_logger

logger = get_logger("data_io")


class DataIO:
    """
    Handles file system and network I/O for the pipeline.

    Supports single-extension and Multi-Extension FITS; for mosaic cameras
    the caller should iterate ``process_mef_extensions()`` to obtain one
    ``DataIO`` instance per amplifier/extension.
    """

    def __init__(self):
        self.image_data = None
        self.header = None
        self.wcs = None
        self.obstime = None
        self.ra_center = None
        self.dec_center = None
        self.saturation_level = config.saturation_level
        self.gain = config.gain
        self.readnoise = config.readnoise
        self.instrument = "UNKNOWN"
        self.telescope = "UNKNOWN"
        self.filter_name = "UNKNOWN"
        self.pixel_scale_arcsec = None
        self.airmass = None
        self.seeing_arcsec = None
        self.exptime = None
        # Overscan handling for raw frames (FORS2, EFOSC2, etc.)
        self.has_overscan = False
        self.overscan_cols = None  # (start, end) of overscan region in X

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def load_fits(self, file_path: Path, extension: int = 0):
        """
        Loads a FITS file and initializes WCS and observation metadata.
        Dispatches to MEF-aware logic so the same code path works for
        single-extension and multi-extension FITS files.
        """
        try:
            # Suppress non-critical FITS verification warnings from
            # ESO non-standard header cards
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=AstropyWarning)
                warnings.simplefilter("ignore", category=fits.verify.VerifyWarning)
                with fits.open(file_path) as hdul:
                    # Detect MEF if multiple image-like extensions with WCS exist
                    image_hdus = [
                        i
                        for i, hdu in enumerate(hdul)
                        if hdu.data is not None and len(hdu.data.shape) >= 2
                    ]
                    if len(image_hdus) > 1 and config.process_all_mef_extensions:
                        logger.info(
                            "MEF detected: %d image extensions. Processing extension %d.",
                            len(image_hdus),
                            extension,
                        )
                    if extension < len(hdul) and hdul[extension].data is not None:
                        data = hdul[extension].data
                        self.header = hdul[extension].header.copy()
                    else:
                        data = None
                        for hdu in hdul:
                            if hdu.data is not None and len(hdu.data.shape) >= 2:
                                data = hdu.data
                                self.header = hdu.header.copy()
                                break
                        if data is None:
                            raise ValueError("No 2D image data found in FITS file.")

                    # Squeeze singleton dimensions (e.g., (1, N, M) -> (N, M))
                    if len(data.shape) > 2:
                        data = np.squeeze(data)

                    self.image_data = data.astype(np.float32)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"FITS file not found: {file_path}") from exc

        # Parse header metadata for instrument, filter, exposure etc.
        self._parse_instrument_metadata()

        # Bias/overscan removal for raw ESO frames
        self._apply_basic_calibration()

        try:
            # Suppress non-critical astropy WCS/FITS warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=AstropyWarning)
                warnings.simplefilter("ignore", category=fits.verify.VerifyWarning)
                self.wcs = WCS(self.header)
                if not self.wcs.has_celestial:
                    self._create_basic_wcs()
        except (ValueError, TypeError, AttributeError):
            self._create_basic_wcs()

        self.obstime = self._get_observation_time()
        self.ra_center = self._get_header_value(
            ["CRVAL1", "RA", "HIERARCH ESO OBS RA", "OBJRA"], default=0.0
        )
        self.dec_center = self._get_header_value(
            ["CRVAL2", "DEC", "HIERARCH ESO OBS DEC", "OBJDEC"], default=0.0
        )

        if isinstance(self.ra_center, str):
            self.ra_center = self._parse_coordinate(self.ra_center, "ra")
        if isinstance(self.dec_center, str):
            self.dec_center = self._parse_coordinate(self.dec_center, "dec")

        # Compute pixel scale from WCS
        try:
            self.pixel_scale_arcsec = (
                np.abs(np.diag(self.wcs.pixel_scale_matrix)).mean() * 3600.0
            )
        except (ValueError, TypeError, AttributeError):
            self.pixel_scale_arcsec = None

        logger.info(
            'Loaded FITS: %s | instr=%s | shape=%s | pixscale=%.4f"/px | '
            'filter=%s | exptime=%.1fs | airmass=%s | seeing=%s"',
            Path(file_path).name,
            self.instrument,
            self.image_data.shape,
            self.pixel_scale_arcsec if self.pixel_scale_arcsec else -1,
            self.filter_name,
            self.exptime if self.exptime else -1,
            f"{self.airmass:.3f}" if self.airmass else "?",
            f'{self.seeing_arcsec:.2f}"' if self.seeing_arcsec else "?",
        )

    @staticmethod
    def list_mef_extensions(file_path: Path):
        """
        Returns a list of (extension_index, name, shape) tuples for all
        image-like extensions in a MEF file. Use this to discover
        amplifier/chip structure before calling load_fits(ext=N).
        """
        extensions = []
        with fits.open(file_path) as hdul:
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    extensions.append((i, hdu.name, tuple(hdu.data.shape)))
        return extensions

    def _apply_basic_calibration(self):
        """
        Applies bias/overscan correction for raw ESO frames.

        Detects overscan columns, subtracts per-row bias level, and removes
        residual 2D bias structure. No-op for pre-processed data.
        """
        if self.image_data is None:
            return

        inst_upper = str(self.instrument).upper()
        is_raw_eso = any(x in inst_upper for x in ("FORS2", "EFOSC", "VIRCAM", "HAWKI"))

        # Median < 1000 ADU indicates pre-reduced data (DES coadds);
        # raw ESO frames have bias + sky ~ 2000-5000 ADU.
        median_val = float(np.nanmedian(self.image_data))
        if not is_raw_eso and median_val < 1000:
            logger.debug(
                "Skipping basic calibration (already reduced, median=%.1f ADU)",
                median_val,
            )
            return

        if not is_raw_eso:
            return

        logger.info(
            "Applying basic bias/overscan calibration for raw %s frame "
            "(median=%.1f ADU)",
            self.instrument,
            median_val,
        )

        ny, nx = self.image_data.shape
        # Detect overscan columns: low variance and median near bias level
        col_medians = np.median(self.image_data, axis=0)
        col_stds = np.std(self.image_data, axis=0)
        bias_estimate = np.min(col_medians)
        # Overscan: std < 20% of median std, median within 10% of bias
        median_std = np.median(col_stds)
        if median_std > 0:
            overscan_mask = (col_stds < 0.3 * median_std) & (
                np.abs(col_medians - bias_estimate) < 0.2 * abs(bias_estimate) + 50
            )
        else:
            overscan_mask = np.zeros(nx, dtype=bool)

        if overscan_mask.sum() > 10:
            # Estimate per-row bias from overscan columns
            overscan_cols = np.where(overscan_mask)[0]
            self.has_overscan = True
            self.overscan_cols = (overscan_cols.min(), overscan_cols.max())
            row_bias = np.median(self.image_data[:, overscan_cols], axis=1)
            # Subtract row-by-row bias
            self.image_data = self.image_data - row_bias[:, np.newaxis]
            logger.info(
                "Overscan correction: %d bias cols [%d:%d], "
                "mean bias=%.1f ADU subtracted",
                overscan_mask.sum(),
                self.overscan_cols[0],
                self.overscan_cols[1],
                float(np.mean(row_bias)),
            )
        else:
            # Subtract global bias estimate when no overscan region detected
            self.image_data = self.image_data - bias_estimate
            logger.info(
                "No overscan detected; subtracted global bias=%.1f ADU",
                bias_estimate,
            )

    # ------------------------------------------------------------------
    # Header-agnostic metadata parsing
    # ------------------------------------------------------------------

    def _get_header_value(self, keys, default=None):
        """
        Returns the first available value from a list of candidate header
        keys. Supports both flat keywords and ESO ``HIERARCH ESO ...`` keys.
        """
        for key in keys:
            if key in self.header:
                return self.header[key]
        return default

    def _parse_instrument_metadata(self):
        """
        Parses instrument, telescope, filter, gain, saturation, airmass,
        seeing, and exposure time from FITS headers using instrument-agnostic
        key lookups covering DES/NOIRLab, ESO, and generic conventions.
        """
        h = self.header

        # Instrument & telescope
        self.instrument = self._get_header_value(
            ["INSTRUME", "HIERARCH ESO INS ID", "INSTRUMENT"], default="UNKNOWN"
        )
        self.telescope = self._get_header_value(
            ["TELESCOP", "HIERARCH ESO TEL ID", "TELESCOPE"], default="UNKNOWN"
        )

        # Filter  --  ESO uses HIERARCH ESO INS FILT1 NAME, DES uses FILTER
        self.filter_name = self._get_header_value(
            ["HIERARCH ESO INS FILT1 NAME", "FILTER", "FILTER1", "FILT1"],
            default="UNKNOWN",
        )

        # Exposure time
        self.exptime = self._get_header_value(
            ["EXPTIME", "EXPOSURE", "HIERARCH ESO DET EXP1 TIME"],
            default=None,
        )

        # Gain (e-/ADU)  --  ESO stores it per-CCD
        gain_val = self._get_header_value(
            ["GAIN", "HIERARCH ESO DET OUT1 GAIN", "HIERARCH ESO DET CHIP1 GAIN"],
            default=None,
        )
        if gain_val is not None:
            try:
                self.gain = float(gain_val)
            except (ValueError, TypeError):
                pass

        # Saturation level  --  instrument-specific
        sat_val = self._get_header_value(
            ["SATURATE", "SATURATION", "HIERARCH ESO DET OUT1 SATUR"],
            default=None,
        )
        if sat_val is not None:
            try:
                self.saturation_level = float(sat_val)
            except (ValueError, TypeError):
                pass
        else:
            # Instrument-specific defaults when not in header
            inst_upper = str(self.instrument).upper()
            if "FORS2" in inst_upper:
                # FORS2 16-bit ADC saturates near ~65000 ADU; ~55000 with bias
                self.saturation_level = 55000.0
            elif "EFOSC" in inst_upper:
                self.saturation_level = 50000.0
            elif "OMEGACAM" in inst_upper or "OMEGA" in inst_upper:
                self.saturation_level = 60000.0
            logger.debug(
                "SATURATE not in header; using instrument default %.0f for %s",
                self.saturation_level,
                self.instrument,
            )

        # Airmass  --  ESO: HIERARCH ESO TEL AIRM START/END
        airm_val = self._get_header_value(
            ["AIRMASS", "HIERARCH ESO TEL AIRM START", "HIERARCH ESO TEL AIRM END"],
            default=None,
        )
        if airm_val is not None:
            try:
                self.airmass = float(airm_val)
            except (ValueError, TypeError):
                pass

        # Seeing  --  ESO: HIERARCH ESO TEL AMBI FWHM START (arcsec)
        seeing_val = self._get_header_value(
            [
                "SEEING",
                "HIERARCH ESO TEL AMBI FWHM START",
                "HIERARCH ESO TEL IA FWHM",
                "DIMMSEE",
            ],
            default=None,
        )
        if seeing_val is not None:
            try:
                self.seeing_arcsec = float(seeing_val)
            except (ValueError, TypeError):
                pass

    def _get_observation_time(self):
        """
        Parses observation time from a priority list of header keywords
        spanning DES, ESO, and generic FITS conventions.
        """
        # (keyword, format) pairs in priority order
        time_keywords = [
            ("MJD-OBS", "mjd"),
            ("MJD", "mjd"),
            ("HIERARCH ESO DET EXP1 MJD", "mjd"),
            ("DATE-OBS", "isot"),
            ("DATE_OBS", "isot"),
            ("HIERARCH ESO TPL START", "isot"),
            ("HIERARCH ESO OBS START", "isot"),
            ("UTSTART", "isot"),
            ("UT-START", "isot"),
            ("JD", "jd"),
        ]
        for keyword, fmt in time_keywords:
            if keyword in self.header:
                time_str = self.header[keyword]
                try:
                    if fmt == "jd":
                        return Time(float(time_str), format="jd")
                    if fmt == "mjd":
                        return Time(float(time_str), format="mjd")
                    return Time(time_str, format="isot")
                except (ValueError, TypeError):
                    try:
                        return Time(time_str)
                    except (ValueError, TypeError):
                        continue
        logger.warning("No observation time keyword found; using 2023-01-01")
        return Time("2023-01-01T00:00:00")

    def _parse_coordinate(self, coord_str, coord_type):
        try:
            return float(coord_str)
        except (ValueError, TypeError):
            try:
                unit = u.hourangle if coord_type == "ra" else u.deg
                coord = SkyCoord(coord_str, unit=(unit, u.deg))
                return coord.ra.deg if coord_type == "ra" else coord.dec.deg
            except (ValueError, TypeError, AttributeError):
                return 0.0

    def _create_basic_wcs(self):
        """
        Builds a basic TAN WCS from pixel scale and CRVAL keywords when the
        native WCS is missing or invalid.
        """
        try:
            pixscale = self._get_header_value(
                ["PIXSCALE", "PIXSCAL1", "CDELT1", "HIERARCH ESO DET CHIP1 PIXSCALE"],
                default=0.27 / 3600.0,
            )
            if abs(pixscale) > 1:
                pixscale = pixscale / 3600.0
            w = WCS(naxis=2)
            w.wcs.crpix = [
                self.image_data.shape[1] / 2,
                self.image_data.shape[0] / 2,
            ]
            w.wcs.crval = [self.ra_center, self.dec_center]
            w.wcs.cd = np.array([[-pixscale, 0], [0, pixscale]])
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            self.wcs = w
        except (ValueError, TypeError, AttributeError):
            self.wcs = None

    def download_gaia_catalog(self, output_file: Path):
        """
        Queries Gaia archive for sources within the image field of view.
        """
        from astroquery.gaia import Gaia  # pylint: disable=import-outside-toplevel

        ny, nx = self.image_data.shape
        try:
            field_ra, field_dec = self.wcs.all_pix2world(nx / 2, ny / 2, 0)
        except (ValueError, TypeError, AttributeError):
            field_ra, field_dec = self.ra_center, self.dec_center

        corners_pix = [(0, 0), (nx, 0), (0, ny), (nx, ny)]
        corner_ra, corner_dec = self.wcs.all_pix2world(
            [c[0] for c in corners_pix], [c[1] for c in corners_pix], 0
        )
        corner_coords = SkyCoord(ra=corner_ra * u.deg, dec=corner_dec * u.deg)
        center_coord = SkyCoord(ra=field_ra * u.deg, dec=field_dec * u.deg)
        corner_seps = center_coord.separation(corner_coords).deg
        auto_radius = float(np.max(corner_seps)) * 1.1
        search_radius = max(auto_radius, config.gaia_search_radius_deg)

        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        Gaia.ROW_LIMIT = -1
        coord = SkyCoord(ra=field_ra, dec=field_dec, unit="deg", frame="icrs")
        job = Gaia.cone_search_async(coord, radius=u.Quantity(search_radius, u.deg))
        result = job.get_results()

        wanted = [
            "ra",
            "dec",
            "pmra",
            "pmdec",
            "pmra_error",
            "pmdec_error",
            "ra_error",
            "dec_error",
            "ref_epoch",
            "parallax",
            "phot_g_mean_mag",
            "bp_rp",
            "ruwe",
            "astrometric_matched_transits",
            "astrometric_excess_noise",
        ]
        available = [c for c in wanted if c in result.colnames]
        df = result[available].to_pandas()
        df.to_csv(output_file, index=False)
        return True

    def load_gaia_catalog(self, file_path: Path):
        """
        Loads the Gaia catalog from disk, downloading it if necessary.
        """
        # Re-download if file missing OR if critical columns are absent
        _CRITICAL_COLS = ["ruwe", "pmra_error", "ra_error"]
        needs_download = False
        if not file_path.exists():
            needs_download = True
        elif config.auto_download_gaia:
            try:
                _probe = pd.read_csv(file_path, nrows=1)
                for _col in _CRITICAL_COLS:
                    if _col not in _probe.columns:
                        needs_download = True
                        break
            except (ValueError, pd.errors.EmptyDataError, OSError):
                needs_download = True
        if needs_download and config.auto_download_gaia:
            print("  [INFO] Downloading fresh Gaia catalog...")
            self.download_gaia_catalog(file_path)

        try:
            gaia_df = pd.read_csv(file_path)

            # Cache validation: if the catalog is from a completely different part of the sky, redownload
            if config.auto_download_gaia and len(gaia_df) > 0:
                mean_ra = gaia_df["ra"].mean()
                mean_dec = gaia_df["dec"].mean()
                # Handle RA wraparound at 360/0 by computing shortest distance
                ra_diff = min(
                    abs(mean_ra - self.ra_center), 360 - abs(mean_ra - self.ra_center)
                )
                if ra_diff > 2.0 or abs(mean_dec - self.dec_center) > 2.0:
                    print(
                        "  [INFO] Cached Gaia catalog is for a different field. Re-downloading..."
                    )
                    self.download_gaia_catalog(file_path)
                    gaia_df = pd.read_csv(file_path)

            return {
                "ra": gaia_df["ra"].values,
                "dec": gaia_df["dec"].values,
                "pmra": (
                    gaia_df["pmra"].values
                    if "pmra" in gaia_df.columns
                    else np.zeros(len(gaia_df))
                ),
                "pmdec": (
                    gaia_df["pmdec"].values
                    if "pmdec" in gaia_df.columns
                    else np.zeros(len(gaia_df))
                ),
                "ref_epoch": (
                    gaia_df["ref_epoch"].values
                    if "ref_epoch" in gaia_df.columns
                    else np.full(len(gaia_df), 2016.0)
                ),
                "parallax": (
                    gaia_df["parallax"].values
                    if "parallax" in gaia_df.columns
                    else np.zeros(len(gaia_df))
                ),
                "phot_g_mean_mag": (
                    gaia_df["phot_g_mean_mag"].values
                    if "phot_g_mean_mag" in gaia_df.columns
                    else np.zeros(len(gaia_df))
                ),
                "bp_rp": (
                    gaia_df["bp_rp"].values
                    if "bp_rp" in gaia_df.columns
                    else np.full(len(gaia_df), np.nan)
                ),
                "ruwe": (
                    gaia_df["ruwe"].values
                    if "ruwe" in gaia_df.columns
                    else np.ones(len(gaia_df))
                ),
                "ra_error": (
                    gaia_df["ra_error"].values
                    if "ra_error" in gaia_df.columns
                    else np.full(len(gaia_df), np.nan)
                ),
                "dec_error": (
                    gaia_df["dec_error"].values
                    if "dec_error" in gaia_df.columns
                    else np.full(len(gaia_df), np.nan)
                ),
                "pmra_error": (
                    gaia_df["pmra_error"].values
                    if "pmra_error" in gaia_df.columns
                    else np.full(len(gaia_df), np.nan)
                ),
                "pmdec_error": (
                    gaia_df["pmdec_error"].values
                    if "pmdec_error" in gaia_df.columns
                    else np.full(len(gaia_df), np.nan)
                ),
                "astrometric_excess_noise": (
                    gaia_df["astrometric_excess_noise"].values
                    if "astrometric_excess_noise" in gaia_df.columns
                    else np.zeros(len(gaia_df))
                ),
            }
        except (ValueError, pd.errors.EmptyDataError, KeyError):
            return None

    def download_milliquas_catalog(self, output_file: Path):
        """
        Queries Milliquas catalog for quasars within the image field of view.
        """
        from astroquery.vizier import Vizier  # pylint: disable=import-outside-toplevel

        ny, nx = self.image_data.shape
        corners = self.wcs.all_pix2world([[0, 0], [nx, 0], [0, ny], [nx, ny]], 0)
        center_coord = SkyCoord(ra=self.ra_center * u.deg, dec=self.dec_center * u.deg)
        corner_coords = SkyCoord(ra=corners[:, 0] * u.deg, dec=corners[:, 1] * u.deg)
        search_radius = float(np.max(center_coord.separation(corner_coords).deg)) * 1.1

        coord = SkyCoord(ra=self.ra_center * u.deg, dec=self.dec_center * u.deg)
        tbl = None
        for catalog_id in ("VII/294", "VII/290"):
            try:
                v = Vizier(
                    columns=["RAJ2000", "DEJ2000", "Rmag", "Name", "Type"], row_limit=-1
                )
                results = v.query_region(
                    coord, radius=search_radius * u.deg, catalog=catalog_id
                )
                if results and len(results) > 0:
                    tbl = results[0]
                    break
            except (ValueError, TypeError, AttributeError):
                pass

        if tbl is None:
            return False

        ra_col = "RAJ2000" if "RAJ2000" in tbl.colnames else tbl.colnames[0]
        dec_col = "DEJ2000" if "DEJ2000" in tbl.colnames else tbl.colnames[1]
        df = tbl.to_pandas().rename(columns={ra_col: "ra", dec_col: "dec"})

        if "Type" in df.columns:
            df = df[df["Type"].str.contains("Q", na=False, case=False)].copy()

        df.to_csv(output_file, index=False)
        return True

    def load_milliquas_catalog(self, file_path: Path):
        """
        Loads the Milliquas catalog from disk, downloading it if necessary.
        """
        if not file_path.exists() and config.auto_download_milliquas:
            self.download_milliquas_catalog(file_path)

        try:
            df = pd.read_csv(file_path)

            # Cache validation: if the catalog is from a completely different part of the sky, redownload
            if config.auto_download_milliquas and len(df) > 0:
                mean_ra = df["ra"].mean()
                mean_dec = df["dec"].mean()
                ra_diff = min(
                    abs(mean_ra - self.ra_center), 360 - abs(mean_ra - self.ra_center)
                )
                if ra_diff > 2.0 or abs(mean_dec - self.dec_center) > 2.0:
                    print(
                        "  [INFO] Cached Milliquas catalog is for a different field. Re-downloading..."
                    )
                    self.download_milliquas_catalog(file_path)
                    df = pd.read_csv(file_path)

            return {
                "ra": df["ra"].values.astype(float),
                "dec": df["dec"].values.astype(float),
                "rmag": (
                    df["Rmag"].values.astype(float)
                    if "Rmag" in df.columns
                    else np.full(len(df), np.nan)
                ),
                "name": (
                    df["Name"].values.astype(str)
                    if "Name" in df.columns
                    else np.array([""] * len(df))
                ),
            }
        except (ValueError, pd.errors.EmptyDataError, KeyError):
            return None
