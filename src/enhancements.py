"""
Advanced astrometric calibration enhancements.

Provides classes for magnitude stratification with faint-source local
anchoring, strict train/test hold-out cross-validation, differential
chromatic refraction (DCR) correction, and CCD edge/tree-ring diagnostics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import ICRS, AltAz, EarthLocation, SkyCoord
from astropy.time import Time

try:
    from erfa import ErfaWarning
except ImportError:
    pass

from src.gpu import Backend
from src.logger import get_logger

logger = get_logger("enhancements")


# =============================================================================
# 1. Magnitude stratification & faint-star local anchoring
# =============================================================================

# Gaia DR3 per-coordinate positional uncertainty floor at epoch J2016.
# Piecewise-linear interpolation in log space; extrapolated past G=21.
_GAIA_FLOOR_MAS = {
    # G : sigma_pos_mas  (per coordinate, 5p+6p solutions)
    12: 0.02,
    14: 0.03,
    16: 0.05,
    17: 0.07,
    18: 0.15,
    19: 0.30,
    20: 0.55,
    21: 1.00,
    21.5: 1.40,
    22: 2.0,
}


def gaia_positional_floor_mas(
    g_mag: np.ndarray | float,
    per_star_error_mas: np.ndarray | float | None = None,
) -> np.ndarray | float:
    """Gaia DR3 per-coordinate positional uncertainty in mas.

    When ``per_star_error_mas`` is provided (a per-source quadrature sum
    of ra_error and dec_error), it is used directly.  Otherwise, the
    function linearly interpolates tabulated values (Lindegren et al.
    2021, Table B.1).  Beyond G=22 the floor grows linearly at 0.8 mas/mag.
    """
    if per_star_error_mas is not None:
        p = np.atleast_1d(per_star_error_mas).astype(float)
        floor = np.where(np.isfinite(p), p, np.nan)
        if np.isscalar(per_star_error_mas):
            return float(floor[0])
        return floor

    g = np.atleast_1d(g_mag).astype(float)
    keys = np.array(sorted(_GAIA_FLOOR_MAS.keys()))
    vals = np.array([_GAIA_FLOOR_MAS[k] for k in keys])
    floor = np.interp(g, keys, vals)
    # Extrapolation past G=22 with a steeper slope
    mask = g > 22.0
    if np.any(mask):
        floor[mask] = _GAIA_FLOOR_MAS[22] + (g[mask] - 22.0) * 0.8
    if np.isscalar(g_mag):
        return float(floor[0])
    return floor


@dataclass
class MagBin:
    """Container for a single magnitude-bin diagnostic entry."""

    lo: float
    hi: float
    n: int
    rms_ra_mas: float
    rms_dec_mas: float
    rms_tot_mas: float
    median_sep_mas: float
    p95_sep_mas: float
    gaia_floor_mas: float
    centroid_noise_mas: float
    expected_total_mas: float  # quadrature of floor + centroid noise


class MagnitudeStratifier:
    """
    Bins astrometric residuals by Gaia G magnitude and applies
    faint-source local anchoring to the bright-star reference grid.

    For each faint target (G > faint_threshold), the n_anchor nearest
    bright Gaia stars within radius_pix pixels are used to fit an affine
    transformation between measured and Gaia-reference pixel positions.
    The predicted reference position of the faint target is then compared
    to its measured position, yielding a local positional precision that
    cancels global atmospheric and optical systematics.

    Parameters
    ----------
    matched_data : pandas.DataFrame
        Advanced-calibrated matched catalog with x_pixel, y_pixel,
        ra_measured, dec_measured, ra_gaia, dec_gaia, and gaia_mag.
    wcs : astropy.wcs.WCS
        Image WCS for Gaia RA/Dec to pixel conversion.
    pixel_scale_arcsec : float
        Pixel scale in arcsec/px.
    fwhm_pixels : float
        Estimated PSF FWHM in pixels for centroid-noise modelling.
    faint_threshold : float, default 21.0
        Gaia G magnitude threshold for faint-source anchoring.
    """

    def __init__(
        self,
        matched_data: pd.DataFrame,
        wcs,
        pixel_scale_arcsec: float,
        fwhm_pixels: float,
        faint_threshold: float = 21.0,
        bin_edges: Sequence[float] | None = None,
    ):
        self.matched_data = matched_data
        self.wcs = wcs
        self.pixel_scale_arcsec = pixel_scale_arcsec
        self.fwhm_pixels = fwhm_pixels
        self.faint_threshold = faint_threshold
        if bin_edges is None:
            # Default binning: 1 mag wide from G=12 to G=22, plus an open
            # bin at the faint end.
            bin_edges = [12, 16, 17, 18, 19, 19.5, 20, 20.5, 21, 21.5, 22, 25]
        self.bin_edges = list(bin_edges)
        self._bins: list[MagBin] = []
        self.faint_anchor_results: pd.DataFrame | None = None
        self.photon_break_mag: float | None = None

    # -- public API --------------------------------------------------------

    def run(self) -> dict:
        """Executes magnitude stratification and faint-source anchoring."""
        if self.matched_data is None or len(self.matched_data) == 0:
            return {
                "magnitude_bins": [],
                "photon_break_mag": None,
                "faint_anchor_results": None,
                "summary": {},
            }

        # Compute residuals in mas
        cos_dec = np.cos(np.radians(self.matched_data["dec_gaia"]))
        dra_mas = (
            (self.matched_data["ra_final"] - self.matched_data["ra_gaia"])
            * 3600e3
            * cos_dec
        )
        ddec_mas = (
            self.matched_data["dec_final"] - self.matched_data["dec_gaia"]
        ) * 3600e3

        # Per-source SNR from DES r-band depth calibration
        mag = self.matched_data["gaia_mag"].values
        snr = 5.0 * 10.0 ** ((23.7 - mag) / 2.5)
        snr = np.clip(snr, 1.0, 5.0e4)
        fwhm_arcsec = self.fwhm_pixels * self.pixel_scale_arcsec
        centroid_mas = (fwhm_arcsec / (2.355 * snr)) * 1000.0
        # Use per-star Gaia positional errors when available
        if (
            "gaia_ra_error" in self.matched_data.columns
            and "gaia_dec_error" in self.matched_data.columns
        ):
            _re = np.where(
                np.isfinite(self.matched_data["gaia_ra_error"]),
                self.matched_data["gaia_ra_error"],
                np.nan,
            )
            _de = np.where(
                np.isfinite(self.matched_data["gaia_dec_error"]),
                self.matched_data["gaia_dec_error"],
                np.nan,
            )
            per_star_err = np.sqrt(_re**2 + _de**2)
        else:
            per_star_err = None
        gaia_floor = gaia_positional_floor_mas(mag, per_star_error_mas=per_star_err)

        # Bin the residuals
        self._bins = []
        for lo, hi in zip(self.bin_edges[:-1], self.bin_edges[1:]):
            mask = (mag >= lo) & (mag < hi)
            if mask.sum() < 1:
                continue
            self._bins.append(
                MagBin(
                    lo=lo,
                    hi=hi,
                    n=int(mask.sum()),
                    rms_ra_mas=float(np.sqrt(np.mean(dra_mas.values[mask] ** 2))),
                    rms_dec_mas=float(np.sqrt(np.mean(ddec_mas.values[mask] ** 2))),
                    rms_tot_mas=float(
                        np.sqrt(
                            np.mean(
                                dra_mas.values[mask] ** 2 + ddec_mas.values[mask] ** 2
                            )
                        )
                    ),
                    median_sep_mas=float(
                        np.median(np.hypot(dra_mas.values[mask], ddec_mas.values[mask]))
                    ),
                    p95_sep_mas=float(
                        np.percentile(
                            np.hypot(dra_mas.values[mask], ddec_mas.values[mask]), 95
                        )
                    ),
                    gaia_floor_mas=float(np.median(gaia_floor[mask])),
                    centroid_noise_mas=float(np.median(centroid_mas[mask])),
                    expected_total_mas=float(
                        np.sqrt(
                            np.median(gaia_floor[mask]) ** 2
                            + np.median(centroid_mas[mask]) ** 2
                        )
                    ),
                )
            )

        # Identify magnitude where RMS exceeds sqrt(2) * min_rms,
        # indicating photon-noise-limited precision.
        if len(self._bins) >= 2:
            min_rms = min(b.rms_tot_mas for b in self._bins if b.n >= 3)
            for b in self._bins:
                if b.n >= 3 and b.rms_tot_mas > np.sqrt(2) * min_rms:
                    self.photon_break_mag = 0.5 * (b.lo + b.hi)
                    break

        # Local anchoring for faint sources (G > faint_threshold)
        self.faint_anchor_results = self._anchor_faint_sources()

        return self._telemetry()

    # -- faint-source local anchoring -------------------------------------

    def _anchor_faint_sources(self) -> pd.DataFrame | None:
        """Anchors faint sources to the local bright-star reference grid.

        For each star above the faint threshold, the nearest bright stars
        within radius_pix are used to fit an affine transformation between
        measured and Gaia-reference pixel coordinates. The residual between
        the measured and predicted position provides the local positional
        error, with global systematics removed.
        """
        md = self.matched_data
        if md is None or len(md) == 0:
            return None

        # Separate bright anchors from faint targets
        bright_mask = md["gaia_mag"] < (self.faint_threshold - 1.0)
        faint_mask = md["gaia_mag"] >= self.faint_threshold
        n_bright = int(bright_mask.sum())
        n_faint = int(faint_mask.sum())
        logger.info(
            "Faint-star local anchoring: %d bright anchors (G<%.1f), "
            "%d faint targets (G>=%.1f)",
            n_bright,
            self.faint_threshold - 1.0,
            n_faint,
            self.faint_threshold,
        )
        if n_bright < 6 or n_faint < 1:
            logger.info(
                "Local anchoring skipped (insufficient bright anchors or "
                "no faint targets in Gaia catalog)."
            )
            return None

        # Compute Gaia reference pixel positions for all matched stars.
        gaia_ra = md["ra_gaia"].values
        gaia_dec = md["dec_gaia"].values
        try:
            gx, gy = self.wcs.all_world2pix(gaia_ra, gaia_dec, 0)
        except Exception as exc:
            logger.warning("WCS world2pix failed in faint anchoring: %s", exc)
            return None
        x_meas = md["x_pixel"].values
        y_meas = md["y_pixel"].values

        bright_idx = np.where(bright_mask)[0]
        faint_idx = np.where(faint_mask)[0]

        # Anchor parameters: 25 nearest bright stars within 200 px
        # (~50 arcsec on DECam, ~1 arcmin on FORS2, within atmospheric
        # coherence scale for absorbing large-scale systematics).
        n_anchor = min(25, n_bright)
        radius_pix = 200.0

        # Optionally compute all faint→bright pairwise distances on the
        # GPU in one shot; the dispatcher returns a dense NumPy matrix
        # (with distances beyond radius_pix set to inf) regardless of
        # device.  For small catalogs the CPU path is used automatically.
        faint_positions = np.column_stack([x_meas[faint_idx], y_meas[faint_idx]])
        bright_positions = np.column_stack([x_meas[bright_idx], y_meas[bright_idx]])
        try:
            backend = Backend.instance()
            dist_matrix = backend.pairwise_distances(
                faint_positions, bright_positions, max_radius=radius_pix
            )
            logger.debug(
                "Faint-anchor neighbour search on device: %s", backend.dm.device
            )
        except Exception:
            # Fallback: dense CPU pairwise distance via broadcasting
            diff = faint_positions[:, None, :] - bright_positions[None, :, :]
            dist_matrix = np.sqrt(np.sum(diff * diff, axis=-1))
            dist_matrix[dist_matrix > radius_pix] = np.inf

        results = []
        for f_row, fi in enumerate(faint_idx):
            # Use the precomputed distance matrix (row f_row of dist_matrix
            # holds distances from faint star fi to every bright anchor).
            r_row = dist_matrix[f_row]
            within = np.where(np.isfinite(r_row))[0]
            if len(within) < 6:
                # Fall back to all bright stars when local anchors are
                # insufficient; flag result as global rather than local.
                local_idx = bright_idx
                local_anchor_flag = 0
            else:
                # Take the n_anchor nearest within-radius stars
                sorted_within = within[np.argsort(r_row[within])][:n_anchor]
                local_idx = bright_idx[sorted_within]
                local_anchor_flag = 1
            if len(local_idx) < 6:
                continue

            # Fit affine: (gx, gy) = A @ (x_meas, y_meas) + t
            # Solve for A (2x2) and t (2,) using least squares.
            src = np.column_stack(
                [x_meas[local_idx], y_meas[local_idx], np.ones(len(local_idx))]
            )
            tgt_x = gx[local_idx]
            tgt_y = gy[local_idx]
            # Weighted by SNR (bright stars weighted more, faint anchors less)
            w = (
                1.0
                / np.clip(
                    gaia_positional_floor_mas(md["gaia_mag"].values[local_idx]),
                    0.02,
                    5.0,
                )
                ** 2
            )
            # Solve weighted least squares for X target
            sw = np.sqrt(w)
            try:
                coef_x, *_ = np.linalg.lstsq(src * sw[:, None], tgt_x * sw, rcond=None)
                coef_y, *_ = np.linalg.lstsq(src * sw[:, None], tgt_y * sw, rcond=None)
            except np.linalg.LinAlgError:
                continue

            # Predict the faint star's reference pixel position
            src_faint = np.array([x_meas[fi], y_meas[fi], 1.0])
            gx_pred = float(src_faint @ coef_x)
            gy_pred = float(src_faint @ coef_y)

            # Residual in pixel space  ->  mas
            dx_pix = x_meas[fi] - gx_pred
            dy_pix = y_meas[fi] - gy_pred
            dra_mas = dx_pix * self.pixel_scale_arcsec * 1000.0
            ddec_mas = dy_pix * self.pixel_scale_arcsec * 1000.0
            sep_mas = float(np.hypot(dra_mas, ddec_mas))

            results.append(
                {
                    "gaia_mag": float(md["gaia_mag"].values[fi]),
                    "x_pixel": float(x_meas[fi]),
                    "y_pixel": float(y_meas[fi]),
                    "dra_local_mas": float(dra_mas),
                    "ddec_local_mas": float(ddec_mas),
                    "sep_local_mas": sep_mas,
                    "n_anchor": len(local_idx),
                    "local_anchors_used": int(local_anchor_flag),
                }
            )

        if not results:
            logger.info("Local anchoring: no faint stars with sufficient anchors.")
            return None
        df = pd.DataFrame(results)
        rms_local = float(
            np.sqrt(np.mean(df["dra_local_mas"] ** 2 + df["ddec_local_mas"] ** 2))
        )
        logger.info(
            "Local anchoring complete: N=%d faint stars, "
            "local residual RMS = %.2f mas (vs. global RMS = %.2f mas)",
            len(df),
            rms_local,
            float(
                np.sqrt(
                    np.mean(
                        (
                            md.loc[faint_mask].apply(
                                lambda r: (
                                    (
                                        (r["ra_final"] - r["ra_gaia"])
                                        * 3600e3
                                        * np.cos(np.radians(r["dec_gaia"]))
                                    )
                                    ** 2
                                    + ((r["dec_final"] - r["dec_gaia"]) * 3600e3) ** 2
                                ),
                                axis=1,
                            )
                        ).mean()
                    )
                )
            ),
        )
        return df

    # -- telemetry --------------------------------------------------------

    def _telemetry(self) -> dict:
        bins_serial = [
            {
                "lo": b.lo,
                "hi": b.hi,
                "n": b.n,
                "rms_ra_mas": b.rms_ra_mas,
                "rms_dec_mas": b.rms_dec_mas,
                "rms_tot_mas": b.rms_tot_mas,
                "median_sep_mas": b.median_sep_mas,
                "p95_sep_mas": b.p95_sep_mas,
                "gaia_floor_mas": b.gaia_floor_mas,
                "centroid_noise_mas": b.centroid_noise_mas,
                "expected_total_mas": b.expected_total_mas,
            }
            for b in self._bins
        ]
        faint_summary = None
        if self.faint_anchor_results is not None and len(self.faint_anchor_results) > 0:
            far = self.faint_anchor_results
            faint_summary = {
                "n_faint": len(far),
                "local_rms_mas": float(
                    np.sqrt(
                        np.mean(far["dra_local_mas"] ** 2 + far["ddec_local_mas"] ** 2)
                    )
                ),
                "local_median_sep_mas": float(far["sep_local_mas"].median()),
                "local_p95_sep_mas": float(far["sep_local_mas"].quantile(0.95)),
                "mean_n_anchor": float(far["n_anchor"].mean()),
            }
        return {
            "magnitude_bins": bins_serial,
            "photon_break_mag": self.photon_break_mag,
            "faint_anchor_results": self.faint_anchor_results,
            "faint_summary": faint_summary,
        }


# =============================================================================
# 1b. Faint-star precision analysis (pipeline vs Gaia baseline)
# =============================================================================


class PrecisionAnalyzer:
    """
    Extracts faint-star (G >= faint_threshold) astrometric precision
    diagnostics, comparing pipeline-calibrated positions to the Gaia
    reference frame.

    For each magnitude bin (default 0.5 mag wide) the following are
    tabulated:

        * pipeline_rms_mas   — observed RMS of (pipeline - Gaia) residuals
        * gaia_floor_mas     — median Gaia positional floor in the bin
        * centroid_noise_mas — median centroid noise in the bin
        * expected_total_mas — quadrature sum of floor + centroid noise
        * improvement_factor — expected_total / pipeline_rms
                               (>1 means pipeline outperforms the combined
                                Gaia+centroid expectation)

    A dedicated catalog of faint-star measurements is saved for external
    analysis, and per-range comparisons (e.g. 19 < G < 22) are produced
    alongside the primary G >= faint_threshold results.
    """

    def __init__(
        self,
        matched_data: pd.DataFrame,
        pixel_scale_arcsec: float,
        fwhm_pixels: float,
        bin_width: float = 0.5,
        faint_threshold: float = 21.0,
        extra_ranges: Sequence[tuple[float, float]] | None = None,
    ):
        self.matched_data = matched_data
        self.pixel_scale_arcsec = pixel_scale_arcsec
        self.fwhm_pixels = fwhm_pixels
        self.bin_width = bin_width
        self.faint_threshold = faint_threshold
        self.extra_ranges = list(extra_ranges) if extra_ranges else []

    # -- public API --------------------------------------------------------

    def run(self) -> dict:
        """Runs the full faint-star precision analysis pipeline."""
        md = self.matched_data
        if md is None or len(md) == 0:
            return {"faint_catalog": None, "g21_bins": [], "range_bins_list": []}

        # Compute per-star residuals in mas
        cos_dec = np.cos(np.radians(md["dec_gaia"]))
        dra_mas = (md["ra_final"] - md["ra_gaia"]) * 3600e3 * cos_dec
        ddec_mas = (md["dec_final"] - md["dec_gaia"]) * 3600e3
        sep_mas = np.hypot(dra_mas, ddec_mas)

        # Per-star centroid noise and Gaia floor
        mag = md["gaia_mag"].values
        snr = 5.0 * 10.0 ** ((23.7 - mag) / 2.5)
        snr = np.clip(snr, 1.0, 5.0e4)
        fwhm_arcsec = self.fwhm_pixels * self.pixel_scale_arcsec
        centroid_mas = (fwhm_arcsec / (2.355 * snr)) * 1000.0
        # Use per-star Gaia positional errors when available
        if "gaia_ra_error" in md.columns and "gaia_dec_error" in md.columns:
            _re = np.where(
                np.isfinite(md["gaia_ra_error"]), md["gaia_ra_error"], np.nan
            )
            _de = np.where(
                np.isfinite(md["gaia_dec_error"]), md["gaia_dec_error"], np.nan
            )
            per_star_err = np.sqrt(_re**2 + _de**2)
        else:
            per_star_err = None
        gaia_floor = gaia_positional_floor_mas(mag, per_star_error_mas=per_star_err)

        # Build faint-star catalog (G >= faint_threshold)
        faint_mask = mag >= self.faint_threshold
        n_faint = int(faint_mask.sum())

        faint_catalog = None
        if n_faint > 0:
            expected_mas = np.sqrt(
                gaia_floor[faint_mask] ** 2 + centroid_mas[faint_mask] ** 2
            )
            pipeline_rms = sep_mas[faint_mask]
            # improvement > 1 means pipeline beats the combined expectation
            improv = np.where(pipeline_rms > 0, expected_mas / pipeline_rms, 0.0)

            faint_catalog = pd.DataFrame(
                {
                    "gaia_mag": mag[faint_mask],
                    "x_pixel": md["x_pixel"].values[faint_mask],
                    "y_pixel": md["y_pixel"].values[faint_mask],
                    "ra_pipeline": md["ra_final"].values[faint_mask],
                    "dec_pipeline": md["dec_final"].values[faint_mask],
                    "ra_gaia": md["ra_gaia"].values[faint_mask],
                    "dec_gaia": md["dec_gaia"].values[faint_mask],
                    "dra_mas": dra_mas.values[faint_mask],
                    "ddec_mas": ddec_mas.values[faint_mask],
                    "sep_mas": sep_mas.values[faint_mask],
                    "gaia_floor_mas": gaia_floor[faint_mask],
                    "centroid_noise_mas": centroid_mas[faint_mask],
                    "expected_total_mas": expected_mas,
                    "pipeline_rms_mas": pipeline_rms,
                    "improvement_factor": improv,
                }
            )
            logger.info(
                "PrecisionAnalyzer: %d faint stars (G>=%.1f) catalog built.",
                n_faint,
                self.faint_threshold,
            )

        # Per-bin statistics for G >= faint_threshold
        g21_bins = self._compute_per_bin_stats(
            mag,
            dra_mas,
            ddec_mas,
            gaia_floor,
            centroid_mas,
            mag_lo=self.faint_threshold,
            mag_hi=max(mag.max(), self.faint_threshold + 1),
        )
        logger.info(
            "PrecisionAnalyzer: %d magnitude bins for G>=%.1f.",
            len(g21_bins),
            self.faint_threshold,
        )

        # Per-range statistics for each extra range
        range_bins_list = []
        for rlo, rhi in self.extra_ranges:
            bins = self._compute_per_bin_stats(
                mag,
                dra_mas,
                ddec_mas,
                gaia_floor,
                centroid_mas,
                mag_lo=rlo,
                mag_hi=rhi,
            )
            range_bins_list.append({"range": (rlo, rhi), "bins": bins})
            logger.info(
                "PrecisionAnalyzer: %d bins for range [%.1f, %.1f].",
                len(bins),
                rlo,
                rhi,
            )

        return {
            "faint_catalog": faint_catalog,
            "g21_bins": g21_bins,
            "range_bins_list": range_bins_list,
        }

    # -- internal ----------------------------------------------------------

    def _compute_per_bin_stats(
        self,
        mag: np.ndarray,
        dra_mas: np.ndarray,
        ddec_mas: np.ndarray,
        gaia_floor: np.ndarray,
        centroid_mas: np.ndarray,
        mag_lo: float,
        mag_hi: float,
    ) -> list:
        """Computes per-bin precision statistics over [mag_lo, mag_hi)."""
        bins = []
        # Round edges to nearest bin_width
        lo = np.floor(mag_lo / self.bin_width) * self.bin_width
        hi = np.ceil(mag_hi / self.bin_width) * self.bin_width
        edges = np.arange(lo, hi + self.bin_width / 2, self.bin_width)

        for i in range(len(edges) - 1):
            e_lo, e_hi = edges[i], edges[i + 1]
            mask = (mag >= e_lo) & (mag < e_hi)
            n = int(mask.sum())
            if n < 1:
                continue

            pipe_rms = float(np.sqrt(np.mean(dra_mas[mask] ** 2 + ddec_mas[mask] ** 2)))
            gfloor = float(np.median(gaia_floor[mask]))
            cnoise = float(np.median(centroid_mas[mask]))
            expected = float(np.sqrt(gfloor**2 + cnoise**2))
            improv = expected / pipe_rms if pipe_rms > 0 else 0.0

            bins.append(
                {
                    "lo": float(e_lo),
                    "hi": float(e_hi),
                    "n": n,
                    "pipeline_rms_mas": pipe_rms,
                    "gaia_floor_mas": gfloor,
                    "centroid_noise_mas": cnoise,
                    "expected_total_mas": expected,
                    "improvement_factor": improv,
                }
            )
        return bins


# --- 2. Hold-out cross-validation ---


class HoldoutCV:
    """
    Strict train/test cross-validation for astrometric transformation models.

    Fits the spatial model on a random subset of matched stars and evaluates
    residual RMS on held-out stars, eliminating in-sample bias. Multiple
    random splits are averaged to estimate predictive precision.

    Parameters
    ----------
    matched_data : pandas.DataFrame
        Catalog with x_pixel, y_pixel, ra_measured, dec_measured,
        ra_gaia, dec_gaia, gaia_mag, and mag_weight columns.
    wcs : astropy.wcs.WCS
        Image WCS.
    train_fraction : float, default 0.7
        Fraction of stars used for model fitting.
    n_splits : int, default 5
        Number of random train/test splits to average.
    random_seed : int, default 42
    """

    def __init__(
        self,
        matched_data: pd.DataFrame,
        wcs,
        train_fraction: float = 0.7,
        n_splits: int = 5,
        random_seed: int = 42,
    ):
        self.matched_data = matched_data
        self.wcs = wcs
        self.train_fraction = train_fraction
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.results_: list[dict] = []
        self.summary_: dict = {}

    def run(self) -> dict:
        if self.matched_data is None or len(self.matched_data) < 20:
            logger.info(
                "HoldoutCV skipped (insufficient stars: %s)",
                0 if self.matched_data is None else len(self.matched_data),
            )
            return {"n_splits": 0, "test_rms_mas": None, "train_rms_mas": None}

        rng = np.random.default_rng(self.random_seed)
        n = len(self.matched_data)
        n_train = max(10, int(round(self.train_fraction * n)))

        # Pre-extract arrays
        x = self.matched_data["x_pixel"].values.astype(float)
        y = self.matched_data["y_pixel"].values.astype(float)
        cos_dec = np.cos(np.radians(self.matched_data["dec_gaia"].values))
        # Residuals in arcsec: measured - gaia
        dra = (
            (self.matched_data["ra_measured"] - self.matched_data["ra_gaia"]).values
            * 3600.0
            * cos_dec
        )
        ddec = (
            self.matched_data["dec_measured"] - self.matched_data["dec_gaia"]
        ).values * 3600.0
        if "mag_weight" in self.matched_data.columns:
            w = self.matched_data["mag_weight"].values.astype(float)
        else:
            w = np.ones(n)

        for split_i in range(self.n_splits):
            idx_all = np.arange(n)
            rng.shuffle(idx_all)
            train_idx = idx_all[:n_train]
            test_idx = idx_all[n_train:]

            # Build a 4th-order 2D polynomial design matrix on normalized coords
            x_mean, x_std = x[train_idx].mean(), x[train_idx].std()
            y_mean, y_std = y[train_idx].mean(), y[train_idx].std()
            x_std = x_std if x_std > 1e-9 else 1.0
            y_std = y_std if y_std > 1e-9 else 1.0
            xn_tr = (x[train_idx] - x_mean) / x_std
            yn_tr = (y[train_idx] - y_mean) / y_std
            xn_te = (x[test_idx] - x_mean) / x_std
            yn_te = (y[test_idx] - y_mean) / y_std

            A_tr = self._poly_design(xn_tr, yn_tr, degree=2)
            A_te = self._poly_design(xn_te, yn_te, degree=2)

            # Weighted ridge regression via normal equations
            alpha = 1.0e-3 * len(train_idx)
            sw_tr = np.sqrt(w[train_idx])
            Xw_tr = A_tr * sw_tr[:, None]
            y_ra_tr = dra[train_idx] * sw_tr
            y_dec_tr = ddec[train_idx] * sw_tr
            # Gram matrix + ridge
            G = Xw_tr.T @ Xw_tr + alpha * np.eye(A_tr.shape[1])
            b_ra = Xw_tr.T @ y_ra_tr
            b_dec = Xw_tr.T @ y_dec_tr
            try:
                coef_ra = np.linalg.solve(G, b_ra)
                coef_dec = np.linalg.solve(G, b_dec)
            except np.linalg.LinAlgError:
                coef_ra, *_ = np.linalg.lstsq(G, b_ra, rcond=None)
                coef_dec, *_ = np.linalg.lstsq(G, b_dec, rcond=None)

            # Predictions
            ra_pred_tr = A_tr @ coef_ra
            dec_pred_tr = A_tr @ coef_dec
            ra_pred_te = A_te @ coef_ra
            dec_pred_te = A_te @ coef_dec

            # Residuals: observed - predicted
            dra_res_tr = dra[train_idx] - ra_pred_tr
            ddec_res_tr = ddec[train_idx] - dec_pred_tr
            dra_res_te = dra[test_idx] - ra_pred_te
            ddec_res_te = ddec[test_idx] - dec_pred_te

            rms_train = float(np.sqrt(np.mean(dra_res_tr**2 + ddec_res_tr**2)))
            rms_test = float(np.sqrt(np.mean(dra_res_te**2 + ddec_res_te**2)))
            self.results_.append(
                {
                    "split": split_i,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "train_rms_mas": rms_train * 1000.0,
                    "test_rms_mas": rms_test * 1000.0,
                }
            )

        if not self.results_:
            return {"n_splits": 0, "test_rms_mas": None, "train_rms_mas": None}

        test_rms = np.array([r["test_rms_mas"] for r in self.results_])
        train_rms = np.array([r["train_rms_mas"] for r in self.results_])
        self.summary_ = {
            "n_splits": len(self.results_),
            "test_rms_mean_mas": float(test_rms.mean()),
            "test_rms_std_mas": float(test_rms.std(ddof=1)),
            "train_rms_mean_mas": float(train_rms.mean()),
            "train_rms_std_mas": float(train_rms.std(ddof=1)),
            "overfit_ratio": float(train_rms.mean() / test_rms.mean()),
            "per_split": self.results_,
        }
        logger.info(
            "HoldoutCV: %d splits | train RMS = %.2f ± %.2f mas | "
            "TEST RMS = %.2f ± %.2f mas | overfit_ratio = %.3f",
            self.summary_["n_splits"],
            self.summary_["train_rms_mean_mas"],
            self.summary_["train_rms_std_mas"],
            self.summary_["test_rms_mean_mas"],
            self.summary_["test_rms_std_mas"],
            self.summary_["overfit_ratio"],
        )
        return self.summary_

    @staticmethod
    def _poly_design(x, y, degree=4):
        """Builds a 2D polynomial design matrix up to the specified degree."""
        cols = []
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                cols.append(x**i * y**j)
        return np.column_stack(cols)


# --- 3. Differential Chromatic Refraction correction ---


class DCRCorrector:
    """
    Differential Chromatic Refraction (DCR) correction using Gaia BP-RP colour.

    Corrects astrometric residuals for chromatic effects using the model
    Delta R_DCR = (a_f + b_f * (BP-RP)) * tan(z), where z is the zenith
    distance and (a_f, b_f) are filter-dependent coefficients determined
    by linear regression of residuals projected along the zenith direction
    via the parallactic angle.

    Parameters
    ----------
    matched_data : pandas.DataFrame
        Catalog with ra_measured, dec_measured, ra_gaia, dec_gaia,
        gaia_mag, bp_rp, x_pixel, y_pixel columns.
    obs_time : astropy.time.Time
        Observation epoch.
    ra_center_deg, dec_center_deg : float
        Field centre coordinates in degrees.
    observatory_lat_deg : float, default -30.17
        Observatory latitude (CTIO default, -24.6273 for Paranal).
    pixel_scale_arcsec : float
    """

    def __init__(
        self,
        matched_data: pd.DataFrame,
        obs_time: Time,
        ra_center_deg: float,
        dec_center_deg: float,
        observatory_lat_deg: float,
        pixel_scale_arcsec: float,
    ):
        self.matched_data = matched_data
        self.obs_time = obs_time
        self.ra_center = ra_center_deg
        self.dec_center = dec_center_deg
        self.lat = observatory_lat_deg
        self.pixel_scale = pixel_scale_arcsec
        # Outputs
        self.parallactic_angles_deg: np.ndarray | None = None
        self.zenith_distances_deg: np.ndarray | None = None
        self.tan_zd: np.ndarray | None = None
        self.coef_a_mas: float | None = None  # DCR offset (filter mean)
        self.coef_b_mas: float | None = None  # DCR colour slope
        self.coef_a_dec_mas: float | None = None
        self.coef_b_dec_mas: float | None = None
        self.rms_before_mas: float | None = None
        self.rms_after_mas: float | None = None
        self.cv_rms_before_mas: float | None = None
        self.cv_rms_after_mas: float | None = None
        self.applied: bool = False

    def run(self) -> dict:
        if self.matched_data is None or len(self.matched_data) < 10:
            logger.info("DCRCorrector: too few stars, skipping.")
            return {"applied": False}

        if (
            "bp_rp" not in self.matched_data.columns
            or self.matched_data["bp_rp"].isna().all()
        ):
            logger.info("DCRCorrector: no bp_rp colour, skipping.")
            return {"applied": False}

        md = self.matched_data
        ra = md["ra_measured"].values
        dec = md["dec_measured"].values
        color = md["bp_rp"].values
        color = np.where(np.isnan(color), np.nanmedian(color), color)

        # Compute parallactic angle and zenith distance per-star
        try:
            q_deg, zd_deg = self._compute_parallactic(ra, dec)
        except Exception as exc:
            logger.warning("DCRCorrector: parallactic angle failed: %s", exc)
            return {"applied": False}

        self.parallactic_angles_deg = q_deg
        self.zenith_distances_deg = zd_deg
        # tan(zd) for the linearised DCR model.  Clip at zd=80 deg to avoid
        # divergence near the horizon (we never expect raw data there).
        tan_zd = np.tan(np.radians(np.clip(zd_deg, 0.0, 80.0)))
        self.tan_zd = tan_zd

        # Convert residuals to RA/Dec, then project to zenith direction
        cos_dec = np.cos(np.radians(md["dec_gaia"].values))
        dra_mas = (md["ra_measured"] - md["ra_gaia"]).values * 3600e3 * cos_dec
        ddec_mas = (md["dec_measured"] - md["dec_gaia"]).values * 3600e3

        # Project (dra, ddec) into the zenith direction (positive = towards zenith)
        q_rad = np.radians(q_deg)
        # Convention: parallactic angle q is measured from North through East.
        # The unit vector towards zenith in (RA-cos_dec, Dec) space is
        # (sin q, cos q).  The component of the residual projected onto the
        # zenith direction (along-zenith) is R_zen = dra * sin(q) + ddec *
        # cos(q); the orthogonal component (along the horizon) is
        # R_hor = dra * cos(q) - ddec * sin(q).
        r_zen = dra_mas * np.sin(q_rad) + ddec_mas * np.cos(q_rad)
        r_hor = dra_mas * np.cos(q_rad) - ddec_mas * np.sin(q_rad)

        # Design matrix:  R_zen = (a + b * color) * tan(zd)  +  noise
        # Equivalent to:   R_zen / tan(zd) = a + b * color   (when tan(zd) > 0)
        valid = tan_zd > 0.05  # avoid division by ~0 (near zenith, no DCR)
        if valid.sum() < 6:
            logger.info(
                "DCRCorrector: %d stars with tan(zd)>0.05 — too few, skipping.",
                int(valid.sum()),
            )
            return {"applied": False}

        A = np.column_stack([np.ones(valid.sum()), color[valid] * tan_zd[valid]])
        # Actually use the form: R_zen = a*tan_zd + b*color*tan_zd
        # i.e. A = [tan_zd, color*tan_zd]
        A = np.column_stack([tan_zd[valid], color[valid] * tan_zd[valid]])
        y = r_zen[valid]

        # Weighted least squares (SNR weights if available)
        if "mag_weight" in md.columns:
            w = md["mag_weight"].values[valid]
        else:
            w = np.ones(valid.sum())
        sw = np.sqrt(w)
        try:
            coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        except np.linalg.LinAlgError:
            logger.warning("DCRCorrector: lstsq failed.")
            return {"applied": False}
        a_zen, b_zen = float(coef[0]), float(coef[1])

        # Leave-one-out CV to evaluate DCR correction improvement
        n_v = valid.sum()
        cv_before = float(np.sqrt(np.mean(y**2)))
        cv_after_resid = np.zeros(n_v)
        A_v = A.copy()
        y_v = y.copy()
        for i in range(n_v):
            mask = np.ones(n_v, dtype=bool)
            mask[i] = False
            try:
                c, *_ = np.linalg.lstsq(
                    A_v[mask] * sw[mask][:, None], y_v[mask] * sw[mask], rcond=None
                )
                cv_after_resid[i] = y_v[i] - A_v[i] @ c
            except np.linalg.LinAlgError:
                cv_after_resid[i] = y_v[i]
        cv_after = float(np.sqrt(np.mean(cv_after_resid**2)))

        # Apply correction only if LOO-CV improves by >= 1%
        applied = cv_after < cv_before * 0.99
        if applied:
            # Compute full corrections for all stars; near-zenith stars
            # (tan_zd ~ 0) correctly get ~0 correction.
            A_full = np.column_stack([tan_zd, color * tan_zd])
            dcr_zen_pred = A_full @ coef
            # Project DCR correction from zenith frame back to equatorial
            # coordinates and subtract from measured positions
            dra_corr_mas = -dcr_zen_pred * np.sin(q_rad)
            ddec_corr_mas = -dcr_zen_pred * np.cos(q_rad)
            # Apply corrections to ra_measured / dec_measured (and downstream columns)
            self.matched_data["ra_measured"] = (
                self.matched_data["ra_measured"] + dra_corr_mas / 3600e3 / cos_dec
            )
            self.matched_data["dec_measured"] = (
                self.matched_data["dec_measured"] + ddec_corr_mas / 3600e3
            )
            # Recompute downstream columns if they exist
            for ra_col, dec_col in [
                ("ra_weighted", "dec_weighted"),
                ("ra_final", "dec_final"),
            ]:
                if ra_col in self.matched_data.columns:
                    self.matched_data[ra_col] = (
                        self.matched_data[ra_col] + dra_corr_mas / 3600e3 / cos_dec
                    )
                    self.matched_data[dec_col] = (
                        self.matched_data[dec_col] + ddec_corr_mas / 3600e3
                    )
            # Recompute residuals
            dra_mas_new = (
                (self.matched_data["ra_measured"] - self.matched_data["ra_gaia"]).values
                * 3600e3
                * cos_dec
            )
            ddec_mas_new = (
                self.matched_data["dec_measured"] - self.matched_data["dec_gaia"]
            ).values * 3600e3
            self.rms_before_mas = float(np.sqrt(np.mean(dra_mas**2 + ddec_mas**2)))
            self.rms_after_mas = float(
                np.sqrt(np.mean(dra_mas_new**2 + ddec_mas_new**2))
            )
            self.cv_rms_before_mas = cv_before
            self.cv_rms_after_mas = cv_after
            self.coef_a_mas = a_zen
            self.coef_b_mas = b_zen
            self.applied = True
            logger.info(
                "DCR correction APPLIED: a=%.2f mas, b=%.2f mas/mag (zenith dir) | "
                "RMS %.2f→%.2f mas (LOO-CV %.2f→%.2f mas) | zd=%.1f°, tan(zd)=%.3f",
                a_zen,
                b_zen,
                self.rms_before_mas,
                self.rms_after_mas,
                cv_before,
                cv_after,
                float(np.median(zd_deg)),
                float(np.median(tan_zd)),
            )
        else:
            self.rms_before_mas = float(np.sqrt(np.mean(dra_mas**2 + ddec_mas**2)))
            self.cv_rms_before_mas = cv_before
            self.cv_rms_after_mas = cv_after
            self.coef_a_mas = a_zen
            self.coef_b_mas = b_zen
            logger.info(
                "DCR correction skipped (LOO-CV did not improve: %.2f→%.2f mas). "
                "Best-fit coeffs: a=%.2f mas, b=%.2f mas/mag.",
                cv_before,
                cv_after,
                a_zen,
                b_zen,
            )

        return self._telemetry()

    def _compute_parallactic(self, ra_deg, dec_deg) -> tuple[np.ndarray, np.ndarray]:
        """Compute parallactic angle and zenith distance per-star.

        Uses astropy's AltAz transform from the observatory location and
        observation time.  Returns (q, zd) in degrees, both as arrays
        matching the input.
        """
        # Field centre location: CTIO for DECam, Paranal for FORS2.
        # The observer's longitude is implicit in the AltAz transform.
        site = EarthLocation(lat=self.lat * u.deg, lon=0.0 * u.deg, height=0.0 * u.m)
        # Set observatory longitude based on instrument:
        # CTIO (-70.81) for DECam, Paranal (-70.40) for FORS2.
        # The ~0.4 deg difference produces negligible parallactic angle
        # error (<0.4 deg) for DCR correction requirements.
        if self.lat < -25.0:  # Paranal
            lon = -70.40
        else:  # CTIO
            lon = -70.81
        site = EarthLocation(lat=self.lat * u.deg, lon=lon * u.deg, height=0.0 * u.m)
        stars = SkyCoord(
            ra=ra_deg * u.deg,
            dec=dec_deg * u.deg,
            frame=ICRS(),
            location=site,
            obstime=self.obs_time,
        )
        altaz = stars.transform_to(AltAz())
        zd = altaz.zen.deg
        # Parallactic angle: q = arctan2(sin(H), tan(dec)*cos(lat) - sin(lat)*cos(H))
        # where H is the hour angle.  We can compute H from LST - RA.
        # astropy doesn't expose q directly; compute from coordinates.
        lst = self.obs_time.sidereal_time("apparent", longitude=lon * u.deg)
        ha = (lst - ra_deg * u.deg).wrap_at(180 * u.deg)
        dec_rad = np.radians(dec_deg)
        lat_rad = np.radians(self.lat)
        ha_rad = np.radians(ha.deg)
        sin_q = np.sin(ha_rad) * np.cos(lat_rad) / np.sin(np.radians(zd))
        cos_q = (
            np.tan(lat_rad) * np.cos(dec_rad) - np.sin(dec_rad) * np.cos(ha_rad)
        ) / np.sin(np.radians(zd))
        # Clip to avoid arcsin domain errors near zenith (where q is undefined anyway)
        sin_q = np.clip(sin_q, -1.0, 1.0)
        cos_q = np.clip(cos_q, -1.0, 1.0)
        q = np.degrees(np.arctan2(sin_q, cos_q))
        # Near zenith (zd < 1 deg), q is undefined  --  set to 0
        q = np.where(zd < 1.0, 0.0, q)
        return q, zd

    def _telemetry(self) -> dict:
        return {
            "applied": self.applied,
            "coef_a_mas": self.coef_a_mas,
            "coef_b_mas": self.coef_b_mas,
            "rms_before_mas": self.rms_before_mas,
            "rms_after_mas": self.rms_after_mas,
            "cv_rms_before_mas": self.cv_rms_before_mas,
            "cv_rms_after_mas": self.cv_rms_after_mas,
            "median_zd_deg": (
                float(np.median(self.zenith_distances_deg))
                if self.zenith_distances_deg is not None
                else None
            ),
            "median_tan_zd": (
                float(np.median(self.tan_zd)) if self.tan_zd is not None else None
            ),
            "median_parallactic_deg": (
                float(np.median(np.abs(self.parallactic_angles_deg)))
                if self.parallactic_angles_deg is not None
                else None
            ),
        }


# --- 4. CCD edge effects and tree-ring diagnostics ---


class CCDEdgeDiagnostic:
    """
    Diagnostics and empirical correction for CCD edge and tree-ring
    sensor-grid biases.

    Produces radial residual profiles, edge-vs-centre RMS statistics,
    and applies an empirical concentric-ring correction using a radial
    sinusoidal model with characteristic periods (200, 300, 500 px).

    Parameters
    ----------
    matched_data : pandas.DataFrame
        Catalog with x_pixel, y_pixel, ra_measured, dec_measured,
        ra_gaia, dec_gaia columns.
    image_shape : tuple
        (ny, nx) image dimensions for CCD centre computation.
    pixel_scale_arcsec : float
    """

    def __init__(
        self,
        matched_data: pd.DataFrame,
        image_shape: tuple,
        pixel_scale_arcsec: float,
    ):
        self.matched_data = matched_data
        self.image_shape = image_shape
        self.pixel_scale = pixel_scale_arcsec
        self.ccd_center = (image_shape[1] / 2.0, image_shape[0] / 2.0)
        # Diagnostics
        self.radial_profile: list[dict] = []
        self.edge_vs_centre: dict = {}
        # Correction
        self.applied: bool = False
        self.rms_before_mas: float | None = None
        self.rms_after_mas: float | None = None
        self.cv_rms_before_mas: float | None = None
        self.cv_rms_after_mas: float | None = None
        self.tree_ring_periods_px: list[float] = [200.0, 300.0, 500.0]
        self.tree_ring_coeffs: np.ndarray | None = None  # (a_k, phi_k) for dx, dy

    def run(self, apply_correction: bool = True) -> dict:
        if self.matched_data is None or len(self.matched_data) < 15:
            logger.info("CCDEdgeDiagnostic: too few stars, skipping.")
            return {"applied": False}

        md = self.matched_data
        ny, nx = self.image_shape
        cx, cy = self.ccd_center
        x = md["x_pixel"].values
        y = md["y_pixel"].values
        r = np.hypot(x - cx, y - cy)

        cos_dec = np.cos(np.radians(md["dec_gaia"].values))
        dra_mas = (md["ra_final"] - md["ra_gaia"]).values * 3600e3 * cos_dec
        ddec_mas = (md["dec_final"] - md["dec_gaia"]).values * 3600e3

        # --- Radial profile diagnostics (5 bins) ---
        r_max = float(np.hypot(nx / 2, ny / 2))
        r_edges = np.linspace(0, r_max, 6)
        self.radial_profile = []
        for i in range(len(r_edges) - 1):
            mask = (r >= r_edges[i]) & (r < r_edges[i + 1])
            if mask.sum() < 1:
                continue
            self.radial_profile.append(
                {
                    "r_lo_px": float(r_edges[i]),
                    "r_hi_px": float(r_edges[i + 1]),
                    "n": int(mask.sum()),
                    "rms_tot_mas": float(
                        np.sqrt(np.mean(dra_mas[mask] ** 2 + ddec_mas[mask] ** 2))
                    ),
                    "median_r_mas": float(
                        np.median(np.hypot(dra_mas[mask], ddec_mas[mask]))
                    ),
                    "mean_dra_mas": float(np.mean(dra_mas[mask])),
                    "mean_ddec_mas": float(np.mean(ddec_mas[mask])),
                }
            )

        # Edge = outer 25% radius, centre = inner 25% radius
        r_quart = np.percentile(r, [25, 75])
        centre_mask = r < r_quart[0]
        edge_mask = r > r_quart[1]

        def rms(arr):
            return float(np.sqrt(np.mean(arr**2))) if len(arr) > 0 else float("nan")

        self.edge_vs_centre = {
            "centre_n": int(centre_mask.sum()),
            "edge_n": int(edge_mask.sum()),
            "centre_rms_mas": rms(
                np.hypot(dra_mas[centre_mask], ddec_mas[centre_mask])
            ),
            "edge_rms_mas": rms(np.hypot(dra_mas[edge_mask], ddec_mas[edge_mask])),
            "centre_median_r_mas": (
                float(np.median(np.hypot(dra_mas[centre_mask], ddec_mas[centre_mask])))
                if centre_mask.any()
                else None
            ),
            "edge_median_r_mas": (
                float(np.median(np.hypot(dra_mas[edge_mask], ddec_mas[edge_mask])))
                if edge_mask.any()
                else None
            ),
        }
        logger.info(
            "CCD edge diagnostic: centre RMS = %.2f mas (N=%d), "
            "edge RMS = %.2f mas (N=%d), edge/centre ratio = %.2f",
            self.edge_vs_centre["centre_rms_mas"],
            self.edge_vs_centre["centre_n"],
            self.edge_vs_centre["edge_rms_mas"],
            self.edge_vs_centre["edge_n"],
            self.edge_vs_centre["edge_rms_mas"]
            / max(self.edge_vs_centre["centre_rms_mas"], 1e-6),
        )

        # --- Concentric-ring correction ---------------------------------
        # Model radial shifts using sinusoidal terms at tree-ring periods
        # plus a linear trend, projected onto radial unit vectors (dx, dy).
        r_safe = np.where(r > 1.0, r, 1.0)
        cos_theta = (x - cx) / r_safe
        sin_theta = (y - cy) / r_safe

        # Radial design matrix: [1, r_norm, sin(2pi r/P_k), cos(2pi r/P_k)]
        r_norm = r / r_max  # normalize to [0, 1]
        radial_features = [np.ones_like(r), r_norm]
        for P in self.tree_ring_periods_px:
            omega = 2.0 * np.pi / P
            radial_features.append(np.sin(omega * r))
            radial_features.append(np.cos(omega * r))
        A_rad = np.column_stack(radial_features)

        # The radial amplitude vector A(r) gives shifts along (cos theta, sin theta).
        # We model dx = A_rad @ c_x_radial * cos_theta, dy = A_rad @ c_y_radial * sin_theta.
        # But to allow independent radial amplitudes for x and y (the
        # distortion isn't perfectly radial), we fit two separate models.
        # Weighted by SNR if available.
        if "mag_weight" in md.columns:
            w = md["mag_weight"].values
        else:
            w = np.ones(len(md))
        sw = np.sqrt(w)

        # Build the full design matrix for dx (cos_theta * radial_features) and dy (sin_theta * radial_features)
        # We stack them and fit jointly to dra_mas and ddec_mas.
        Ax = A_rad * cos_theta[:, None]
        Ay = A_rad * sin_theta[:, None]

        # LOO-CV to decide whether to apply the correction
        n = len(md)
        cv_before_ra = np.zeros(n)
        cv_after_ra = np.zeros(n)
        cv_before_dec = np.zeros(n)
        cv_after_dec = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            Axm = Ax[mask] * sw[mask][:, None]
            Aym = Ay[mask] * sw[mask][:, None]
            try:
                cx_, *_ = np.linalg.lstsq(Axm, dra_mas[mask] * sw[mask], rcond=None)
                cy_, *_ = np.linalg.lstsq(Aym, ddec_mas[mask] * sw[mask], rcond=None)
                cv_after_ra[i] = dra_mas[i] - Ax[i] @ cx_
                cv_after_dec[i] = ddec_mas[i] - Ay[i] @ cy_
            except np.linalg.LinAlgError:
                cv_after_ra[i] = dra_mas[i]
                cv_after_dec[i] = ddec_mas[i]
            cv_before_ra[i] = dra_mas[i]
            cv_before_dec[i] = ddec_mas[i]
        cv_before = float(np.sqrt(np.mean(cv_before_ra**2 + cv_before_dec**2)))
        cv_after = float(np.sqrt(np.mean(cv_after_ra**2 + cv_after_dec**2)))

        self.cv_rms_before_mas = cv_before
        self.cv_rms_after_mas = cv_after

        # Apply correction only if LOO-CV improves by >=1%
        if apply_correction and cv_after < cv_before * 0.99:
            # Fit on full sample
            try:
                cx_full, *_ = np.linalg.lstsq(
                    Ax * sw[:, None], dra_mas * sw, rcond=None
                )
                cy_full, *_ = np.linalg.lstsq(
                    Ay * sw[:, None], ddec_mas * sw, rcond=None
                )
            except np.linalg.LinAlgError:
                logger.warning("CCDEdgeDiagnostic: lstsq failed, no correction.")
                return self._telemetry()

            # Predict and subtract
            dra_pred = Ax @ cx_full
            ddec_pred = Ay @ cy_full
            # Apply: subtract predicted shift from ra_final / dec_final
            # (and from ra_measured / dec_measured so downstream diagnostics
            # see the corrected values)
            self.matched_data["ra_final"] = (
                self.matched_data["ra_final"] - dra_pred / 3600e3 / cos_dec
            )
            self.matched_data["dec_final"] = (
                self.matched_data["dec_final"] - ddec_pred / 3600e3
            )
            self.matched_data["ra_measured"] = (
                self.matched_data["ra_measured"] - dra_pred / 3600e3 / cos_dec
            )
            self.matched_data["dec_measured"] = (
                self.matched_data["dec_measured"] - ddec_pred / 3600e3
            )

            # Recompute residual RMS
            dra_new = (
                (self.matched_data["ra_final"] - self.matched_data["ra_gaia"]).values
                * 3600e3
                * cos_dec
            )
            ddec_new = (
                self.matched_data["dec_final"] - self.matched_data["dec_gaia"]
            ).values * 3600e3
            self.rms_before_mas = float(np.sqrt(np.mean(dra_mas**2 + ddec_mas**2)))
            self.rms_after_mas = float(np.sqrt(np.mean(dra_new**2 + ddec_new**2)))
            self.tree_ring_coeffs = (cx_full, cy_full)
            self.applied = True
            logger.info(
                "Tree-ring/edge correction APPLIED: RMS %.2f→%.2f mas "
                "(LOO-CV %.2f→%.2f mas) | periods=%s px",
                self.rms_before_mas,
                self.rms_after_mas,
                cv_before,
                cv_after,
                self.tree_ring_periods_px,
            )
        else:
            self.rms_before_mas = float(np.sqrt(np.mean(dra_mas**2 + ddec_mas**2)))
            self.rms_after_mas = self.rms_before_mas
            logger.info(
                "Tree-ring/edge correction skipped (LOO-CV did not improve: "
                "%.2f→%.2f mas). Edge/centre ratio = %.2f.",
                cv_before,
                cv_after,
                self.edge_vs_centre["edge_rms_mas"]
                / max(self.edge_vs_centre["centre_rms_mas"], 1e-6),
            )

        return self._telemetry()

    def _telemetry(self) -> dict:
        return {
            "applied": self.applied,
            "radial_profile": self.radial_profile,
            "edge_vs_centre": self.edge_vs_centre,
            "rms_before_mas": self.rms_before_mas,
            "rms_after_mas": self.rms_after_mas,
            "cv_rms_before_mas": self.cv_rms_before_mas,
            "cv_rms_after_mas": self.cv_rms_after_mas,
            "tree_ring_periods_px": self.tree_ring_periods_px,
        }


# --- Telemetry report writer ---


def write_telemetry_report(
    path,
    instrument: str,
    holdout_cv: dict,
    mag_strat: dict,
    dcr: dict,
    ccd_edge: dict,
    final_rms_mas: float,
    n_final: int,
):
    """Writes a markdown telemetry report for the calibration enhancements."""

    def _f(v, default=float("nan")):
        return default if v is None else v

    lines = [
        "# Enhanced Astrometric Pipeline — Telemetry Report",
        f"**Instrument:** {instrument}",
        f"**Final in-sample RMS:** {final_rms_mas:.3f} mas (N={n_final})",
        "",
        "## 1. Strict Hold-out Cross-Validation",
        "",
        f"- Number of splits: **{holdout_cv.get('n_splits', 0)}**",
        "- Train fraction: 70%",
        f"- Train RMS (mean ± std): **{_f(holdout_cv.get('train_rms_mean_mas')):.2f} ± "
        f"{_f(holdout_cv.get('train_rms_std_mas')):.2f} mas**",
        f"- **TEST RMS (mean ± std): {_f(holdout_cv.get('test_rms_mean_mas')):.2f} ± "
        f"{_f(holdout_cv.get('test_rms_std_mas')):.2f} mas**",
        f"- Overfit ratio (train/test): {_f(holdout_cv.get('overfit_ratio')):.3f}",
        "",
        "> The TEST RMS is computed on stars that were **NOT** used to fit the",
        "> astrometric transformation matrix.  This is an unbiased estimate of",
        "> the pipeline's predictive precision on a NEW star.",
        "",
        "## 2. Magnitude Stratification & Faint-Star Anchoring",
        "",
        "| G_lo | G_hi | N | RMS RA (mas) | RMS Dec (mas) | RMS Tot (mas) | "
        "Median sep (mas) | Gaia floor (mas) | Centroid noise (mas) |",
        "|------|------|---|--------------|---------------|---------------|"
        "------------------|------------------|----------------------|",
    ]
    for b in mag_strat.get("magnitude_bins", []):
        lines.append(
            f"| {b['lo']} | {b['hi']} | {b['n']} | {b['rms_ra_mas']:.2f} | "
            f"{b['rms_dec_mas']:.2f} | {b['rms_tot_mas']:.2f} | "
            f"{b['median_sep_mas']:.2f} | {b['gaia_floor_mas']:.3f} | "
            f"{b['centroid_noise_mas']:.3f} |"
        )
    lines.append("")
    if mag_strat.get("photon_break_mag") is not None:
        lines.append(
            f"**Photon-noise break magnitude:** G ≈ {mag_strat['photon_break_mag']:.2f}"
        )
    else:
        lines.append(
            "**Photon-noise break magnitude:** not identified (RMS does not "
            "exceed √2 × minimum within the observed G range)."
        )
    lines.append("")
    fs = mag_strat.get("faint_summary")
    if fs:
        lines.append(
            f"**Faint-star local anchoring (G ≥ 21):** N = {fs['n_faint']}, "
            f"local residual RMS = **{fs['local_rms_mas']:.2f} mas**, "
            f"median sep = {fs['local_median_sep_mas']:.2f} mas, "
            f"95th pct = {fs['local_p95_sep_mas']:.2f} mas, "
            f"mean # anchors = {fs['mean_n_anchor']:.1f}"
        )
    else:
        lines.append(
            "**Faint-star local anchoring:** not applicable (no G ≥ 21 stars "
            "in the Gaia reference catalog for this field)."
        )
    lines.extend(
        [
            "",
            "## 3. Differential Chromatic Refraction (DCR)",
            "",
            f"- Applied: **{dcr.get('applied', False)}**",
            f"- Best-fit coefficients: a = {_f(dcr.get('coef_a_mas')):.2f} mas, "
            f"b = {_f(dcr.get('coef_b_mas')):.2f} mas/mag",
            f"- Median zenith distance: {_f(dcr.get('median_zd_deg')):.1f}°",
            f"- Median tan(zd): {_f(dcr.get('median_tan_zd')):.3f}",
            f"- RMS before: {_f(dcr.get('rms_before_mas')):.2f} mas",
            f"- RMS after:  {_f(dcr.get('rms_after_mas')):.2f} mas",
            f"- LOO-CV before: {_f(dcr.get('cv_rms_before_mas')):.2f} mas",
            f"- LOO-CV after:  {_f(dcr.get('cv_rms_after_mas')):.2f} mas",
            "",
            "## 4. CCD Edge Effects & Tree-Ring Diagnostics",
            "",
            f"- Applied: **{ccd_edge.get('applied', False)}**",
            f"- Tree-ring periods probed: {ccd_edge.get('tree_ring_periods_px', [])} px",
            "",
            "**Edge vs. centre RMS:**",
            "",
            f"- Centre (inner 25% radius): N = {ccd_edge.get('edge_vs_centre', {}).get('centre_n', 0)}, "
            f"RMS = {_f(ccd_edge.get('edge_vs_centre', {}).get('centre_rms_mas')):.2f} mas",
            f"- Edge (outer 25% radius): N = {ccd_edge.get('edge_vs_centre', {}).get('edge_n', 0)}, "
            f"RMS = {_f(ccd_edge.get('edge_vs_centre', {}).get('edge_rms_mas')):.2f} mas",
            "",
            "**Radial residual profile:**",
            "",
            "| r_lo (px) | r_hi (px) | N | RMS (mas) | Median r (mas) | Mean dra (mas) | Mean ddec (mas) |",
            "|-----------|-----------|---|-----------|----------------|----------------|-----------------|",
        ]
    )
    for r in ccd_edge.get("radial_profile", []):
        lines.append(
            f"| {r['r_lo_px']:.0f} | {r['r_hi_px']:.0f} | {r['n']} | "
            f"{r['rms_tot_mas']:.2f} | {r['median_r_mas']:.2f} | "
            f"{r['mean_dra_mas']:.2f} | {r['mean_ddec_mas']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"- RMS before correction: {_f(ccd_edge.get('rms_before_mas')):.2f} mas",
            f"- RMS after correction:  {_f(ccd_edge.get('rms_after_mas')):.2f} mas",
            f"- LOO-CV before: {_f(ccd_edge.get('cv_rms_before_mas')):.2f} mas",
            f"- LOO-CV after:  {_f(ccd_edge.get('cv_rms_after_mas')):.2f} mas",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
