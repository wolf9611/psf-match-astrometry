"""
Astrometric calibration with geometric distortion, magnitude systematics,
thin-plate spline spatial modelling, and Gaussian process residual correction.
"""

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs import utils
from scipy.interpolate import RBFInterpolator
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from src.config import CATALOGS_DIR, LOGS_DIR, config
from src.gpu import Backend
from src.logger import get_logger
from src.plotting import (
    generate_advanced_plots,
    generate_rms_bootstrap_histogram,
    generate_rms_histogram,
    generate_weight_vs_magnitude_plot,
    plot_final_astrometry,
)

# Module-level logger for sub-pixel diagnostic tracking
logger = get_logger("calibration")


class Calibration:
    """
    Applies geometric distortion, magnitude-dependent systematic, spatial
    (TPS/GP), and per-axis sigma-clipped astrometric corrections to
    matched source catalogs.
    """

    def __init__(
        self,
        wcs,
        gaia_corrected,
        estimated_fwhm,
        image_shape,
        sources,
        milliquas_catalog=None,
        catalogs_dir=CATALOGS_DIR,
        logs_dir=LOGS_DIR,
        plots_dir=None,
    ):
        self.wcs = wcs
        self.gaia_corrected = gaia_corrected
        self.estimated_fwhm = estimated_fwhm
        self.image_shape = image_shape
        self.sources = sources
        self.milliquas_catalog = milliquas_catalog
        self.catalogs_dir = catalogs_dir
        self.logs_dir = logs_dir
        self.plots_dir = plots_dir
        self.matched_data = None
        self.rms_history = []
        self.rms_stage_names = []  # parallel list of human-readable stage names
        self.models = {}
        # Leave-one-out CV RMS for the spatial model (set in _tps_calibration)
        self.loo_cv_rms_arcsec = None
        # Estimated airmass at the observation site
        self.airmass = self._estimate_airmass()
        logger.info(
            "Calibration initialised | image_shape=%s | FWHM=%.3f px | airmass≈%.4f",
            image_shape,
            estimated_fwhm,
            self.airmass,
        )

    def _estimate_airmass(self):
        """Estimates airmass from WCS centre, observation time, and site
        latitude. Returns 1.2 if required metadata is missing."""
        try:
            from astropy.time import Time

            ra_center = float(self.wcs.wcs.crval[0])
            dec_center = float(self.wcs.wcs.crval[1])
            # CTIO latitude (Cerro Tololo)
            site_lat = -30.17
            mjd_obs = self.wcs.to_header().get("MJD-OBS")
            if mjd_obs is None:
                logger.info("Airmass: MJD-OBS missing, defaulting to 1.20")
                return 1.20
            t = Time(float(mjd_obs), format="mjd", scale="utc")
            site_lon = -70.81  # degrees west
            lst_deg = float(t.sidereal_time("apparent", longitude=site_lon * u.deg).deg)
            # Hour angle of field centre, wrapped to [-180, 180]
            ha_deg = ((lst_deg - ra_center + 180.0) % 360.0) - 180.0
            ha_rad = np.radians(ha_deg)
            dec_rad = np.radians(dec_center)
            lat_rad = np.radians(site_lat)
            sinh = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(
                dec_rad
            ) * np.cos(ha_rad)
            sinh = float(np.clip(sinh, -1.0, 1.0))
            zd_rad = np.arccos(sinh)
            sec_z = 1.0 / max(np.cos(zd_rad), 1e-6)
            # Negligible at ZD < 60 deg
            airmass = float(min(sec_z, 3.0))
            logger.info(
                "Airmass: RA=%.4f°, Dec=%.4f°, HA=%.2f°, ZD=%.2f°, X=%.4f",
                ra_center,
                dec_center,
                ha_deg,
                np.degrees(zd_rad),
                airmass,
            )
            return airmass
        except (ValueError, TypeError, AttributeError, KeyError, ImportError) as exc:
            logger.debug("Airmass estimation failed (%s); defaulting to 1.20", exc)
            return 1.20

    @staticmethod
    def _snr_weight(
        gaia_mag,
        fwhm_pixels,
        pixel_scale_arcsec,
        gaia_ra_error=None,
        gaia_dec_error=None,
    ):
        """
        Computes inverse-variance astrometric weights from SNR and Gaia's
        per-star positional uncertainty.

        Uses the centroiding error formula sigma_centroid ~ FWHM / (2.355 * SNR)
        with SNR calibrated to the DES r-band 5-sigma depth (23.7 mag).
        The Gaia floor uses the per-star astrometric error (ra_error/dec_error)
        when available, falling back to a magnitude-dependent approximation.

        Returns normalised weights (mean = 1) and per-source total sigma
        in mas (quadrature sum of centroid noise + Gaia floor).
        """
        des_depth = 23.7
        snr = 5.0 * 10.0 ** ((des_depth - gaia_mag) / 2.5)
        snr = np.clip(snr, 1.0, 5.0e4)
        fwhm_arcsec = fwhm_pixels * pixel_scale_arcsec
        sigma_centroid_mas = (fwhm_arcsec / (2.355 * snr)) * 1000.0  # mas

        # Gaia astrometric floor: prefer per-star errors from catalog
        if (
            gaia_ra_error is not None
            and gaia_dec_error is not None
            and np.any(np.isfinite(gaia_ra_error))
            and np.any(np.isfinite(gaia_dec_error))
        ):
            ra_err = np.where(np.isfinite(gaia_ra_error), gaia_ra_error, 0.5)
            dec_err = np.where(np.isfinite(gaia_dec_error), gaia_dec_error, 0.5)
            gaia_floor_mas = np.sqrt(ra_err**2 + dec_err**2)
        else:
            # Magnitude-based approximation: G<15 -> 0.05 mas,
            # G=18 -> 0.2 mas, G=21 -> 1.0 mas
            gaia_floor_mas = (
                0.05 + 1.0 * np.clip((gaia_mag - 16.0) / 5.0, 0.0, 1.0) ** 2
            )

        sigma_total = np.sqrt(sigma_centroid_mas**2 + gaia_floor_mas**2)
        weights = 1.0 / sigma_total**2
        w_median = np.median(weights)
        if w_median > 0:
            weights = np.clip(weights / w_median, 0.1, 10.0)
        weights = weights / np.mean(weights) * len(weights)
        return weights, sigma_total

    def model_systematic_corrections(
        self, matches_file, sip_order=None, n_iterations=None
    ):
        """
        Fits geometric distortion (SIP) and magnitude-dependent systematic
        corrections iteratively with sigma-clipped outlier rejection.
        """
        if sip_order is None:
            sip_order = config.sip_order
        if n_iterations is None:
            n_iterations = config.sip_n_iterations

        if not matches_file.exists():
            logger.error(
                "Matches file %s not found for systematic modeling", matches_file
            )
            print(f"File {matches_file} not found for systematic modeling")
            return None

        matches = pd.read_csv(matches_file)
        # Remove rows with NaN/inf in critical columns
        crit_cols = [
            "ra_psf",
            "dec_psf",
            "ra_gaia_corr",
            "dec_gaia_corr",
            "ra_diff_arcsec",
            "dec_diff_arcsec",
            "phot_g_mean_mag",
        ]
        crit_cols = [c for c in crit_cols if c in matches.columns]
        n_before = len(matches)
        matches = matches.dropna(subset=crit_cols).reset_index(drop=True)
        # Replace inf with NaN then drop
        matches = (
            matches.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=crit_cols)
            .reset_index(drop=True)
        )
        if len(matches) < n_before:
            logger.warning(
                "Dropped %d/%d rows with NaN/inf in critical columns",
                n_before - len(matches),
                n_before,
            )

        # Reduce SIP order if insufficient stars for the requested order
        n_terms_order = lambda o: (o + 1) * (o + 2) // 2 - 1
        while sip_order > 1 and len(matches) < 3 * n_terms_order(sip_order):
            sip_order -= 1
            logger.info(
                "Reducing SIP order to %d (only %d stars available)",
                sip_order,
                len(matches),
            )

        if "x_pixel" not in matches.columns:
            x_pix, y_pix = self.wcs.all_world2pix(
                matches["ra_psf"], matches["dec_psf"], 0
            )
            matches["x_pixel"] = x_pix
            matches["y_pixel"] = y_pix

        logger.info(
            "SIP+mag systematic modelling: %d input pairs | order=%d | iter=%d",
            len(matches),
            sip_order,
            n_iterations,
        )
        final_matches = matches
        for iteration in range(n_iterations):
            sip_model = self._model_geometric_distortion(matches, order=sip_order)
            matches_corrected = self._apply_sip_corrections(matches, sip_model)
            systematic_model = self._model_magnitude_systematics(matches_corrected)
            final_matches = self._apply_all_corrections(
                matches, sip_model, systematic_model
            )

            sep = np.sqrt(
                final_matches["ra_diff_final_arcsec"] ** 2
                + final_matches["dec_diff_final_arcsec"] ** 2
            )
            rms = np.sqrt(np.mean(sep**2))
            logger.info(
                "  SIP iter %d/%d: N=%d | RMS=%.4f arcsec (%.2f mas) | median=%.4f arcsec",
                iteration + 1,
                n_iterations,
                len(final_matches),
                rms,
                rms * 1000.0,
                np.median(sep),
            )

            if iteration < n_iterations - 1:
                tight_mask = sep < 2.5 * rms
                n_before = len(final_matches)
                matches = final_matches[tight_mask].copy()
                matches["ra_diff_arcsec"] = matches["ra_diff_final_arcsec"]
                matches["dec_diff_arcsec"] = matches["dec_diff_final_arcsec"]
                logger.debug(
                    "    iter %d kept %d/%d pairs after 2.5*RMS clip",
                    iteration + 1,
                    len(matches),
                    n_before,
                )

        try:
            pixel_scales = utils.proj_plane_pixel_scales(self.wcs)
            pixel_scale_arcsec = np.mean(pixel_scales) * 3600.0
        except (ValueError, TypeError, AttributeError):
            pixel_scale_arcsec = None

        plot_final_astrometry(
            final_matches,
            self.estimated_fwhm,
            pixel_scale_arcsec,
            plots_dir=self.plots_dir,
        )
        self._save_refined_pixel_coordinates(final_matches)
        return final_matches

    def _model_geometric_distortion(self, matches, order=2, alpha=1e-4):
        """Fits a polynomial distortion model (SIP-like) using Ridge regression."""
        x_ref, y_ref = self.image_shape[1] / 2, self.image_shape[0] / 2
        x_norm = (matches["x_pixel"] - x_ref) / x_ref
        y_norm = (matches["y_pixel"] - y_ref) / y_ref
        pixel_scale = np.abs(np.diag(self.wcs.pixel_scale_matrix)).mean() * 3600
        dx_obs = matches["ra_diff_arcsec"] / pixel_scale
        dy_obs = matches["dec_diff_arcsec"] / pixel_scale

        powers, cols = [], []
        for i in range(order + 1):
            for j in range(order + 1 - i):
                if i + j > 0:
                    powers.append((i, j))
                    cols.append((x_norm**i) * (y_norm**j))

        A_matrix = np.vstack(cols).T
        ridge_x = Ridge(alpha=alpha).fit(A_matrix, dx_obs)
        ridge_y = Ridge(alpha=alpha).fit(A_matrix, dy_obs)

        logger.debug(
            "  SIP order=%d: %d terms, %d pairs | max|a|=%.4f, max|b|=%.4f pix",
            order,
            len(powers),
            len(matches),
            float(np.max(np.abs(ridge_x.coef_))) if len(ridge_x.coef_) else 0.0,
            float(np.max(np.abs(ridge_y.coef_))) if len(ridge_y.coef_) else 0.0,
        )

        return {
            "powers": powers,
            "a_coeffs": ridge_x.coef_,
            "b_coeffs": ridge_y.coef_,
            "crpix": (x_ref, y_ref),
            "pixel_scale": pixel_scale,
        }

    def _apply_sip_corrections(self, matches, sip_model):
        """Applies the SIP polynomial distortion corrections to a matched catalog."""
        x_ref, y_ref = sip_model["crpix"]
        pixel_scale = sip_model["pixel_scale"]
        x_norm = (matches["x_pixel"] - x_ref) / x_ref
        y_norm = (matches["y_pixel"] - y_ref) / y_ref

        dx_pred, dy_pred = np.zeros(len(matches)), np.zeros(len(matches))
        for idx, (i, j) in enumerate(sip_model["powers"]):
            dx_pred += sip_model["a_coeffs"][idx] * (x_norm**i) * (y_norm**j)
            dy_pred += sip_model["b_coeffs"][idx] * (x_norm**i) * (y_norm**j)

        matches_corrected = matches.copy()
        matches_corrected["ra_diff_sip_arcsec"] = (
            matches["ra_diff_arcsec"] / pixel_scale - dx_pred
        ) * pixel_scale
        matches_corrected["dec_diff_sip_arcsec"] = (
            matches["dec_diff_arcsec"] / pixel_scale - dy_pred
        ) * pixel_scale
        return matches_corrected

    def _model_magnitude_systematics(self, matches_corrected):
        """Fits magnitude and colour-dependent systematic corrections
        using Ridge regression with optional DCR-aware design matrix."""
        dx_obs = matches_corrected["ra_diff_sip_arcsec"].values
        dy_obs = matches_corrected["dec_diff_sip_arcsec"].values
        mag = matches_corrected["phot_g_mean_mag"].values
        pixel_scale = np.abs(np.diag(self.wcs.pixel_scale_matrix)).mean() * 3600
        dx_pix, dy_pix = dx_obs / pixel_scale, dy_obs / pixel_scale

        has_color = (
            "bp_rp" in matches_corrected.columns
            and not matches_corrected["bp_rp"].isna().all()
        )
        if has_color:
            color = matches_corrected["bp_rp"].values
            color = np.where(~np.isnan(color), color, np.nanmedian(color))
            # Quadratic colour term and DCR interaction: (airmass - 1) * colour
            color2 = color**2
            dcr_term = (self.airmass - 1.0) * color
            A_matrix = np.column_stack(
                [np.ones(len(mag)), mag, color, color2, dcr_term]
            )
        else:
            A_matrix = np.column_stack([np.ones(len(mag)), mag])

        ridge_dx = Ridge(alpha=1e-3, fit_intercept=False).fit(A_matrix, dx_pix)
        ridge_dy = Ridge(alpha=1e-3, fit_intercept=False).fit(A_matrix, dy_pix)

        if has_color:
            logger.debug(
                "  Mag+color model coeffs (pix): dx=[%s] dy=[%s]",
                ", ".join(f"{c:+.4f}" for c in ridge_dx.coef_),
                ", ".join(f"{c:+.4f}" for c in ridge_dy.coef_),
            )
        else:
            logger.debug(
                "  Mag-only model coeffs (pix): dx=[%s] dy=[%s]",
                ", ".join(f"{c:+.4f}" for c in ridge_dx.coef_),
                ", ".join(f"{c:+.4f}" for c in ridge_dy.coef_),
            )

        return {
            "ridge_dx": ridge_dx,
            "ridge_dy": ridge_dy,
            "has_color": has_color,
            "pixel_scale": pixel_scale,
        }

    def _apply_all_corrections(self, matches, sip_model, systematic_model):
        """Applies both SIP geometric and magnitude systematic corrections
        to produce final corrected residuals."""
        x_ref, y_ref = sip_model["crpix"]
        pixel_scale = sip_model["pixel_scale"]
        x_norm = (matches["x_pixel"] - x_ref) / x_ref
        y_norm = (matches["y_pixel"] - y_ref) / y_ref

        dx_obs = matches["ra_diff_arcsec"] / pixel_scale
        dy_obs = matches["dec_diff_arcsec"] / pixel_scale
        dx_geom, dy_geom = np.zeros_like(dx_obs), np.zeros_like(dy_obs)

        for idx, (i, j) in enumerate(sip_model["powers"]):
            dx_geom += sip_model["a_coeffs"][idx] * (x_norm**i) * (y_norm**j)
            dy_geom += sip_model["b_coeffs"][idx] * (x_norm**i) * (y_norm**j)

        dx_res, dy_res = dx_obs - dx_geom, dy_obs - dy_geom

        mag = matches["phot_g_mean_mag"].values
        has_color = systematic_model.get("has_color", False)
        if has_color and "bp_rp" in matches.columns:
            color = matches["bp_rp"].values
            color = np.where(np.isnan(color), np.nanmedian(color), color)
            color2 = color**2
            dcr_term = (self.airmass - 1.0) * color
            A_matrix = np.column_stack(
                [np.ones(len(mag)), mag, color, color2, dcr_term]
            )
        else:
            A_matrix = np.column_stack([np.ones(len(mag)), mag])

        dx_mag = systematic_model["ridge_dx"].predict(A_matrix)
        dy_mag = systematic_model["ridge_dy"].predict(A_matrix)

        dx_final, dy_final = dx_res - dx_mag, dy_res - dy_mag

        final_matches = matches.copy()
        final_matches["ra_diff_final_arcsec"] = dx_final * pixel_scale
        final_matches["dec_diff_final_arcsec"] = dy_final * pixel_scale
        return final_matches

    def _save_refined_pixel_coordinates(self, final_matches):
        ra_corr = final_matches["ra_gaia_corr"] + final_matches[
            "ra_diff_final_arcsec"
        ] / 3600.0 / np.cos(np.radians(final_matches["dec_gaia_corr"]))
        dec_corr = (
            final_matches["dec_gaia_corr"]
            + final_matches["dec_diff_final_arcsec"] / 3600.0
        )
        x_refined, y_refined = self.wcs.all_world2pix(ra_corr, dec_corr, 0)

        with open(self.catalogs_dir / "xy.txt", "w", encoding="utf-8") as f:
            f.writelines(
                f"{xr:12.6f} {y_refined[i]:12.6f}\n" for i, xr in enumerate(x_refined)
            )

        pd.DataFrame(
            {
                "x_pixel_refined": x_refined,
                "y_pixel_refined": y_refined,
                "x_pixel_original": final_matches["x_pixel"],
                "y_pixel_original": final_matches["y_pixel"],
                "ra_corrected": ra_corr,
                "dec_corrected": dec_corr,
                "ra_original": final_matches["ra_psf"],
                "dec_original": final_matches["dec_psf"],
            }
        ).to_csv(self.catalogs_dir / "refined_coordinates_full.csv", index=False)

    def convert_xy_to_radec(
        self,
        xy_file=None,
        output_file=None,
    ):
        """
        Converts pixel coordinates to RA/DEC world coordinates.
        """
        if xy_file is None:
            xy_file = self.catalogs_dir / "xy.txt"
        if output_file is None:
            output_file = self.catalogs_dir / "xy_radec.csv"
        if not xy_file.exists():
            return False
        xy_data = pd.read_csv(
            xy_file,
            sep=r"\s+",
            header=None,
            names=["x_pixel", "y_pixel"],
            comment="#",
            engine="python",
        )
        ra_coords, dec_coords = self.wcs.all_pix2world(
            xy_data["x_pixel"].values, xy_data["y_pixel"].values, 0
        )
        pd.DataFrame(
            {
                "x_pixel": xy_data["x_pixel"].values,
                "y_pixel": xy_data["y_pixel"].values,
                "ra": ra_coords,
                "dec": dec_coords,
            }
        ).to_csv(output_file, index=False)
        return True

    def match_xy_radec_with_gaia(
        self,
        xy_radec_file=None,
        max_sep=2.0,
    ):
        """
        Performs a final validation match between refined coordinates and Gaia.
        """
        if xy_radec_file is None:
            xy_radec_file = self.catalogs_dir / "xy_radec.csv"
        if not xy_radec_file.exists():
            return False
        xy_radec = pd.read_csv(xy_radec_file)

        gaia_coords = SkyCoord(
            ra=self.gaia_corrected["ra_corr"].values * u.deg,
            dec=self.gaia_corrected["dec_corr"].values * u.deg,
        )
        xy_coords = SkyCoord(
            ra=xy_radec["ra"].values * u.deg, dec=xy_radec["dec"].values * u.deg
        )

        idx_fwd, d2d_fwd, _ = xy_coords.match_to_catalog_sky(gaia_coords)
        idx_rev, _, _ = gaia_coords.match_to_catalog_sky(xy_coords)

        mask = np.zeros(len(xy_coords), dtype=bool)
        for i in range(len(xy_coords)):
            if idx_rev[idx_fwd[i]] == i and d2d_fwd[i] <= max_sep * u.arcsec:
                mask[i] = True

        if np.sum(mask) == 0:
            return False

        gaia_m = self.gaia_corrected.iloc[idx_fwd[mask]]
        matches = pd.DataFrame(
            {
                "x_pixel": xy_radec.loc[mask, "x_pixel"].values,
                "y_pixel": xy_radec.loc[mask, "y_pixel"].values,
                "ra_xy": xy_radec.loc[mask, "ra"].values,
                "dec_xy": xy_radec.loc[mask, "dec"].values,
                "ra_gaia": gaia_m["ra_corr"].values,
                "dec_gaia": gaia_m["dec_corr"].values,
                "separation_arcsec": d2d_fwd[mask].arcsec,
            }
        )
        matches["ra_diff_arcsec"] = (
            (matches["ra_xy"] - matches["ra_gaia"])
            * 3600
            * np.cos(np.radians(matches["dec_gaia"]))
        )
        matches["dec_diff_arcsec"] = (matches["dec_xy"] - matches["dec_gaia"]) * 3600

        rms_total = np.sqrt(
            np.mean(matches["ra_diff_arcsec"] ** 2)
            + np.mean(matches["dec_diff_arcsec"] ** 2)
        )
        matches.to_csv(self.catalogs_dir / "final_matched_catalog.csv", index=False)
        generate_rms_histogram(matches, rms_total, plots_dir=self.plots_dir)
        return True

    def run_advanced_calibration(self):
        """
        Executes the advanced calibration chain: robust polynomial fitting,
        second-pass colour/magnitude correction, TPS/GP spatial residual
        modelling, quasar frame-tie, and per-axis sigma clipping.
        """
        if not (self.catalogs_dir / "xy_radec.csv").exists():
            return False
        if not self._prepare_advanced_data():
            return False

        self._robust_polynomial_fit()
        self._advanced_analysis_systematic()
        self._tps_calibration()

        # Per-axis sigma clipping on calibrated residuals. Removes outliers
        # in either coordinate axis independently (standard astrometric
        # practice for precision reporting).
        self._final_per_axis_sigma_clip()

        if self.milliquas_catalog is not None:
            self._apply_quasar_frame_tie()
        else:
            self.matched_data["ra_icrf"] = self.matched_data["ra_final"]
            self.matched_data["dec_icrf"] = self.matched_data["dec_final"]

        self._save_advanced_results()
        generate_advanced_plots(
            self.matched_data,
            self.rms_history,
            stage_names=self.rms_stage_names,
            plots_dir=self.plots_dir,
        )
        generate_weight_vs_magnitude_plot(self.matched_data, plots_dir=self.plots_dir)
        generate_rms_bootstrap_histogram(self.matched_data, plots_dir=self.plots_dir)
        return True

    def _final_per_axis_sigma_clip(self, sigma=None, max_iterations=5):
        """Iteratively clips outliers per coordinate axis using MAD-based
        thresholds, recomputing the MAD after each pass."""
        if sigma is None:
            sigma = config.final_sigma_clip_sigma
        if self.matched_data is None or len(self.matched_data) < 10:
            return

        for it in range(max_iterations):
            cos_dec = np.cos(np.radians(self.matched_data["dec_gaia"]))
            dra = (
                (self.matched_data["ra_final"] - self.matched_data["ra_gaia"])
                * 3600e3
                * cos_dec
            )
            ddec = (
                self.matched_data["dec_final"] - self.matched_data["dec_gaia"]
            ) * 3600e3
            med_ra, med_dec = float(np.median(dra)), float(np.median(ddec))
            mad_ra = float(np.median(np.abs(dra - med_ra)))
            mad_dec = float(np.median(np.abs(ddec - med_dec)))
            if mad_ra <= 0 or mad_dec <= 0:
                logger.warning("  Final per-axis clip iter %d: MAD=0, stopping", it + 1)
                break
            thr_ra = sigma * 1.4826 * mad_ra
            thr_dec = sigma * 1.4826 * mad_dec
            mask_keep = (np.abs(dra - med_ra) <= thr_ra) & (
                np.abs(ddec - med_dec) <= thr_dec
            )
            n_drop = int((~mask_keep).sum())
            if n_drop == 0:
                logger.info(
                    "  Final per-axis %.1fσ clip iter %d: 0 outliers, converged "
                    "(MAD_RA=%.2f mas, MAD_Dec=%.2f mas, N=%d)",
                    sigma,
                    it + 1,
                    mad_ra,
                    mad_dec,
                    len(self.matched_data),
                )
                break
            logger.info(
                "  Final per-axis %.1fσ clip iter %d: dropping %d/%d outliers "
                "(thr: |dra|<%.2f mas, |ddec|<%.2f mas)",
                sigma,
                it + 1,
                n_drop,
                len(self.matched_data),
                thr_ra,
                thr_dec,
            )
            self.matched_data = self.matched_data[mask_keep].reset_index(drop=True)

        # Final cleaned-sample RMS
        cos_dec = np.cos(np.radians(self.matched_data["dec_gaia"]))
        dra = (
            (self.matched_data["ra_final"] - self.matched_data["ra_gaia"])
            * 3600e3
            * cos_dec
        )
        ddec = (self.matched_data["dec_final"] - self.matched_data["dec_gaia"]) * 3600e3
        rms_clean = float(np.sqrt(np.mean(dra**2 + ddec**2)))  # in mas
        # Record the in-sample cleaned RMS separately from the LOO-CV
        # predictive precision (computed by HoldoutCV in enhancements.py).
        self.in_sample_cleaned_rms_arcsec = rms_clean / 1000.0
        # Append post-clip stage to RMS history
        self.rms_history.append(rms_clean / 1000.0)
        self.rms_stage_names.append("Final per-axis 3σ clip (in-sample)")
        logger.info(
            "  Final cleaned-sample (in-sample) RMS: %.3f mas (N=%d) — "
            "NOT a hold-out estimate; see HoldoutCV report for predictive RMS",
            rms_clean,
            len(self.matched_data),
        )

    def _prepare_advanced_data(self):
        xy_data = pd.read_csv(self.catalogs_dir / "xy_radec.csv")
        xy_coords = SkyCoord(ra=xy_data["ra"], dec=xy_data["dec"], unit="deg")
        gaia_coords = SkyCoord(
            ra=self.gaia_corrected["ra_corr"].values,
            dec=self.gaia_corrected["dec_corr"].values,
            unit="deg",
        )

        idx, d2d, _ = xy_coords.match_to_catalog_sky(gaia_coords)
        match_radius = config.advanced_match_radius_arcsec
        good_matches = d2d.arcsec < match_radius
        matched_xy = xy_data[good_matches].copy()
        matched_gaia = self.gaia_corrected.iloc[idx[good_matches]].copy()

        logger.info(
            "Advanced data prep: %d/%d matches within %.2f arcsec",
            int(good_matches.sum()),
            len(xy_data),
            match_radius,
        )

        self.matched_data = pd.DataFrame(
            {
                "x_pixel": matched_xy["x_pixel"].values,
                "y_pixel": matched_xy["y_pixel"].values,
                "ra_measured": matched_xy["ra"].values,
                "dec_measured": matched_xy["dec"].values,
                "ra_gaia": matched_gaia["ra_corr"].values,
                "dec_gaia": matched_gaia["dec_corr"].values,
                "gaia_mag": matched_gaia["phot_g_mean_mag"].values,
            }
        )
        # Attach bp_rp color when available for DCR modelling
        if "bp_rp" in matched_gaia.columns:
            self.matched_data["bp_rp"] = matched_gaia["bp_rp"].values
        # Attach per-star Gaia astrometric errors for SNR weighting
        for _err_col in ("ra_error", "dec_error", "pmra_error", "pmdec_error"):
            if _err_col in matched_gaia.columns:
                self.matched_data[f"gaia_{_err_col}"] = matched_gaia[_err_col].values
        self.matched_data["radial_distance"] = np.sqrt(
            (self.matched_data["x_pixel"] - np.mean(self.matched_data["x_pixel"])) ** 2
            + (self.matched_data["y_pixel"] - np.mean(self.matched_data["y_pixel"]))
            ** 2
        )

        self._calculate_advanced_rms(
            "ra_measured", "dec_measured", stage_name="Initial (raw match)"
        )
        return True

    def _robust_polynomial_fit(self):
        # Sigma-clip raw residuals with configurable threshold and iterations
        clip_sigma = config.sigma_clip_threshold_sigma
        clip_iters = config.sigma_clip_max_iterations
        current_data = self.matched_data.copy()
        for it in range(clip_iters):
            cos_dec = np.cos(np.radians(current_data["dec_gaia"]))
            ra_diff = (
                (current_data["ra_measured"] - current_data["ra_gaia"]) * 3600 * cos_dec
            )
            dec_diff = (current_data["dec_measured"] - current_data["dec_gaia"]) * 3600
            residuals = np.sqrt(ra_diff**2 + dec_diff**2)
            median_res = np.median(residuals)
            mad_res = np.median(np.abs(residuals - median_res))
            if mad_res <= 0:
                logger.debug("  sigma-clip iter %d: MAD=0, stopping", it + 1)
                break
            threshold = median_res + clip_sigma * 1.4826 * mad_res
            outliers = residuals > threshold
            n_outliers = int(np.sum(outliers))
            logger.debug(
                "  sigma-clip iter %d: median=%.3f mas, MAD=%.3f mas, "
                "thr=%.3f mas, outliers=%d/%d",
                it + 1,
                median_res * 1000,
                mad_res * 1000,
                threshold * 1000,
                n_outliers,
                len(current_data),
            )
            if n_outliers == 0:
                break
            current_data = current_data[~outliers]
        self.matched_data = current_data

        # SNR-based inverse-variance weights with per-star Gaia floor
        pixel_scale = np.abs(np.diag(self.wcs.pixel_scale_matrix)).mean() * 3600
        _gaia_ra_err = (
            self.matched_data["gaia_ra_error"].values
            if "gaia_ra_error" in self.matched_data.columns
            else None
        )
        _gaia_dec_err = (
            self.matched_data["gaia_dec_error"].values
            if "gaia_dec_error" in self.matched_data.columns
            else None
        )
        snr_weights, sigma_mas = self._snr_weight(
            self.matched_data["gaia_mag"].values,
            self.estimated_fwhm,
            pixel_scale,
            gaia_ra_error=_gaia_ra_err,
            gaia_dec_error=_gaia_dec_err,
        )
        self.matched_data["mag_weight"] = snr_weights
        self.matched_data["sigma_centroid_mas"] = sigma_mas
        logger.info(
            "  SNR weights: median=%.3f, min=%.3f, max=%.3f | σ_centroid median=%.2f mas",
            float(np.median(snr_weights)),
            float(np.min(snr_weights)),
            float(np.max(snr_weights)),
            float(np.median(sigma_mas)),
        )

        X_feats = self.matched_data[["x_pixel", "y_pixel"]].values
        cos_dec = np.cos(np.radians(self.matched_data["dec_gaia"]))
        ra_diff = (
            (self.matched_data["ra_measured"] - self.matched_data["ra_gaia"])
            * 3600
            * cos_dec
        )
        dec_diff = (
            self.matched_data["dec_measured"] - self.matched_data["dec_gaia"]
        ) * 3600

        # Normalise pixel coordinates to avoid ill-conditioned polynomial
        # design matrices
        x_mean, x_std = X_feats[:, 0].mean(), X_feats[:, 0].std()
        y_mean, y_std = X_feats[:, 1].mean(), X_feats[:, 1].std()
        x_std = x_std if x_std > 1e-9 else 1.0
        y_std = y_std if y_std > 1e-9 else 1.0
        X_norm = np.column_stack(
            [(X_feats[:, 0] - x_mean) / x_std, (X_feats[:, 1] - y_mean) / y_std]
        )
        self._poly_norm = (x_mean, x_std, y_mean, y_std)

        # Pre-fit RMS (before any advanced spatial correction)
        rms_prefit = float(np.sqrt(np.mean(ra_diff**2 + dec_diff**2)))
        logger.info(
            "  Polynomial pre-fit RMS = %.4f arcsec (%.2f mas) on %d stars",
            rms_prefit,
            rms_prefit * 1000.0,
            len(self.matched_data),
        )

        poly_degree = 3
        # Ridge regularisation scaled to dominate arcsec-scale residual values
        ridge_alpha = 1.0e6 * len(self.matched_data)
        X_poly = PolynomialFeatures(
            degree=poly_degree, include_bias=True
        ).fit_transform(X_norm)
        W_diag = np.sqrt(snr_weights)
        X_w = X_poly * W_diag[:, np.newaxis]

        ridge_ra = Ridge(alpha=ridge_alpha, fit_intercept=False).fit(
            X_w, ra_diff * W_diag
        )
        ridge_dec = Ridge(alpha=ridge_alpha, fit_intercept=False).fit(
            X_w, dec_diff * W_diag
        )

        # Leave-one-out CV to decide whether to commit the polynomial correction.
        # When the input is large enough, use the closed-form hat-matrix
        # formula (O(N²) memory, single solve) instead of the per-fold
        # re-solve loop.  The dispatcher falls back to the CPU loop for
        # small catalogs or when no GPU is available.
        ra_diff_np = np.asarray(ra_diff, dtype=float)
        dec_diff_np = np.asarray(dec_diff, dtype=float)
        X_poly_np = np.asarray(X_poly, dtype=float)
        W_diag_np = np.asarray(W_diag, dtype=float)
        n = len(X_poly_np)
        n_features = X_poly_np.shape[1]

        try:
            backend = Backend.instance()
            cv_ra = backend.closed_form_loocv_linear(
                X_poly_np, ra_diff_np, ridge_alpha, weights=W_diag_np**2
            )
            cv_dec = backend.closed_form_loocv_linear(
                X_poly_np, dec_diff_np, ridge_alpha, weights=W_diag_np**2
            )
            logger.debug("Polynomial LOO-CV via backend device: %s", backend.dm.device)
        except Exception:
            # Per-fold re-solve fallback (small N or backend failure)
            cv_ra = np.zeros(n)
            cv_dec = np.zeros(n)
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                Xw_tr = X_poly_np[mask] * W_diag_np[mask][:, None]
                y_ra_tr = ra_diff_np[mask] * W_diag_np[mask]
                y_dec_tr = dec_diff_np[mask] * W_diag_np[mask]
                A_mat = Xw_tr.T @ Xw_tr + ridge_alpha * np.eye(n_features)
                try:
                    cra = np.linalg.solve(A_mat, Xw_tr.T @ y_ra_tr)
                    cdec = np.linalg.solve(A_mat, Xw_tr.T @ y_dec_tr)
                    cv_ra[i] = ra_diff_np[i] - X_poly_np[i] @ cra
                    cv_dec[i] = dec_diff_np[i] - X_poly_np[i] @ cdec
                except (ValueError, TypeError, np.linalg.LinAlgError):
                    cv_ra[i] = ra_diff_np[i]
                    cv_dec[i] = dec_diff_np[i]
        rms_cv = float(np.sqrt(np.mean(cv_ra**2 + cv_dec**2)))
        logger.info(
            "  Polynomial LOO-CV RMS = %.4f arcsec (%.2f mas) | helps=%s",
            rms_cv,
            rms_cv * 1000.0,
            rms_cv < rms_prefit * 0.98,
        )

        # Only commit polynomial if LOO-CV improves by >= 2%
        if rms_cv < rms_prefit * 0.98:
            ra_corr = X_poly @ ridge_ra.coef_
            dec_corr = X_poly @ ridge_dec.coef_
            cos_dec_m = np.cos(np.radians(self.matched_data["dec_measured"]))
            self.matched_data["ra_weighted"] = (
                self.matched_data["ra_measured"] - ra_corr / 3600 / cos_dec_m
            )
            self.matched_data["dec_weighted"] = (
                self.matched_data["dec_measured"] - dec_corr / 3600
            )
            logger.info("  Polynomial correction committed (LOO-CV improved ≥2%%)")
        else:
            # Keep raw measured coordinates when polynomial does not help
            self.matched_data["ra_weighted"] = self.matched_data["ra_measured"]
            self.matched_data["dec_weighted"] = self.matched_data["dec_measured"]
            logger.info(
                "  Polynomial correction SKIPPED (LOO-CV did not improve ≥2%%) — "
                "systematic SIP+mag stage already removed spatial structure"
            )
        self._calculate_advanced_rms(
            "ra_weighted", "dec_weighted", stage_name="Polynomial+SNR-weighted"
        )

    def _advanced_analysis_systematic(self):
        self.matched_data["ra_residual"] = (
            (self.matched_data["ra_weighted"] - self.matched_data["ra_gaia"])
            * 3600
            * np.cos(np.radians(self.matched_data["dec_gaia"]))
        )
        self.matched_data["dec_residual"] = (
            self.matched_data["dec_weighted"] - self.matched_data["dec_gaia"]
        ) * 3600
        self.matched_data["total_residual"] = np.sqrt(
            self.matched_data["ra_residual"] ** 2
            + self.matched_data["dec_residual"] ** 2
        )

        # Second-pass colour and magnitude correction on post-polynomial
        # residuals, committed only if LOO-CV confirms improvement.
        if not (
            "bp_rp" in self.matched_data.columns
            and self.matched_data["bp_rp"].notna().any()
        ):
            return
        mag = self.matched_data["gaia_mag"].values
        color = self.matched_data["bp_rp"].values
        color = np.where(np.isnan(color), np.nanmedian(color), color)
        color2 = color**2
        dcr_term = (self.airmass - 1.0) * color
        A = np.column_stack([np.ones(len(mag)), mag, color, color2, dcr_term])

        ra_res = self.matched_data["ra_residual"].values
        dec_res = self.matched_data["dec_residual"].values
        w = self.matched_data["mag_weight"].values

        rms_before = float(np.sqrt(np.mean(ra_res**2 + dec_res**2)))

        # Leave-one-out CV for the colour/mag fit
        n = len(A)
        A_np = np.asarray(A, dtype=float)
        ra_res_np = np.asarray(ra_res, dtype=float)
        dec_res_np = np.asarray(dec_res, dtype=float)
        w_np = np.asarray(w, dtype=float)
        cv_err_ra = np.zeros(n)
        cv_err_dec = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            Aw = A_np[mask] * np.sqrt(w_np[mask])[:, None]
            try:
                cra, *_ = np.linalg.lstsq(
                    Aw, ra_res_np[mask] * np.sqrt(w_np[mask]), rcond=None
                )
                cdec, *_ = np.linalg.lstsq(
                    Aw, dec_res_np[mask] * np.sqrt(w_np[mask]), rcond=None
                )
                cv_err_ra[i] = ra_res_np[i] - A_np[i] @ cra
                cv_err_dec[i] = dec_res_np[i] - A_np[i] @ cdec
            except (ValueError, TypeError, np.linalg.LinAlgError):
                cv_err_ra[i] = ra_res_np[i]
                cv_err_dec[i] = dec_res_np[i]
        rms_after_cv = float(np.sqrt(np.mean(cv_err_ra**2 + cv_err_dec**2)))

        if rms_after_cv < rms_before * 0.999:
            Aw = A_np * np.sqrt(w_np)[:, None]
            try:
                coef_ra, *_ = np.linalg.lstsq(Aw, ra_res_np * np.sqrt(w_np), rcond=None)
                coef_dec, *_ = np.linalg.lstsq(
                    Aw, dec_res_np * np.sqrt(w_np), rcond=None
                )
                ra_corr = A_np @ coef_ra
                dec_corr = A_np @ coef_dec
                self.matched_data["ra_weighted"] = self.matched_data["ra_weighted"] - (
                    ra_corr
                    / 3600.0
                    / np.cos(np.radians(self.matched_data["dec_weighted"]))
                )
                self.matched_data["dec_weighted"] = self.matched_data[
                    "dec_weighted"
                ] - (dec_corr / 3600.0)
                # Recompute residuals
                self.matched_data["ra_residual"] = (
                    (self.matched_data["ra_weighted"] - self.matched_data["ra_gaia"])
                    * 3600
                    * np.cos(np.radians(self.matched_data["dec_gaia"]))
                )
                self.matched_data["dec_residual"] = (
                    self.matched_data["dec_weighted"] - self.matched_data["dec_gaia"]
                ) * 3600
                self.matched_data["total_residual"] = np.sqrt(
                    self.matched_data["ra_residual"] ** 2
                    + self.matched_data["dec_residual"] ** 2
                )
                logger.info(
                    "  2nd-pass colour/mag correction committed: "
                    "RMS %.3f → %.3f mas (LOO-CV %.3f → %.3f mas)",
                    rms_before * 1000,
                    float(
                        np.sqrt(
                            np.mean(
                                self.matched_data["ra_residual"].values ** 2
                                + self.matched_data["dec_residual"].values ** 2
                            )
                        )
                    )
                    * 1000,
                    rms_before * 1000,
                    rms_after_cv * 1000,
                )
                self._calculate_advanced_rms(
                    "ra_weighted", "dec_weighted", stage_name="Colour/mag 2nd-pass"
                )
            except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
                logger.warning("  2nd-pass colour/mag correction failed: %s", exc)
        else:
            logger.info(
                "  2nd-pass colour/mag correction skipped (no CV improvement: "
                "%.3f → %.3f mas)",
                rms_before * 1000,
                rms_after_cv * 1000,
            )

    def _tps_calibration(self):
        """
        Models spatially-correlated residuals using TPS or Gaussian Process,
        selecting the method with the lowest leave-one-out CV RMS.
        """
        X_feats = self.matched_data[["x_pixel", "y_pixel"]].values.astype(float)
        ra_res = self.matched_data["ra_residual"].values.astype(float)
        dec_res = self.matched_data["dec_residual"].values.astype(float)

        try:
            rms_before = np.sqrt(np.mean(ra_res**2 + dec_res**2))
            logger.info(
                "  TPS pre-fit residual RMS = %.4f arcsec (%.2f mas)",
                rms_before,
                rms_before * 1000.0,
            )

            # Leave-one-out CV via closed-form hat-matrix formula.
            # The Backend dispatcher routes to CuPy when available or
            # falls back to the CPU closed-form; a per-fold re-fit loop
            # is kept as a last-resort fallback for near-singular
            # matrices.
            n_samples = len(X_feats)
            smoothing_candidates = np.logspace(-5, 2, 25)

            try:
                backend = Backend.instance()
            except Exception:
                backend = None

            def loo_rms(smoothing, res_vals):
                if backend is not None and n_samples >= 5:
                    try:
                        r_loo = backend.tps_loocv(
                            X_feats, res_vals, smoothing=smoothing, degree=1
                        )
                        return float(np.sqrt(np.mean(r_loo**2)))
                    except Exception:
                        pass
                # Per-fold re-fit fallback
                errs = np.empty(n_samples)
                for i in range(n_samples):
                    test_mask = np.zeros(n_samples, dtype=bool)
                    test_mask[i] = True
                    train_mask = ~test_mask
                    if np.sum(train_mask) < 10:
                        errs[i] = res_vals[i]
                        continue
                    try:
                        tps = RBFInterpolator(
                            X_feats[train_mask],
                            res_vals[train_mask],
                            kernel="thin_plate_spline",
                            smoothing=smoothing,
                            degree=1,
                        )
                        errs[i] = res_vals[i] - tps(X_feats[test_mask])[0]
                    except (ValueError, TypeError, RuntimeError):
                        errs[i] = res_vals[i]
                return np.sqrt(np.mean(errs**2))

            best_s, best_cv = None, np.inf
            for s_val in smoothing_candidates:
                cv_ra = loo_rms(s_val, ra_res)
                cv_dec = loo_rms(s_val, dec_res)
                cv_tot = np.sqrt(cv_ra**2 + cv_dec**2)
                if cv_tot < best_cv:
                    best_cv, best_s = cv_tot, s_val
            logger.info(
                "  TPS LOO-CV: best smoothing=%.4g, CV RMS=%.4f arcsec (%.2f mas)",
                best_s if best_s is not None else float("nan"),
                best_cv,
                best_cv * 1000.0,
            )

            tps_helps = best_cv < rms_before * 0.95  # require >= 5% LOO-CV improvement

            # Gaussian Process alternative (Matern 3/2 with explicit noise)
            gp_cv = np.inf
            gp_helps = False
            if config.enable_gaussian_process:
                gp_cv, gp_helps = self._gp_loo_cv(ra_res, dec_res, X_feats)
                logger.info(
                    "  GP  LOO-CV: CV RMS=%.4f arcsec (%.2f mas) | helps=%s",
                    gp_cv,
                    gp_cv * 1000.0,
                    gp_helps,
                )

            # Pick the best spatial model
            if tps_helps and (not gp_helps or best_cv <= gp_cv):
                tps_ra = RBFInterpolator(
                    X_feats,
                    ra_res,
                    kernel="thin_plate_spline",
                    smoothing=best_s,
                    degree=1,
                )
                tps_dec = RBFInterpolator(
                    X_feats,
                    dec_res,
                    kernel="thin_plate_spline",
                    smoothing=best_s,
                    degree=1,
                )
                ra_corr, dec_corr = tps_ra(X_feats), tps_dec(X_feats)
                self.loo_cv_rms_arcsec = best_cv
                logger.info(
                    "  Selected: TPS (smoothing=%.4g, LOO-CV=%.2f mas)",
                    best_s,
                    best_cv * 1000.0,
                )
            elif gp_helps:
                ra_corr, dec_corr = self._gp_predict(ra_res, dec_res, X_feats)
                self.loo_cv_rms_arcsec = gp_cv
                logger.info(
                    "  Selected: Gaussian Process (LOO-CV=%.2f mas)", gp_cv * 1000.0
                )
            else:
                self.matched_data["ra_final"] = self.matched_data["ra_weighted"]
                self.matched_data["dec_final"] = self.matched_data["dec_weighted"]
                self._calculate_advanced_rms(
                    "ra_final", "dec_final", stage_name="TPS/GP (no correction)"
                )
                self.loo_cv_rms_arcsec = rms_before
                logger.info(
                    "  Selected: NO spatial correction (residuals uncorrelated, "
                    "LOO-CV=%.2f mas vs pre-fit=%.2f mas)",
                    min(best_cv, gp_cv) * 1000.0,
                    rms_before * 1000.0,
                )
                return

            self.matched_data["ra_final"] = self.matched_data[
                "ra_weighted"
            ] - ra_corr / 3600 / np.cos(np.radians(self.matched_data["dec_weighted"]))
            self.matched_data["dec_final"] = (
                self.matched_data["dec_weighted"] - dec_corr / 3600
            )
            self._calculate_advanced_rms(
                "ra_final", "dec_final", stage_name="TPS/GP spatial"
            )
            if self.loo_cv_rms_arcsec is None:
                self.loo_cv_rms_arcsec = (
                    self.rms_history[-1] if self.rms_history else None
                )
        except (ValueError, TypeError, RuntimeError):
            self.matched_data["ra_final"] = self.matched_data["ra_weighted"]
            self.matched_data["dec_final"] = self.matched_data["dec_weighted"]
            self._calculate_advanced_rms(
                "ra_final", "dec_final", stage_name="TPS/GP (error fallback)"
            )
            if self.loo_cv_rms_arcsec is None:
                self.loo_cv_rms_arcsec = (
                    self.rms_history[-1] if self.rms_history else None
                )

    def _gp_loo_cv(self, ra_res, dec_res, X_feats):
        """Leave-one-out CV using a Matern 3/2 Gaussian Process with
        white-noise kernel via scikit-learn."""
        import warnings

        try:
            from sklearn.exceptions import ConvergenceWarning
        except ImportError:
            ConvergenceWarning = None
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import (
                ConstantKernel as C,
            )
            from sklearn.gaussian_process.kernels import (
                Matern,
                WhiteKernel,
            )
        except ImportError:
            return np.inf, False

        n_samples = len(X_feats)
        if n_samples < 20:
            return np.inf, False

        # Length scale ~ image diagonal / 5 (~150 px for 1364 px image)
        length_scale = config.gp_length_scale_pixels
        # Bounds allow optimisation down to sub-10 px for tree-ring scales
        kernel = C(1.0, (1e-3, 1e1)) * Matern(
            length_scale=length_scale, length_scale_bounds=(5.0, 2000.0), nu=1.5
        ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e2))

        def loo_residuals(res_vals):
            errs = np.empty(n_samples)
            for i in range(n_samples):
                test_mask = np.zeros(n_samples, dtype=bool)
                test_mask[i] = True
                train_mask = ~test_mask
                if train_mask.sum() < 10:
                    errs[i] = res_vals[i]
                    continue
                try:
                    # Suppress sklearn ConvergenceWarning when GP
                    # hyperparameter optimisation hits kernel bounds
                    with warnings.catch_warnings():
                        if ConvergenceWarning is not None:
                            warnings.simplefilter("ignore", category=ConvergenceWarning)
                        gp = GaussianProcessRegressor(
                            kernel=kernel,
                            alpha=1e-6,
                            normalize_y=True,
                            n_restarts_optimizer=0,
                        )
                        gp.fit(X_feats[train_mask], res_vals[train_mask])
                        errs[i] = res_vals[i] - gp.predict(X_feats[test_mask])[0]
                except (ValueError, TypeError, RuntimeError):
                    errs[i] = res_vals[i]
            return np.sqrt(np.mean(errs**2))

        cv_ra = loo_residuals(ra_res)
        cv_dec = loo_residuals(dec_res)
        cv_tot = np.sqrt(cv_ra**2 + cv_dec**2)
        rms_before = np.sqrt(np.mean(ra_res**2 + dec_res**2))
        # Require >= 5% LOO-CV improvement to select GP over no correction
        return cv_tot, cv_tot < rms_before * 0.95

    def _gp_predict(self, ra_res, dec_res, X_feats):
        """Fits a Matern 3/2 GP on the full sample and returns
        in-sample predictions for smooth residual removal."""
        import warnings

        try:
            from sklearn.exceptions import ConvergenceWarning
        except ImportError:
            ConvergenceWarning = None
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            ConstantKernel as C,
        )
        from sklearn.gaussian_process.kernels import (
            Matern,
            WhiteKernel,
        )

        length_scale = config.gp_length_scale_pixels
        kernel = C(1.0, (1e-3, 1e1)) * Matern(
            length_scale=length_scale, length_scale_bounds=(5.0, 2000.0), nu=1.5
        ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e2))
        with warnings.catch_warnings():
            if ConvergenceWarning is not None:
                warnings.simplefilter("ignore", category=ConvergenceWarning)
            gp_ra = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=0
            )
            gp_dec = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=0
            )
            gp_ra.fit(X_feats, ra_res)
            gp_dec.fit(X_feats, dec_res)
        return gp_ra.predict(X_feats), gp_dec.predict(X_feats)

    def _apply_quasar_frame_tie(self):
        """Applies a median-offset frame tie to ICRF via Milliquas quasars."""
        if self.milliquas_catalog is None or len(self.milliquas_catalog["ra"]) == 0:
            self.matched_data["ra_icrf"] = self.matched_data["ra_final"]
            self.matched_data["dec_icrf"] = self.matched_data["dec_final"]
            return

        meas_ra = self.matched_data["ra_final"].values
        meas_dec = self.matched_data["dec_final"].values
        meas_coords = SkyCoord(ra=meas_ra * u.deg, dec=meas_dec * u.deg)
        qso_coords = SkyCoord(
            ra=self.milliquas_catalog["ra"] * u.deg,
            dec=self.milliquas_catalog["dec"] * u.deg,
        )
        idx_fwd, d2d_fwd, _ = meas_coords.match_to_catalog_sky(qso_coords)
        idx_rev, _, _ = qso_coords.match_to_catalog_sky(meas_coords)

        symmetric_mask = np.zeros(len(meas_coords), dtype=bool)
        for i in range(len(meas_coords)):
            if idx_rev[idx_fwd[i]] == i and d2d_fwd[i] <= 2.0 * u.arcsec:
                symmetric_mask[i] = True

        if np.sum(symmetric_mask) < 3:
            self.matched_data["ra_icrf"] = self.matched_data["ra_final"]
            self.matched_data["dec_icrf"] = self.matched_data["dec_final"]
            return

        qso_idx = idx_fwd[symmetric_mask]
        matched = pd.DataFrame(
            {
                "ra_measured": meas_ra[symmetric_mask],
                "dec_measured": meas_dec[symmetric_mask],
                "ra_qso": self.milliquas_catalog["ra"][qso_idx],
                "dec_qso": self.milliquas_catalog["dec"][qso_idx],
            }
        )
        matched["dra_arcsec"] = (
            (matched["ra_measured"] - matched["ra_qso"])
            * 3600
            * np.cos(np.radians(matched["dec_qso"]))
        )
        matched["ddec_arcsec"] = (matched["dec_measured"] - matched["dec_qso"]) * 3600

        dra_tie, ddec_tie = (
            float(np.median(matched["dra_arcsec"])),
            float(np.median(matched["ddec_arcsec"])),
        )
        self.matched_data["ra_icrf"] = self.matched_data[
            "ra_final"
        ] - dra_tie / 3600.0 / np.cos(np.radians(self.matched_data["dec_final"]))
        self.matched_data["dec_icrf"] = (
            self.matched_data["dec_final"] - ddec_tie / 3600.0
        )

    def _calculate_advanced_rms(self, ra_col, dec_col, stage_name=None):
        """Computes RMS from specified coordinate columns and appends
        to the RMS history."""
        ra_diff = (
            (self.matched_data[ra_col] - self.matched_data["ra_gaia"])
            * 3600
            * np.cos(np.radians(self.matched_data["dec_gaia"]))
        )
        dec_diff = (self.matched_data[dec_col] - self.matched_data["dec_gaia"]) * 3600
        rms = np.sqrt(np.mean(ra_diff**2 + dec_diff**2))
        self.rms_history.append(rms)
        if stage_name is None:
            stage_name = f"Stage {len(self.rms_history)}"
        self.rms_stage_names.append(stage_name)

    def _save_advanced_results(self):
        """Saves the advanced calibration results and writes a report."""
        self.matched_data.to_csv(
            self.catalogs_dir / "advanced_calibrated_coordinates.csv", index=False
        )
        with open(
            self.logs_dir / "advanced_calibration_report.txt", "w", encoding="utf-8"
        ) as f:
            f.write("ADVANCED ASTROMETRIC CALIBRATION REPORT\n")
            f.write("=" * 55 + "\n\n")
            f.write(f"Image shape: {self.image_shape}\n")
            f.write(f"Estimated FWHM: {self.estimated_fwhm:.3f} px\n")
            f.write(f"Estimated airmass: {self.airmass:.4f}\n")
            f.write(f"Stars matched (advanced): {len(self.matched_data)}\n\n")
            f.write("RMS EVOLUTION:\n")
            for i, rms in enumerate(self.rms_history):
                stage = (
                    self.rms_stage_names[i]
                    if i < len(self.rms_stage_names)
                    else f"Stage {i + 1}"
                )
                f.write(
                    f"  Stage {i + 1}: {rms:.6f} arcsec ({rms * 1000:.3f} mas) — {stage}\n"
                )
            f.write("\n")
            f.write("FINAL RESIDUAL STATISTICS:\n")
            f.write("  (in-sample RMS — computed against the same Gaia stars\n")
            f.write("   used to fit the calibration model. May be biased low\n")
            f.write("   if the spatial model overfits; see LOO-CV below.)\n")
            ra_res = (
                (self.matched_data["ra_final"] - self.matched_data["ra_gaia"])
                * 3600e3
                * np.cos(np.radians(self.matched_data["dec_gaia"]))
            )
            dec_res = (
                self.matched_data["dec_final"] - self.matched_data["dec_gaia"]
            ) * 3600e3
            sep = np.hypot(ra_res, dec_res)
            f.write(
                f"  RA  RMS: {np.sqrt(np.mean(ra_res**2)):.3f} mas | "
                f"median |dra|: {np.median(np.abs(ra_res)):.3f} mas\n"
            )
            f.write(
                f"  Dec RMS: {np.sqrt(np.mean(dec_res**2)):.3f} mas | "
                f"median |ddec|: {np.median(np.abs(dec_res)):.3f} mas\n"
            )
            f.write(f"  Total RMS (in-sample): {np.sqrt(np.mean(sep**2)):.3f} mas\n")
            f.write(f"  Median sep: {np.median(sep):.3f} mas\n")
            f.write(f"  95th pct sep: {np.percentile(sep, 95):.3f} mas\n")
            f.write(f"  Max sep: {np.max(sep):.3f} mas\n\n")
            if self.loo_cv_rms_arcsec is not None:
                f.write("PREDICTIVE PRECISION (leave-one-out CV):\n")
                f.write("  (unbiased estimate of how well the calibration would\n")
                f.write("   generalise to a NEW star at a random position)\n")
                f.write(f"  LOO-CV RMS: {self.loo_cv_rms_arcsec * 1000:.3f} mas\n\n")
            if "sigma_centroid_mas" in self.matched_data.columns:
                f.write(
                    f"  Median σ_centroid (SNR model): "
                    f"{np.median(self.matched_data['sigma_centroid_mas']):.3f} mas\n"
                )
            f.write(f"\n  Airmass (estimated): {self.airmass:.4f}\n")
