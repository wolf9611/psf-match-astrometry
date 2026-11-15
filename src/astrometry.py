"""
Astrometric matching and proper motion propagation.
"""

import warnings

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import ICRS, SkyCoord
from astropy.time import Time

try:
    from erfa import ErfaWarning
except ImportError:
    from astropy.utils.exceptions import ErfaWarning
from src.config import CATALOGS_DIR, LOGS_DIR, config
from src.logger import get_logger
from src.plotting import plot_match_results

logger = get_logger("astrometry")


class Astrometry:
    """
    Handles coordinate propagation and cross-matching with reference catalogs.
    """

    def __init__(
        self,
        gaia_catalog,
        obstime,
        catalogs_dir=CATALOGS_DIR,
        logs_dir=LOGS_DIR,
        plots_dir=None,
    ):
        self.gaia_catalog = gaia_catalog
        self.obstime = obstime
        self.catalogs_dir = catalogs_dir
        self.logs_dir = logs_dir
        self.plots_dir = plots_dir

    def match_catalogs(self, photometry_results, max_sep=2.0):
        """
        Cross-matches each photometry result against the reference Gaia catalog.
        """
        if self.gaia_catalog is None:
            return

        gaia_corrected = self.get_proper_motion_corrected_gaia()
        if gaia_corrected is None:
            return

        for method in photometry_results.keys():
            self._match_single_catalog(gaia_corrected, method, max_sep)

    def get_proper_motion_corrected_gaia(self):
        """
        Propagates Gaia coordinates to the observation epoch using proper motions.

        2-parameter sources (null pmra/pmdec) are retained with zero
        proper motion — the epoch propagation becomes a no-op for those
        stars, which is the standard treatment for Gaia 2-parameter
        solutions.

        PM damping: when enabled, scales down proper motions for stars
        with G >= pm_damping_mag_threshold, using the per-star PM
        uncertainty when available (pmra_error) for an adaptive factor.
        """
        gaia_df = pd.DataFrame(self.gaia_catalog)
        required = ["ra", "dec", "pmra", "pmdec", "ref_epoch"]
        if not all(col in gaia_df.columns for col in required):
            return None

        # When include_faint_stars is False, drop 2-parameter sources
        # (null pmra/pmdec) entirely.  When True, retain them with zero
        # proper motion — the epoch propagation becomes a no-op, which
        # is the standard treatment for Gaia 2-parameter solutions.
        if config.include_faint_stars:
            gaia_clean = gaia_df.copy()
            gaia_clean["pmra"] = gaia_clean["pmra"].fillna(0.0)
            gaia_clean["pmdec"] = gaia_clean["pmdec"].fillna(0.0)
        else:
            gaia_clean = gaia_df.copy()
        gaia_clean = gaia_clean.dropna(subset=required).copy()
        if gaia_clean.empty:
            return None

        # --- PM mode handling ---
        if config.pm_mode == "none":
            # Zero out all proper motions: use Gaia epoch positions only
            gaia_clean["pmra"] = 0.0
            gaia_clean["pmdec"] = 0.0
            logger.info(
                "PM mode = none: all proper motions zeroed (epoch positions only)."
            )
        elif config.enable_pm_damping and "phot_g_mean_mag" in gaia_clean.columns:
            damp_mask = gaia_clean["phot_g_mean_mag"] >= config.pm_damping_mag_threshold
            n_damped = int(damp_mask.sum())
            if n_damped > 0:
                if "pmra_error" in gaia_clean.columns:
                    # Adaptive per-star damping based on PM SNR
                    pmra = gaia_clean["pmra"].values
                    pmra_err = gaia_clean["pmra_error"].values
                    pmra_err_safe = np.where(
                        (pmra_err is not None) & (pmra_err > 0) & (~np.isnan(pmra_err)),
                        pmra_err,
                        1e-6,
                    )
                    pm_snr = np.abs(pmra / pmra_err_safe)
                    # Per-star factor: 1.0 for high SNR, damping_factor for low SNR
                    per_star_factor = np.clip(
                        pm_snr / 3.0, config.pm_damping_factor, 1.0
                    )
                    per_star_factor = np.where(damp_mask.values, per_star_factor, 1.0)
                    gaia_clean["pmra"] = gaia_clean["pmra"] * per_star_factor
                    gaia_clean["pmdec"] = gaia_clean["pmdec"] * per_star_factor
                    median_factor = float(np.median(per_star_factor[damp_mask]))
                    logger.info(
                        "PM damping: %d stars G>=%.1f, median scale=%.3f "
                        "(adaptive from PM SNR)",
                        n_damped,
                        config.pm_damping_mag_threshold,
                        median_factor,
                    )
                else:
                    # Uniform damping for all stars above threshold
                    gaia_clean.loc[damp_mask, "pmra"] *= config.pm_damping_factor
                    gaia_clean.loc[damp_mask, "pmdec"] *= config.pm_damping_factor
                    logger.info(
                        "PM damping: %d stars G>=%.1f, uniform scale=%.3f",
                        n_damped,
                        config.pm_damping_mag_threshold,
                        config.pm_damping_factor,
                    )

        try:
            coords_ref = SkyCoord(
                ra=gaia_clean["ra"].values * u.deg,
                dec=gaia_clean["dec"].values * u.deg,
                pm_ra_cosdec=gaia_clean["pmra"].values * u.mas / u.yr,
                pm_dec=gaia_clean["pmdec"].values * u.mas / u.yr,
                obstime=Time(gaia_clean["ref_epoch"].values, format="jyear"),
                frame=ICRS(),
            )
            # Suppress ERFA warnings for Gaia 2-parameter sources missing
            # parallax; the propagation remains valid
            # informational.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ErfaWarning)
                coords_obs = coords_ref.apply_space_motion(new_obstime=self.obstime)
            gaia_clean["ra_corr"] = coords_obs.ra.deg
            gaia_clean["dec_corr"] = coords_obs.dec.deg
            return gaia_clean
        except (ValueError, TypeError, AttributeError) as exc:
            logger.error("Proper motion correction failed: %s", exc)
            print(f"Error in proper motion correction: {exc}")
            return None

    def _match_single_catalog(self, gaia_corrected, method, max_sep):
        xy_file = self.catalogs_dir / f"xy_radec_{method}.txt"
        if not xy_file.exists():
            return

        try:
            xy_df = pd.read_csv(
                xy_file,
                sep=r"\s+",
                header=None,
                names=["ra_psf", "dec_psf"],
                engine="python",
            )

            if "phot_g_mean_mag" in gaia_corrected.columns:
                g_mag = gaia_corrected["phot_g_mean_mag"]
                n_saturated = int((g_mag < config.gaia_saturated_mag_limit).sum())
                if n_saturated > 0:
                    logger.info(
                        "  [%s] %d saturated stars (G<%.1f) excluded",
                        method.upper(),
                        n_saturated,
                        config.gaia_saturated_mag_limit,
                    )
                n_too_faint = int((g_mag > config.gaia_faint_limit).sum())
                if n_too_faint > 0:
                    logger.info(
                        "  [%s] %d stars beyond faint limit (G>%.1f)",
                        method.upper(),
                        n_too_faint,
                        config.gaia_faint_limit,
                    )
                in_range = (g_mag >= config.gaia_bright_limit) & (
                    g_mag <= config.gaia_faint_limit
                )
                # Reject Gaia sources with RUWE > threshold (likely astrometric
                # binaries or with poor single-star fits)  --  these contribute
                # spuriously large reference-frame noise.
                if "ruwe" in gaia_corrected.columns:
                    ruwe_raw = gaia_corrected["ruwe"].fillna(1.0)
                    if (
                        config.include_faint_stars
                        and "phot_g_mean_mag" in gaia_corrected.columns
                    ):
                        g_for_ruwe = gaia_corrected["phot_g_mean_mag"].fillna(0)
                        per_star_limit = np.where(
                            g_for_ruwe < config.gaia_ruwe_transition_mag,
                            config.gaia_ruwe_limit,
                            config.gaia_ruwe_faint_limit,
                        )
                        ruwe_mask = ruwe_raw <= per_star_limit
                        n_bright_reject = int(
                            (
                                (g_for_ruwe < config.gaia_ruwe_transition_mag)
                                & (ruwe_raw > config.gaia_ruwe_limit)
                            ).sum()
                        )
                        n_faint_reject = int(
                            (
                                (g_for_ruwe >= config.gaia_ruwe_transition_mag)
                                & (ruwe_raw > config.gaia_ruwe_faint_limit)
                            ).sum()
                        )
                        if n_bright_reject + n_faint_reject > 0:
                            logger.info(
                                "  [%s] RUWE rejection: %d bright (G<%.1f, RUWE>%.1f), "
                                "%d faint (G>=%.1f, RUWE>%.1f)",
                                method.upper(),
                                n_bright_reject,
                                config.gaia_ruwe_transition_mag,
                                config.gaia_ruwe_limit,
                                n_faint_reject,
                                config.gaia_ruwe_transition_mag,
                                config.gaia_ruwe_faint_limit,
                            )
                    else:
                        ruwe_mask = ruwe_raw <= config.gaia_ruwe_limit
                        n_bright_reject = int((~ruwe_mask).sum())
                        if n_bright_reject > 0:
                            logger.info(
                                "  [%s] rejecting %d Gaia sources with RUWE > %.1f",
                                method.upper(),
                                n_bright_reject,
                                config.gaia_ruwe_limit,
                            )
                    in_range = in_range & ruwe_mask
                # Reject stars outside the colour range (very blue or very red
                # stars have poorly-modelled DCR and larger astrometric noise).
                if "bp_rp" in gaia_corrected.columns:
                    color_mask = (
                        gaia_corrected["bp_rp"].fillna(1.0)
                        >= config.gaia_color_blue_limit
                    ) & (
                        gaia_corrected["bp_rp"].fillna(1.0)
                        <= config.gaia_color_red_limit
                    )
                    n_color_reject = int((~color_mask).sum())
                    if n_color_reject > 0:
                        logger.info(
                            "  [%s] rejecting %d Gaia sources outside colour [%.1f, %.1f]",
                            method.upper(),
                            n_color_reject,
                            config.gaia_color_blue_limit,
                            config.gaia_color_red_limit,
                        )
                    in_range = in_range & color_mask
                gaia_corrected = gaia_corrected[in_range].reset_index(drop=True)

            gaia_coords = SkyCoord(
                ra=gaia_corrected["ra_corr"].values * u.deg,
                dec=gaia_corrected["dec_corr"].values * u.deg,
            )
            xy_coords = SkyCoord(
                ra=xy_df["ra_psf"].values * u.deg, dec=xy_df["dec_psf"].values * u.deg
            )

            idx_fwd, d2d_fwd, _ = xy_coords.match_to_catalog_sky(gaia_coords)
            idx_rev, _, _ = gaia_coords.match_to_catalog_sky(xy_coords)

            symmetric_mask = np.zeros(len(xy_coords), dtype=bool)
            for i in range(len(xy_coords)):
                j = idx_fwd[i]
                if idx_rev[j] == i and d2d_fwd[i] <= max_sep * u.arcsec:
                    symmetric_mask[i] = True

            mask = symmetric_mask
            if np.sum(mask) == 0:
                return

            gaia_matched = gaia_corrected.iloc[idx_fwd[mask]]
            bp_rp_col = (
                gaia_matched["bp_rp"].values
                if "bp_rp" in gaia_matched.columns
                else np.full(mask.sum(), np.nan)
            )
            ruwe_col = (
                gaia_matched["ruwe"].values
                if "ruwe" in gaia_matched.columns
                else np.full(mask.sum(), 1.0)
            )
            matches = pd.DataFrame(
                {
                    "ra_gaia_corr": gaia_matched["ra_corr"].values,
                    "dec_gaia_corr": gaia_matched["dec_corr"].values,
                    "ra_psf": xy_df.loc[mask, "ra_psf"].values,
                    "dec_psf": xy_df.loc[mask, "dec_psf"].values,
                    "separation_arcsec": d2d_fwd[mask].arcsec,
                    "phot_g_mean_mag": gaia_matched["phot_g_mean_mag"].values,
                    "bp_rp": bp_rp_col,
                    "ruwe": ruwe_col,
                }
            )

            matches["ra_diff_arcsec"] = (
                (matches["ra_psf"] - matches["ra_gaia_corr"])
                * 3600
                * np.cos(np.radians(matches["dec_gaia_corr"]))
            )
            matches["dec_diff_arcsec"] = (
                matches["dec_psf"] - matches["dec_gaia_corr"]
            ) * 3600

            stats = self._calculate_match_statistics(matches, method)
            matches.to_csv(
                self.catalogs_dir / f"matched_catalog_{method}.csv", index=False
            )
            plot_match_results(
                matches, method, stats["rms_total"], plots_dir=self.plots_dir
            )
        except (ValueError, pd.errors.EmptyDataError, KeyError) as exc:
            print(f"Error matching {method}: {exc}")

    def _calculate_match_statistics(self, matches, method):
        stats = {
            "n_matches": len(matches),
            "mean_separation": matches["separation_arcsec"].mean(),
            "median_separation": matches["separation_arcsec"].median(),
            "rms_ra": np.sqrt(np.mean(matches["ra_diff_arcsec"] ** 2)),
            "rms_dec": np.sqrt(np.mean(matches["dec_diff_arcsec"] ** 2)),
            "rms_total": np.sqrt(
                np.mean(matches["ra_diff_arcsec"] ** 2)
                + np.mean(matches["dec_diff_arcsec"] ** 2)
            ),
        }

        stat_file = self.logs_dir / "match_statistics.txt"
        mode = "a" if stat_file.exists() else "w"
        with open(stat_file, mode, encoding="utf-8") as f:
            f.write(f"\n{'=' * 15} STATS ({method.upper()}) {'=' * 15}\n")
            f.writelines(
                f"{key}: {value:.6f}\n"
                if isinstance(value, float)
                else f"{key}: {value}\n"
                for key, value in stats.items()
            )
        return stats
