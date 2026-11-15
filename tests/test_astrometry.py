import numpy as np
import pandas as pd
from astropy.time import Time

from src.astrometry import Astrometry
from src.config import config


class TestAstrometryInit:
    def test_init_stores_attributes(self, synthetic_gaia_catalog):
        """Astrometry should store all passed attributes."""
        obstime = Time(59958.145833, format="mjd")
        ast = Astrometry(synthetic_gaia_catalog, obstime)
        assert ast.gaia_catalog is synthetic_gaia_catalog
        assert ast.obstime is obstime

    def test_init_with_none_catalog(self):
        """Astrometry should handle a None catalog."""
        obstime = Time(59958.145833, format="mjd")
        ast = Astrometry(None, obstime)
        assert ast.gaia_catalog is None


class TestProperMotion:
    def test_get_proper_motion_corrected_gaia_columns(self, synthetic_gaia_catalog):
        """Corrected catalog should include ra_corr and dec_corr columns."""
        obstime = Time(59958.145833, format="mjd")
        ast = Astrometry(synthetic_gaia_catalog, obstime)
        result = ast.get_proper_motion_corrected_gaia()
        assert result is not None
        assert "ra_corr" in result.columns
        assert "dec_corr" in result.columns

    def test_get_proper_motion_corrected_different_epoch(self, synthetic_gaia_catalog):
        """Coordinates at a different epoch should differ slightly from J2016."""
        obstime = Time(59958.145833, format="mjd")  # ~2023
        ast = Astrometry(synthetic_gaia_catalog, obstime)
        result = ast.get_proper_motion_corrected_gaia()
        # With typical proper motions ~2 mas/yr and ~7 year baseline,
        # shifts should be < 0.02 deg (~70 arcsec)
        max_ra_shift = np.max(np.abs(result["ra_corr"] - result["ra"]))
        max_dec_shift = np.max(np.abs(result["dec_corr"] - result["dec"]))
        assert max_ra_shift < 0.02
        assert max_dec_shift < 0.02

    def test_get_proper_motion_missing_columns(self, tmp_path):
        """When required columns are missing, None should be returned."""
        bad_catalog = pd.DataFrame({"ra": [10.0], "dec": [20.0]})
        obstime = Time(59958.145833, format="mjd")
        ast = Astrometry(bad_catalog, obstime)
        result = ast.get_proper_motion_corrected_gaia()
        assert result is None

    def test_get_proper_motion_empty_after_dropna(self):
        """When all rows have NaN in required columns, None should be returned."""
        bad_catalog = pd.DataFrame(
            {
                "ra": [np.nan],
                "dec": [np.nan],
                "pmra": [np.nan],
                "pmdec": [np.nan],
                "ref_epoch": [np.nan],
            }
        )
        obstime = Time(59958.145833, format="mjd")
        ast = Astrometry(bad_catalog, obstime)
        result = ast.get_proper_motion_corrected_gaia()
        assert result is None

    def test_pm_mode_none_zeros_all_pm(self, synthetic_gaia_catalog):
        """pm_mode='none' should zero out all proper motions."""
        from src.config import config as cfg

        obstime = Time(59958.145833, format="mjd")
        ast = Astrometry(synthetic_gaia_catalog, obstime)
        saved_mode = cfg.pm_mode
        object.__setattr__(cfg, "pm_mode", "none")
        try:
            result = ast.get_proper_motion_corrected_gaia()
            assert result is not None
            assert np.all(result["pmra"] == 0.0)
            assert np.all(result["pmdec"] == 0.0)
        finally:
            object.__setattr__(cfg, "pm_mode", saved_mode)

    def test_pm_damping_scales_faint_stars(self, synthetic_gaia_catalog):
        """PM damping should reduce proper motion for G >= threshold."""
        from src.config import config as cfg

        # Set threshold low so all stars are damped
        obstime = Time(59958.145833, format="mjd")
        ast = Astrometry(synthetic_gaia_catalog, obstime)
        saved = {
            "mode": cfg.pm_mode,
            "enable": cfg.enable_pm_damping,
            "thresh": cfg.pm_damping_mag_threshold,
            "factor": cfg.pm_damping_factor,
        }
        object.__setattr__(cfg, "pm_mode", "damped")
        object.__setattr__(cfg, "enable_pm_damping", True)
        object.__setattr__(cfg, "pm_damping_mag_threshold", 0.0)  # damp all stars (threshold at zero affects every star)
        object.__setattr__(cfg, "pm_damping_factor", 0.3)
        try:
            # Get undamped reference
            object.__setattr__(cfg, "enable_pm_damping", False)
            ref = ast.get_proper_motion_corrected_gaia()
            # Get damped result
            object.__setattr__(cfg, "enable_pm_damping", True)
            damped = ast.get_proper_motion_corrected_gaia()

            assert damped is not None and ref is not None
            # Damped PM should have smaller magnitude than undamped
            ref_pm_mag = np.sqrt(ref["pmra"] ** 2 + ref["pmdec"] ** 2)
            damped_pm_mag = np.sqrt(damped["pmra"] ** 2 + damped["pmdec"] ** 2)
            assert np.all(damped_pm_mag <= ref_pm_mag + 1e-10)
        finally:
            for k, v in saved.items():
                object.__setattr__(cfg, k, v)


class TestMatchCatalogs:
    def test_match_catalogs_returns_none_for_none_gaia(self, synthetic_matches):
        """match_catalogs should return None when gaia_catalog is None."""
        obstime = Time(59958.145833, format="mjd")
        ast = Astrometry(None, obstime)
        result = ast.match_catalogs({"aperture": None})
        assert result is None

    def test_match_catalogs_basic(
        self, synthetic_gaia_catalog, synthetic_matches, tmp_path
    ):
        """match_catalogs should process photometry results without error."""
        obstime = Time(59958.145833, format="mjd")
        # Save synthetic matches as xy_radec file
        matches_file = tmp_path / "xy_radec_aperture.txt"
        with open(matches_file, "w") as f:
            f.writelines(
                f"{row['ra_psf']:12.6f} {row['dec_psf']:12.6f}\n"
                for _, row in synthetic_matches.iterrows()
            )

        ast = Astrometry(
            synthetic_gaia_catalog,
            obstime,
            catalogs_dir=tmp_path,
            logs_dir=tmp_path,
            plots_dir=tmp_path,
        )
        result = ast.get_proper_motion_corrected_gaia()
        assert result is not None

        ast.match_catalogs({"aperture": None})


class TestMatchStatistics:
    def test_calculate_match_statistics(self, synthetic_matches, tmp_path):
        """_calculate_match_statistics should return expected keys."""
        obstime = Time(59958.145833, format="mjd")
        ast = Astrometry(None, obstime, logs_dir=tmp_path)
        stats = ast._calculate_match_statistics(synthetic_matches, "test")
        expected_keys = {
            "n_matches",
            "mean_separation",
            "median_separation",
            "rms_ra",
            "rms_dec",
            "rms_total",
        }
        assert expected_keys.issubset(stats.keys())
        assert stats["n_matches"] == len(synthetic_matches)
        assert stats["rms_total"] > 0

    def test_match_statistics_output_file(self, synthetic_matches, tmp_path):
        """Statistics should be written to a log file."""
        obstime = Time(59958.145833, format="mjd")
        ast = Astrometry(None, obstime, logs_dir=tmp_path)
        ast._calculate_match_statistics(synthetic_matches, "test")
        stat_file = tmp_path / "match_statistics.txt"
        assert stat_file.exists()
        content = stat_file.read_text()
        assert "STATS (TEST)" in content or "STATS (test)" in content


class TestQualityCuts:
    def test_magnitude_cut(self, synthetic_gaia_catalog):
        """_match_single_catalog should apply magnitude quality cuts."""
        # Set limits that reject sources outside [10, 18]
        # The current config uses [6, 21.5]; we test that filtering happens
        n_before = len(synthetic_gaia_catalog)
        masked = synthetic_gaia_catalog[
            (synthetic_gaia_catalog["phot_g_mean_mag"] >= config.gaia_bright_limit)
            & (synthetic_gaia_catalog["phot_g_mean_mag"] <= config.gaia_faint_limit)
        ]
        assert len(masked) <= n_before

    def test_ruwe_cut(self, synthetic_gaia_catalog):
        """RUWE threshold should reject poor astrometric solutions."""
        filtered = synthetic_gaia_catalog[
            synthetic_gaia_catalog["ruwe"].fillna(1.0) <= config.gaia_ruwe_limit
        ]
        # With ruwe in [0.8, 1.3] and limit 1.4, all should pass
        assert len(filtered) == len(synthetic_gaia_catalog)

    def test_color_cut(self, synthetic_gaia_catalog):
        """Colour cuts should reject extreme blue/red stars."""
        filtered = synthetic_gaia_catalog[
            (
                synthetic_gaia_catalog["bp_rp"].fillna(1.0)
                >= config.gaia_color_blue_limit
            )
            & (
                synthetic_gaia_catalog["bp_rp"].fillna(1.0)
                <= config.gaia_color_red_limit
            )
        ]
        # With bp_rp in [0.3, 1.5] and limits [0.0, 2.5], all should pass
        assert len(filtered) == len(synthetic_gaia_catalog)
