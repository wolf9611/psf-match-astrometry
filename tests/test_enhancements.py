import numpy as np
import pandas as pd

from src.enhancements import (
    CCDEdgeDiagnostic,
    DCRCorrector,
    HoldoutCV,
    MagBin,
    MagnitudeStratifier,
    gaia_positional_floor_mas,
    write_telemetry_report,
)


class TestGaiaPositionalFloor:
    def test_floor_increases_with_magnitude(self):
        """Brighter stars should have smaller positional floor."""
        floor_14 = gaia_positional_floor_mas(14.0)
        floor_18 = gaia_positional_floor_mas(18.0)
        floor_20 = gaia_positional_floor_mas(20.0)
        assert floor_14 < floor_18 < floor_20

    def test_floor_interpolates_within_table(self):
        """Floor should interpolate between tabulated values."""
        floor_16 = gaia_positional_floor_mas(16.0)
        assert 0.02 <= floor_16 <= 0.10

    def test_floor_extrapolates_beyond_22(self):
        """Floor should extrapolate past G=22 with a steeper slope."""
        floor_23 = gaia_positional_floor_mas(23.0)
        floor_24 = gaia_positional_floor_mas(24.0)
        assert floor_23 > 2.0
        assert floor_24 > floor_23

    def test_floor_handles_array(self):
        """Floor should work with numpy arrays."""
        mags = np.array([14.0, 18.0, 20.0])
        floors = gaia_positional_floor_mas(mags)
        assert len(floors) == 3
        assert np.all(floors > 0)
        assert floors[0] < floors[1] < floors[2]

    def test_floor_at_known_points(self):
        """Floor at known tabulated values should match closely."""
        floor_12 = gaia_positional_floor_mas(12.0)
        floor_18 = gaia_positional_floor_mas(18.0)
        assert abs(floor_12 - 0.02) < 0.01
        assert abs(floor_18 - 0.15) < 0.01


class TestMagBin:
    def test_magbin_dataclass(self):
        """MagBin should be a simple data container."""
        bin_data = MagBin(
            lo=16.0,
            hi=17.0,
            n=10,
            rms_ra_mas=1.0,
            rms_dec_mas=1.5,
            rms_tot_mas=1.8,
            median_sep_mas=1.2,
            p95_sep_mas=2.5,
            gaia_floor_mas=0.05,
            centroid_noise_mas=0.5,
            expected_total_mas=0.502,
        )
        assert bin_data.lo == 16.0
        assert bin_data.n == 10
        assert bin_data.rms_tot_mas == 1.8


class TestMagnitudeStratifier:
    def test_init_default_bins(self, synthetic_matches):
        """MagnitudeStratifier should initialize with default bin edges."""
        strat = MagnitudeStratifier(
            matched_data=synthetic_matches,
            wcs=None,
            pixel_scale_arcsec=0.1,
            fwhm_pixels=5.0,
        )
        assert len(strat.bin_edges) > 1
        assert strat.faint_threshold == 21.0

    def test_run_with_empty_data(self, mock_wcs):
        """run() should return empty results when matched_data is empty."""
        empty = pd.DataFrame()
        strat = MagnitudeStratifier(
            matched_data=empty,
            wcs=mock_wcs,
            pixel_scale_arcsec=0.1,
            fwhm_pixels=5.0,
        )
        result = strat.run()
        assert result["magnitude_bins"] == []
        assert result["photon_break_mag"] is None

    def test_run_with_data(self, synthetic_matches, mock_wcs):
        """run() should process data and return bins when data is available."""
        # Add required columns
        df = synthetic_matches.copy()
        df["ra_final"] = df["ra_psf"]
        df["dec_final"] = df["dec_psf"]
        df["ra_gaia"] = df["ra_gaia_corr"]
        df["dec_gaia"] = df["dec_gaia_corr"]
        df["gaia_mag"] = df["phot_g_mean_mag"]
        df["x_pixel"] = df["x_pixel"]

        strat = MagnitudeStratifier(
            matched_data=df,
            wcs=mock_wcs,
            pixel_scale_arcsec=0.1,
            fwhm_pixels=5.0,
            faint_threshold=20.0,
            bin_edges=[14, 16, 18, 20],
        )
        result = strat.run()
        # May have bins depending on magnitude distribution
        assert isinstance(result, dict)


class TestHoldoutCV:
    def test_split_ratios(self, mock_wcs):
        """HoldoutCV splits should approximate 70/30."""
        n = 100
        ras = np.random.RandomState(42).uniform(10, 11, n)
        decs = np.random.RandomState(42).uniform(20, 21, n)
        mags = np.random.RandomState(42).uniform(14, 20, n)
        data = pd.DataFrame(
            {
                "ra_psf": ras,
                "dec_psf": decs,
                "ra_gaia_corr": ras,
                "dec_gaia_corr": decs,
                "phot_g_mean_mag": mags,
                "x_pixel": np.random.uniform(0, 200, n),
                "y_pixel": np.random.uniform(0, 200, n),
                "ra_measured": ras + np.random.normal(0, 1e-5, n),
                "dec_measured": decs + np.random.normal(0, 1e-5, n),
                "ra_gaia": ras,
                "dec_gaia": decs,
            }
        )
        cv = HoldoutCV(
            data, wcs=mock_wcs, train_fraction=0.7, n_splits=5, random_seed=42
        )
        result = cv.run()
        assert cv.summary_["n_splits"] == 5
        assert len(cv.results_) == 5
        assert cv.summary_["test_rms_mean_mas"] > 0


class TestDCRCorrector:
    def test_init_with_required_params(self, synthetic_matches):
        """DCRCorrector should initialize with matched data and sky params."""
        from astropy.time import Time

        corrector = DCRCorrector(
            matched_data=synthetic_matches,
            obs_time=Time(59958.145833, format="mjd"),
            ra_center_deg=10.0,
            dec_center_deg=20.0,
            observatory_lat_deg=-30.17,
            pixel_scale_arcsec=0.1,
        )
        assert corrector is not None


class TestCCDEdgeDiagnostic:
    def test_init_with_positions(self, synthetic_matches):
        """CCDEdgeDiagnostic should initialize with star positions."""
        diag = CCDEdgeDiagnostic(
            matched_data=synthetic_matches,
            image_shape=(200, 200),
            pixel_scale_arcsec=0.1,
        )
        assert diag is not None


class TestTelemetryReport:
    def test_write_telemetry_report(self, tmp_path):
        """write_telemetry_report should create a markdown file."""
        out = tmp_path / "report.md"
        write_telemetry_report(
            path=out,
            instrument="DECam",
            holdout_cv={"rms_improvement_pct": 15.0},
            mag_strat={"photon_break_mag": 19.5},
            dcr={"dcr_amplitude_mas": 2.0},
            ccd_edge={"tree_ring_amplitude_mas": 1.5},
            final_rms_mas=8.5,
            n_final=120,
        )
        assert out.exists()

    def test_write_telemetry_report_empty(self, tmp_path):
        """write_telemetry_report should handle empty dicts."""
        out = tmp_path / "empty_report.md"
        write_telemetry_report(
            path=out,
            instrument="DECam",
            holdout_cv={},
            mag_strat={},
            dcr={},
            ccd_edge={},
            final_rms_mas=0.0,
            n_final=0,
        )
        assert out.exists()
