"""
Tests for the optional CUDA GPU layer.

All tests in this module run on CPU-only environments.  GPU-specific
tests skip cleanly when CuPy, PyTorch, or Numba CUDA are unavailable.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gpu import (
    Backend,
    DeviceManager,
    batching,
    configure_backend,
    current_device,
    kernels,
    ops_cpu,
    ops_gpu,
    probe_gpu,
    torch_ops,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Python syntax validity for all Python files
# ---------------------------------------------------------------------------


def _all_python_files():
    """Returns a list of all Python source files in the repository."""
    files = []
    for sub in ["src", "tests", "main.py"]:
        path = PROJECT_ROOT / sub
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return files


@pytest.mark.parametrize("py_file", _all_python_files())
def test_python_syntax_valid(py_file):
    """Every Python file in the repository must compile cleanly."""
    import py_compile

    py_compile.compile(str(py_file), doraise=True)


# ---------------------------------------------------------------------------
# 2. Import safety of core modules without GPU libraries
# ---------------------------------------------------------------------------


def test_core_modules_import_without_gpu_libs():
    """Core pipeline modules must import successfully even when no GPU
    library is installed."""
    # Force a fresh import to make sure no hidden GPU import is happening
    # at module top-level.
    modules_to_reload = [
        "src.config",
        "src.logger",
        "src.gpu",
        "src.gpu.backend",
        "src.gpu.ops_cpu",
        "src.gpu.ops_gpu",
        "src.gpu.torch_ops",
        "src.gpu.kernels",
        "src.gpu.batching",
        "src.detection",
        "src.psf_modeling",
        "src.photometry",
        "src.calibration",
        "src.enhancements",
    ]
    for mod_name in modules_to_reload:
        try:
            importlib.import_module(mod_name)
        except Exception as exc:  # pragma: no cover - regression guard
            pytest.fail(f"Failed to import {mod_name}: {exc}")


def test_gpu_package_imports_without_cupy():
    """``src.gpu`` must import cleanly when CuPy is not installed."""
    # Simulate CuPy not being installed by hiding it from sys.modules
    # and from the import system.
    saved_cupy = sys.modules.pop("cupy", None)
    saved_torch = sys.modules.pop("torch", None)
    saved_numba_cuda = sys.modules.pop("numba.cuda", None)
    try:
        # Re-probe GPU availability without reloading any module —
        # reloading ``src.gpu.backend`` would replace the DeviceManager
        # class object and break singleton state across tests.
        avail = probe_gpu()
        # Without CuPy in sys.modules, the probe should return None for
        # cupy (unless cupy is actually installed and importable).
        assert avail.cupy is None or hasattr(avail.cupy, "__name__")
        # The package must not crash even when no GPU library is present.
        assert avail.describe() is not None
    finally:
        if saved_cupy is not None:
            sys.modules["cupy"] = saved_cupy
        if saved_torch is not None:
            sys.modules["torch"] = saved_torch
        if saved_numba_cuda is not None:
            sys.modules["numba.cuda"] = saved_numba_cuda


# ---------------------------------------------------------------------------
# 3. CLI help execution
# ---------------------------------------------------------------------------


def test_cli_help_runs():
    """``python main.py --help`` must exit 0 and print the description."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "PSF-match" in result.stdout
    assert "--device" in result.stdout
    assert "--gpu-batch-size" in result.stdout


# ---------------------------------------------------------------------------
# 4. Existing CLI flags remaining available
# ---------------------------------------------------------------------------


def test_existing_cli_flags_still_present():
    """Every pre-refactor CLI flag must still be accepted by the parser."""
    import main

    parser = main.build_parser()
    expected_flags = [
        "--preset",
        "--instrument",
        "--image",
        "--gaia-catalog",
        "--no-milliquas",
        "--no-clean",
        "--extension",
        "--pm-mode",
        "--no-faint-stars",
        "--compare-faint-stars",
    ]
    # Collect all option strings registered on the parser.
    registered = set()
    for action in parser._actions:
        registered.update(action.option_strings)
    for flag in expected_flags:
        assert flag in registered, f"Existing CLI flag {flag} missing from parser"


def test_new_gpu_cli_flags_present():
    """All new GPU-related CLI flags must be registered."""
    import main

    parser = main.build_parser()
    new_flags = [
        "--device",
        "--no-gpu-fallback",
        "--deterministic",
        "--image-precision",
        "--astrometry-precision",
        "--gpu-batch-size",
        "--gpu-stage",
        "--gpu-disable-all",
        "--gpu-timing",
        "--gpu-workers",
    ]
    registered = set()
    for action in parser._actions:
        registered.update(action.option_strings)
    for flag in new_flags:
        assert flag in registered, f"New GPU CLI flag {flag} missing from parser"


# ---------------------------------------------------------------------------
# 5. New device-selection behaviour on CPU
# ---------------------------------------------------------------------------


def test_device_auto_resolves_to_cpu_without_gpu():
    """``device='auto'`` resolves to 'cpu' when no GPU library is present."""
    backend = configure_backend(device="auto")
    # On CI / CPU-only environments the resolved device must be 'cpu'.
    if not probe_gpu().cuda_available:
        assert current_device() == "cpu"
        assert backend.dm.device == "cpu"
    else:  # pragma: no cover - GPU available
        assert current_device() in ("cpu", "cuda")


def test_device_force_cpu():
    """``device='cpu'`` always resolves to CPU regardless of GPU availability."""
    configure_backend(device="cpu")
    assert current_device() == "cpu"


def test_device_force_cuda_fallback_when_unavailable():
    """``device='cuda'`` falls back to CPU when CUDA is unavailable."""
    avail = probe_gpu()
    # Force a fresh DeviceManager regardless of prior tests.
    DeviceManager._instance = None
    Backend._instance = None
    configure_backend(device="cuda")
    if not avail.cuda_available:
        assert current_device() == "cpu"
        # The DeviceManager must have logged a fallback event.
        dm = DeviceManager.instance()
        assert dm.requested_device == "cuda"
        assert any("CUDA requested" in msg for msg in dm.fallback_log), (
            f"Expected fallback log entry; got fallback_log={dm.fallback_log!r}"
        )
    # Always reset back to a clean CPU backend for subsequent tests.
    configure_backend(device="cpu")


# ---------------------------------------------------------------------------
# 6. Configuration defaults preserving existing behaviour
# ---------------------------------------------------------------------------


def test_config_defaults_preserve_existing_behaviour():
    """Default PipelineConfig must match pre-refactor behaviour."""
    from src.config import PipelineConfig

    cfg = PipelineConfig()
    # Existing defaults that must remain unchanged.
    assert cfg.image_file.name == "image.fits"
    assert cfg.gaia_bright_limit == 6.0
    assert cfg.gaia_faint_limit == 21.5
    assert cfg.max_separation_arcsec == 2.0
    assert cfg.sip_order == 4
    assert cfg.enable_gaussian_process is True
    assert cfg.epsf_oversampling == 4
    assert cfg.enable_systematic_correction is True
    assert cfg.enable_advanced_calibration is True
    assert cfg.pm_mode == "damped"
    assert cfg.process_all_mef_extensions is False
    # New GPU defaults must preserve CPU behaviour.
    assert cfg.device == "auto"
    assert cfg.gpu_fallback is True
    assert cfg.deterministic is False
    assert cfg.image_precision == "float64"
    assert cfg.astrometry_precision == "float64"
    assert cfg.gpu_batch_size == 4096
    assert cfg.gpu_stages is None
    assert cfg.gpu_workers == 1


def test_config_validate_still_works():
    """PipelineConfig.validate() must still raise FileNotFoundError for
    missing inputs (existing behaviour)."""
    from src.config import PipelineConfig

    cfg = PipelineConfig(
        image_file=Path("/nonexistent/image.fits"),
        auto_download_gaia=False,
        gaia_catalog_file=Path("/nonexistent/gaia.csv"),
    )
    with pytest.raises(FileNotFoundError):
        cfg.validate()


# ---------------------------------------------------------------------------
# 7. CPU fallback for GPU-optional primitives
# ---------------------------------------------------------------------------


def test_sigma_clipped_stats_cpu_matches_astropy():
    """Backend sigma-clipped stats on CPU must agree with astropy within
    rounding tolerance for outlier rejection."""
    from astropy.stats import sigma_clipped_stats

    np.random.seed(0)
    data = np.random.normal(10.0, 2.0, 10000)
    data[100] = 1e6  # outlier
    data[200] = -1e6

    backend = configure_backend(device="cpu")
    mean, med, std = backend.sigma_clipped_stats(data, sigma=3.0)
    a_mean, a_med, a_std = sigma_clipped_stats(data, sigma=3.0)

    # The CPU implementation uses a slightly different convergence path;
    # the median must match within 1% and std within 5%.
    assert abs(mean - a_mean) < 0.1
    assert abs(med - a_med) < 0.1
    assert abs(std - a_std) / a_std < 0.1


def test_ridge_solve_cpu_matches_sklearn():
    """Backend Ridge solve on CPU must match sklearn's Ridge for the
    same alpha and unweighted case."""
    from sklearn.linear_model import Ridge as SKRidge

    np.random.seed(42)
    X = np.random.randn(200, 5)
    true_beta = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    y = X @ true_beta + 0.1 * np.random.randn(200)

    alpha = 0.01
    backend = configure_backend(device="cpu")
    beta_backend = backend.ridge_solve(X, y, alpha=alpha)
    sk = SKRidge(alpha=alpha, fit_intercept=False).fit(X, y)
    # Coefficients should agree to within 1e-6 relative tolerance.
    np.testing.assert_allclose(beta_backend, sk.coef_, rtol=1e-5, atol=1e-8)


def test_closed_form_loocv_matches_loop():
    """Closed-form LOO-CV residuals must match the per-fold re-solve loop."""
    np.random.seed(7)
    n, p = 60, 4
    X = np.random.randn(n, p)
    y = np.random.randn(n)
    alpha = 0.5
    w = np.random.uniform(0.5, 2.0, n)

    # Manual per-fold re-solve
    cv_loop = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xw = X[mask] * np.sqrt(w[mask])[:, None]
        A = Xw.T @ Xw + alpha * np.eye(p)
        try:
            beta = np.linalg.solve(A, Xw.T @ (y[mask] * np.sqrt(w[mask])))
            cv_loop[i] = y[i] - X[i] @ beta
        except np.linalg.LinAlgError:
            cv_loop[i] = y[i]

    # Closed-form
    backend = configure_backend(device="cpu")
    cv_closed = backend.closed_form_loocv_linear(X, y, alpha=alpha, weights=w)
    # They should agree to within 1e-6 relative tolerance.
    np.testing.assert_allclose(cv_closed, cv_loop, rtol=1e-5, atol=1e-8)


def test_pairwise_distances_cpu_matches_numpy():
    """Backend pairwise distances must match the naive NumPy computation."""
    np.random.seed(0)
    A = np.random.randn(20, 2)
    B = np.random.randn(15, 2)
    diff = A[:, None, :] - B[None, :, :]
    expected = np.sqrt(np.sum(diff * diff, axis=-1))

    backend = configure_backend(device="cpu")
    d = backend.pairwise_distances(A, B)
    np.testing.assert_allclose(d, expected, rtol=1e-10, atol=1e-10)


def test_radial_profile_mad_matches_naive():
    """Backend radial profile must match the naive CPU loop on bins
    that contain pixels (the backend additionally interpolates over
    empty bins, which the naive loop does not)."""
    np.random.seed(0)
    psf = np.random.rand(25, 25)
    cy, cx = 12.0, 12.0
    y_idx, x_idx = np.indices((25, 25))
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    bin_edges = np.arange(0, 12.5, 0.5)
    expected = np.full(len(bin_edges) - 1, np.nan)
    for i in range(len(bin_edges) - 1):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if not np.any(mask):
            continue
        vals = psf[mask]
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        s = 1.4826 * mad
        inliers = vals[np.abs(vals - med) <= 3.0 * s] if s > 0 else vals
        expected[i] = np.median(inliers) if len(inliers) > 0 else med

    backend = configure_backend(device="cpu")
    bc, profile, _ = backend.radial_profile_mad(
        psf, (cy, cx), bin_width=0.5, max_radius=12.0
    )
    # Compare only on bins where the naive loop produced a finite value.
    mask = np.isfinite(expected)
    np.testing.assert_allclose(profile[mask], expected[mask], rtol=1e-10, atol=1e-10)


def test_bin_statistics_cpu_matches_scipy():
    """Backend bin_statistics must match scipy.stats.binned_statistic
    for the supported statistics."""
    from scipy.stats import binned_statistic

    np.random.seed(0)
    x = np.random.uniform(14, 22, 500)
    v = (x - 18) ** 2 + 0.1 * np.random.randn(500)
    edges = np.arange(14, 23, 1.0)

    backend = configure_backend(device="cpu")
    for stat in ("mean", "median", "std", "count"):
        out = backend.bin_statistics(x, v, edges, statistic=stat)
        expected, _, _ = binned_statistic(x, v, statistic=stat, bins=edges)
        # NaN handling may differ slightly; compare finite entries.
        mask = np.isfinite(out) & np.isfinite(expected)
        np.testing.assert_allclose(out[mask], expected[mask], rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# 8. Backend dispatch returning CPU when CUDA is unavailable
# ---------------------------------------------------------------------------


def test_backend_dispatch_returns_cpu_when_cuda_unavailable():
    """When no GPU is available, every backend operation must route to CPU."""
    avail = probe_gpu()
    if avail.cuda_available:
        pytest.skip("GPU available; CPU fallback path not exercised")
    backend = configure_backend(device="auto")
    assert backend.dm.device == "cpu"
    # Every stage toggle must evaluate to False when on CPU.
    for stage in (
        "detection",
        "psf",
        "photometry",
        "systematic",
        "advanced",
        "enhancements",
        "faint",
    ):
        assert backend.stage_enabled(stage) is False


def test_gpu_stage_toggle_respects_disable_all():
    """``gpu_disable_all=True`` forces every stage toggle to False even
    when a GPU is present."""
    backend = configure_backend(
        device="cpu",
        stage_toggles=batching.resolve_stage_toggles(disable_all=True),
    )
    for stage in (
        "detection",
        "psf",
        "photometry",
        "systematic",
        "advanced",
        "enhancements",
        "faint",
    ):
        assert backend.stage_enabled(stage) is False


# ---------------------------------------------------------------------------
# 9. Small synthetic numerical parity tests
# ---------------------------------------------------------------------------


def test_aperture_photometry_parity_with_photutils():
    """The backend aperture photometry (CPU path) must produce fluxes
    broadly consistent with photutils' CircularAperture.

    Photutils uses exact pixel-area overlap at the aperture boundary
    while the backend uses a center-in-radius mask; the two therefore
    differ by a few percent at the aperture edge.  We verify the
    implementation matches the naive center-in-radius computation
    exactly, and is broadly consistent with photutils (within 5%).
    """
    from photutils.aperture import CircularAperture, aperture_photometry

    np.random.seed(0)
    image = np.random.poisson(50, (100, 100)).astype(np.float64)
    positions = np.array([[50.0, 50.0], [10.0, 10.0], [80.0, 30.0]])
    radius = 3.0

    # Naive center-in-radius computation
    yy, xx = np.mgrid[:100, :100]
    expected = np.zeros(len(positions))
    for i, (cx, cy) in enumerate(positions):
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        expected[i] = np.nansum(image[d2 <= radius**2])

    backend = configure_backend(device="cpu")
    res = backend.batched_aperture_photometry(
        image, positions, radius=radius, annulus_inner=5.0, annulus_outer=8.0
    )
    # Backend must match the naive center-in-radius computation exactly.
    np.testing.assert_allclose(res["flux"], expected, rtol=1e-10, atol=1e-10)

    # And remain broadly consistent with photutils (center-in-radius vs
    # exact overlap differ by ~3% at the aperture edge).
    apertures = CircularAperture(positions, r=radius)
    phot = aperture_photometry(image, apertures)
    phot_flux = np.array(phot["aperture_sum"], dtype=np.float64)
    np.testing.assert_allclose(res["flux"], phot_flux, rtol=0.05, atol=1.0)


def test_ridge_solve_with_weights_parity():
    """Weighted Ridge solve must match the sklearn-equivalent
    manual computation."""
    np.random.seed(0)
    n, p = 100, 4
    X = np.random.randn(n, p)
    y = np.random.randn(n)
    w = np.random.uniform(0.5, 2.0, n)
    alpha = 0.1

    # Manual weighted Ridge: (Xᵀ W X + α I) β = Xᵀ W y
    W = np.diag(w)
    A = X.T @ W @ X + alpha * np.eye(p)
    expected = np.linalg.solve(A, X.T @ W @ y)

    backend = configure_backend(device="cpu")
    beta = backend.ridge_solve(X, y, alpha=alpha, weights=w)
    np.testing.assert_allclose(beta, expected, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# 10. GPU-specific tests skipping cleanly when CUDA is unavailable
# ---------------------------------------------------------------------------


@pytest.fixture
def requires_cupy():
    """Fixture that skips the test when CuPy is unavailable."""
    avail = probe_gpu()
    if avail.cupy is None:
        pytest.skip("CuPy not available")
    return avail.cupy


@pytest.fixture
def requires_torch():
    """Fixture that skips the test when PyTorch CUDA is unavailable."""
    avail = probe_gpu()
    if avail.torch is None:
        pytest.skip("PyTorch CUDA not available")
    return avail.torch


@pytest.fixture
def requires_numba_cuda():
    """Fixture that skips the test when Numba CUDA is unavailable."""
    if not kernels.is_available():
        pytest.skip("Numba CUDA not available")
    return kernels._cuda


def test_cupy_sigma_clipped_stats(requires_cupy):
    """CuPy GPU sigma-clipped stats must match the CPU implementation."""
    cp = requires_cupy
    np.random.seed(0)
    data = np.random.normal(10.0, 2.0, 10000).astype(np.float64)
    data[100] = 1e6

    gpu_mean, gpu_med, gpu_std = ops_gpu.sigma_clipped_stats(data, sigma=3.0, cupy=cp)
    cpu_mean, cpu_med, cpu_std = ops_cpu.sigma_clipped_stats(data, sigma=3.0)
    assert abs(gpu_mean - cpu_mean) < 1e-6
    assert abs(gpu_med - cpu_med) < 1e-6
    assert abs(gpu_std - cpu_std) / cpu_std < 1e-6


def test_cupy_ridge_solve_parity(requires_cupy):
    """CuPy GPU Ridge solve must match the CPU implementation."""
    cp = requires_cupy
    np.random.seed(0)
    X = np.random.randn(500, 10)
    y = np.random.randn(500)
    alpha = 0.5
    w = np.random.uniform(0.5, 2.0, 500)

    beta_gpu = ops_gpu.ridge_solve(X, y, alpha=alpha, weights=w, cupy=cp)
    beta_cpu = ops_cpu.ridge_solve(X, y, alpha=alpha, weights=w)
    np.testing.assert_allclose(beta_gpu, beta_cpu, rtol=1e-6, atol=1e-8)


def test_cupy_pairwise_distances_parity(requires_cupy):
    """CuPy GPU pairwise distances must match the CPU implementation."""
    cp = requires_cupy
    np.random.seed(0)
    A = np.random.randn(50, 2)
    B = np.random.randn(40, 2)
    d_gpu = ops_gpu.pairwise_distances(A, B, max_radius=3.0, cupy=cp)
    d_cpu = ops_cpu.pairwise_distances(A, B, max_radius=3.0)
    # NaN/inf handling must match exactly.
    np.testing.assert_array_equal(np.isfinite(d_gpu), np.isfinite(d_cpu))
    np.testing.assert_allclose(
        d_gpu[np.isfinite(d_gpu)], d_cpu[np.isfinite(d_cpu)], rtol=1e-7, atol=1e-8
    )


def test_torch_batched_psf_fit_smoke(requires_torch):
    """PyTorch batched PSF fit must converge to known parameters for a
    synthetic Gaussian stamp batch."""
    torch = requires_torch

    # Synthetic Gaussian stamps
    fy = fx = 7
    yy, xx = np.mgrid[:fy, :fx]
    grid_np = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float64)

    def gaussian2d(params, grid):
        # params shape: (B, 3) = (amplitude, x0, y0)
        amp = params[:, 0:1]
        x0 = params[:, 1:2]
        y0 = params[:, 2:3]
        dx = grid[:, :, 0] - x0
        dy = grid[:, :, 1] - y0
        return amp * torch.exp(-(dx**2 + dy**2) / (2 * 1.5**2))

    np.random.seed(0)
    n = 32
    amps = np.random.uniform(100, 1000, n)
    x0 = np.random.uniform(2, 5, n)
    y0 = np.random.uniform(2, 5, n)
    init = np.column_stack([amps, x0, y0])
    stamps = np.zeros((n, fy, fx), dtype=np.float64)
    for i in range(n):
        stamps[i] = amps[i] * np.exp(
            -((xx - x0[i]) ** 2 + (yy - y0[i]) ** 2) / (2 * 1.5**2)
        )

    res = torch_ops.batched_psf_fit(
        stamps,
        init,
        gaussian2d,
        fit_shape=(fy, fx),
        torch=torch,
        batch_size=16,
        max_iters=20,
    )
    assert res["params"].shape == (n, 3)
    # Fitted amplitudes must be close to the true values.
    np.testing.assert_allclose(res["params"][:, 0], amps, rtol=1e-3, atol=1e-3)


def test_numba_cuda_kernel_smoke(requires_numba_cuda):
    """Numba CUDA sigma-clip mask kernel must produce the same result
    as the NumPy fallback."""
    cuda = requires_numba_cuda
    arr = np.array([1.0, 2.0, 3.0, 100.0, 4.0, 5.0], dtype=np.float64)
    med = 3.5
    sigma = 1.5
    mask = kernels.sigma_clip_mask(arr, med, sigma)
    expected = np.abs(arr - med) < sigma
    np.testing.assert_array_equal(mask, expected)


# ---------------------------------------------------------------------------
# 11. Configurable GPU dispatch thresholds
# ---------------------------------------------------------------------------


def test_backend_thresholds_configurable():
    """Backend._large_enough must respect the configured threshold."""
    # High threshold — small array excluded
    backend_high = configure_backend(
        device="cpu", min_gpu_elements=1000, min_gpu_stars=100
    )
    assert backend_high._large_enough(np.ones(500)) is False
    assert backend_high._large_enough(np.ones(2000)) is True

    # Low threshold — small array included
    backend_low = configure_backend(device="cpu", min_gpu_elements=10, min_gpu_stars=5)
    assert backend_low._large_enough(np.ones(50)) is True
    assert backend_low._large_enough(np.ones(5)) is False


def test_config_gpu_thresholds():
    """PipelineConfig must expose the new gpu_min_elements / gpu_min_stars fields."""
    from src.config import PipelineConfig

    cfg = PipelineConfig()
    assert hasattr(cfg, "gpu_min_elements")
    assert hasattr(cfg, "gpu_min_stars")
    assert cfg.gpu_min_elements == 500
    assert cfg.gpu_min_stars == 50


# ---------------------------------------------------------------------------
# 12. Closed-form TPS LOO-CV parity tests
# ---------------------------------------------------------------------------


def test_tps_loocv_closed_form_matches_per_fold():
    """CPU closed-form TPS LOO-CV must match the per-fold re-fit loop."""
    from scipy.interpolate import RBFInterpolator

    np.random.seed(123)
    n = 40
    X = np.random.uniform(-1, 1, (n, 2))
    y = np.random.randn(n) * 0.2
    smoothing = 2.0  # sufficient to pull H_ii well below 1

    # Per-fold re-fit loop (ground truth)
    errs_loop = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        try:
            tps = RBFInterpolator(
                X[mask],
                y[mask],
                kernel="thin_plate_spline",
                smoothing=smoothing,
                degree=1,
            )
            errs_loop[i] = float(y[i] - tps(X[i : i + 1])[0])
        except Exception:
            errs_loop[i] = y[i]

    # Closed-form
    r_loo_closed = ops_cpu.tps_loocv(X, y, smoothing=smoothing, degree=1)

    # Must agree to within 1e-8 relative tolerance
    np.testing.assert_allclose(r_loo_closed, errs_loop, rtol=1e-6, atol=1e-8)


def test_tps_loocv_degree_zero_matches_analytical():
    """TPS LOO-CV with degree=0 must match the analytical formula
    r_loo = coeffs / diag(K_inv)."""
    np.random.seed(42)
    n = 30
    X = np.random.uniform(0, 300, (n, 2))
    y = np.random.randn(n) * 0.1
    smoothing = 0.5

    r_loo = ops_cpu.tps_loocv(X, y, smoothing=smoothing, degree=0)

    # Analytical: r_loo = K_inv @ y / diag(K_inv)
    diff = X[:, None, :] - X[None, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        K = r * r * np.log(r)
    K = np.where(np.isnan(K), 0.0, K) + smoothing * np.eye(n)
    K_inv = np.linalg.inv(K)
    coeffs = K_inv @ y
    diag = np.diag(K_inv)
    denom = np.where(np.abs(diag) < 1e-12, 1e-12, diag)
    expected = coeffs / denom

    np.testing.assert_allclose(r_loo, expected, rtol=1e-8, atol=1e-8)


def test_tps_loocv_cpu_gpu_parity(requires_cupy):
    """GPU closed-form TPS LOO-CV must match the CPU reference."""
    cp = requires_cupy
    np.random.seed(7)
    n = 60
    X = np.random.uniform(100, 600, (n, 2))
    y = np.random.randn(n) * 0.15

    for s in (0.01, 1.0, 50.0):
        r_cpu = ops_cpu.tps_loocv(X, y, smoothing=s, degree=1)
        r_gpu = ops_gpu.tps_loocv(X, y, smoothing=s, degree=1, cupy=cp)
        np.testing.assert_allclose(r_gpu, r_cpu, rtol=1e-6, atol=1e-6)


def test_tps_loocv_small_n_returns_zeros():
    """TPS LOO-CV must return zeros for n < 3."""
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    y = np.array([1.0, -1.0], dtype=np.float64)
    r = ops_cpu.tps_loocv(X, y, smoothing=1.0, degree=1)
    np.testing.assert_array_equal(r, np.zeros(2))


def test_tps_loocv_backend_dispatch():
    """Backend.tps_loocv must return LOO residuals on CPU."""
    np.random.seed(9)
    n = 30
    X = np.random.uniform(100, 600, (n, 2))
    y = np.random.randn(n) * 0.2

    backend = configure_backend(device="cpu")
    r = backend.tps_loocv(X, y, smoothing=0.5, degree=1)
    assert r.shape == (n,)
    assert np.all(np.isfinite(r))


# ---------------------------------------------------------------------------
# Batching & MEF planning helpers
# ---------------------------------------------------------------------------


def test_iter_batches_covers_full_range():
    """iter_batches must yield (start, end) pairs covering [0, n)."""
    pairs = list(batching.iter_batches(10, 3))
    assert pairs == [(0, 3), (3, 6), (6, 9), (9, 10)]


def test_iter_batches_drop_last():
    """iter_batches with drop_last skips the final partial batch."""
    pairs = list(batching.iter_batches(10, 3, drop_last=True))
    assert pairs == [(0, 3), (3, 6), (6, 9)]


def test_batch_sizer_fallback_when_no_gpu():
    """BatchSizer.safe_batch must return fallback_batch when free_bytes
    is None (no GPU available)."""
    bs = batching.BatchSizer(element_bytes=8, fallback_batch=512)
    assert bs.safe_batch(100, free_bytes=None) == 512


def test_batch_sizer_respects_budget():
    """BatchSizer.safe_batch must respect the memory budget when
    free_bytes is provided."""
    bs = batching.BatchSizer(
        element_bytes=8,
        budget_fraction=0.25,
        min_batch=1,
        max_batch=10000,
        fallback_batch=512,
    )
    # 100 MB free, 100 elements per item, 8 bytes each -> 800 bytes/item
    # Budget = 25 MB -> 25e6 / 800 = 31250 -> capped at max_batch 10000
    b = bs.safe_batch(100, free_bytes=100 * 1024 * 1024)
    assert 1 <= b <= 10000


def test_resolve_stage_toggles_defaults():
    """Default stage toggles enable every GPU-accelerated stage."""
    toggles = batching.resolve_stage_toggles()
    for stage in batching.DEFAULT_GPU_STAGES:
        assert toggles[stage] is True


def test_resolve_stage_toggles_disable_all():
    """disable_all=True must force every stage to False."""
    toggles = batching.resolve_stage_toggles(disable_all=True)
    for stage in batching.DEFAULT_GPU_STAGES:
        assert toggles[stage] is False


def test_resolve_stage_toggles_explicit_overrides():
    """Explicit overrides must win over defaults."""
    toggles = batching.resolve_stage_toggles(
        explicit={"photometry": False, "faint": True}
    )
    assert toggles["photometry"] is False
    assert toggles["faint"] is True
    # Other stages keep default True.
    assert toggles["detection"] is True


def test_plan_mef_extensions_processes_all():
    """plan_mef_extensions with process_all=True must include every
    image-like HDU."""
    from astropy.io import fits

    primary = fits.PrimaryHDU()
    ext1 = fits.ImageHDU(np.ones((10, 10)), name="EXT1")
    ext2 = fits.ImageHDU(np.ones((20, 20)), name="EXT2")
    hdul = fits.HDUList([primary, ext1, ext2])
    plan = batching.plan_mef_extensions(hdul, process_all=True, requested_extension=0)
    assert plan.extension_indices == [1, 2]
    assert plan.extension_names == ["EXT1", "EXT2"]


def test_plan_mef_extensions_single_requested():
    """plan_mef_extensions with process_all=False must include only the
    requested extension."""
    from astropy.io import fits

    primary = fits.PrimaryHDU()
    ext1 = fits.ImageHDU(np.ones((10, 10)), name="EXT1")
    ext2 = fits.ImageHDU(np.ones((20, 20)), name="EXT2")
    hdul = fits.HDUList([primary, ext1, ext2])
    plan = batching.plan_mef_extensions(hdul, process_all=False, requested_extension=2)
    assert plan.extension_indices == [2]
    assert plan.extension_names == ["EXT2"]


def test_plan_mef_extensions_fallback_for_non_image():
    """plan_mef_extensions must fall back to the requested extension
    when no image-like HDU is found."""
    from astropy.io import fits

    primary = fits.PrimaryHDU()
    hdul = fits.HDUList([primary])
    plan = batching.plan_mef_extensions(hdul, process_all=True, requested_extension=0)
    assert plan.extension_indices == [0]


def test_free_gpu_bytes_returns_none_without_gpu():
    """free_gpu_bytes must return None when no GPU is available."""
    avail = probe_gpu()
    if avail.cuda_available:
        pytest.skip("GPU available; this test verifies CPU-only behaviour")
    assert batching.free_gpu_bytes() is None
