"""
GPU backend abstraction for the PSF-match pipeline.

Provides device detection, optional import handling, and a single
``Backend`` dispatcher that exposes CPU fallbacks for every GPU
primitive.  All GPU libraries (CuPy, Numba CUDA, PyTorch) are optional;
this module imports cleanly without them.

Design
------

* ``Device`` — lightweight enum-like string identifying the active
  execution device ("cpu" or "cuda").
* ``DeviceManager`` — singleton that resolves the requested device,
  probes for CUDA availability, and emits structured log records of
  device selection and fallback events.
* ``Backend`` — central dispatch object.  Each numerical helper
  (sigma-clipped stats, polynomial features, ridge solve, batched
  photometry, batched PSF fitting) exposes a single method that picks
  the CPU or CUDA implementation based on the active device, the input
  size, and the configured stage toggles.

The dispatcher is intentionally minimal: stage modules call
``Backend.instance().<op>(...)`` and obtain NumPy arrays back.  CuPy
arrays are moved back to the host before returning so that downstream
astropy / pandas / scikit-learn code is unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.logger import get_logger

logger = get_logger("gpu.backend")


# ---------------------------------------------------------------------------
# Optional GPU library probes
# ---------------------------------------------------------------------------


def _probe_cupy() -> Any | None:
    """Returns the CuPy module if importable and CUDA is available."""
    try:
        import cupy  # type: ignore

        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
        return None
    except Exception:  # pragma: no cover - GPU not present in CI
        return None


def _probe_torch() -> Any | None:
    """Returns the torch module if importable and CUDA is available."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return torch
        return None
    except Exception:  # pragma: no cover
        return None


def _probe_numba_cuda() -> Any | None:
    """Returns ``numba.cuda`` if importable and a CUDA device is present."""
    try:
        from numba import cuda  # type: ignore

        if cuda.is_available():
            try:
                if cuda.gpus and len(cuda.gpus) > 0:
                    return cuda
            except Exception:
                return None
        return None
    except Exception:  # pragma: no cover
        return None


@dataclass(frozen=True)
class GPUAvailability:
    """Snapshot of optional GPU library availability."""

    cupy: Any | None
    torch: Any | None
    numba_cuda: Any | None

    @property
    def cuda_available(self) -> bool:
        return any((self.cupy, self.torch, self.numba_cuda))

    def describe(self) -> str:
        parts = []
        parts.append(f"cupy={'yes' if self.cupy else 'no'}")
        parts.append(f"torch={'yes' if self.torch else 'no'}")
        parts.append(f"numba_cuda={'yes' if self.numba_cuda else 'no'}")
        return ", ".join(parts)


def probe_gpu() -> GPUAvailability:
    """Probes the runtime for optional GPU libraries."""
    return GPUAvailability(
        cupy=_probe_cupy(),
        torch=_probe_torch(),
        numba_cuda=_probe_numba_cuda(),
    )


# ---------------------------------------------------------------------------
# Device manager
# ---------------------------------------------------------------------------


class DeviceManager:
    """
    Singleton device selector.

    Resolution order:

    1. If ``device="cpu"`` is forced, CPU is used regardless of GPU
       availability.
    2. If ``device="cuda"`` is forced but no GPU is available, the
       manager logs a fallback event and selects CPU.
    3. If ``device="auto"`` (default), CUDA is used when any GPU
       library is available; otherwise CPU.
    """

    _instance: DeviceManager | None = None

    def __init__(
        self,
        device: str = "auto",
        availability: GPUAvailability | None = None,
        deterministic: bool = False,
    ):
        self.requested_device = device.lower()
        self.deterministic = deterministic
        self.availability = availability or probe_gpu()
        # fallback_log is initialised before _resolve_device() to capture
        # any fallback events that _resolve_device() may record.
        self.fallback_log: list[str] = []
        self.device = self._resolve_device()
        logger.info(
            "DeviceManager initialised: requested=%s, resolved=%s, "
            "deterministic=%s, gpu_libs=[%s]",
            self.requested_device,
            self.device,
            self.deterministic,
            self.availability.describe(),
        )

    def _resolve_device(self) -> str:
        req = self.requested_device
        if req == "cpu":
            return "cpu"
        if req == "cuda":
            if not self.availability.cuda_available:
                self.fallback_log.append(
                    "CUDA requested but no GPU library available; falling back to CPU"
                )
                logger.warning(self.fallback_log[-1])
                return "cpu"
            return "cuda"
        # auto
        if self.availability.cuda_available:
            return "cuda"
        return "cpu"

    @classmethod
    def instance(cls) -> DeviceManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls, **kwargs) -> DeviceManager:
        cls._instance = cls(**kwargs)
        return cls._instance

    def record_fallback(self, stage: str, reason: str) -> None:
        msg = f"GPU fallback in stage '{stage}': {reason}"
        self.fallback_log.append(msg)
        logger.info(msg)

    def is_cuda(self) -> bool:
        return self.device == "cuda"


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


@dataclass
class StageTimer:
    """Accumulates per-stage wall-clock timings for GPU/CPU execution."""

    enabled: bool = False
    records: list[dict] = None

    def __post_init__(self):
        if self.records is None:
            self.records = []

    def time(self, label: str):
        outer = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.perf_counter()
                return self

            def __exit__(self, *exc):
                self.t1 = time.perf_counter()
                if outer.enabled:
                    outer.records.append(
                        {"label": label, "elapsed_s": self.t1 - self.t0}
                    )
                    logger.debug("gpu timing: %s = %.4fs", label, self.t1 - self.t0)

        return _Ctx()


# ---------------------------------------------------------------------------
# Backend dispatcher
# ---------------------------------------------------------------------------


class Backend:
    """
    Central numerical dispatch object.

    Stage modules call ``Backend.instance().<op>(...)`` and obtain
    NumPy arrays back.  Each operation:

    1. Checks the active device and per-stage GPU toggles.
    2. Selects the CPU or CUDA implementation.
    3. On CUDA failure, records a fallback event and retries on CPU.

    The dispatcher is intentionally conservative: small inputs are
    always handled on CPU to avoid host-device transfer overhead.
    """

    _instance: Backend | None = None

    # Defaults; overridden from config when a PipelineConfig is available.
    _DEFAULT_MIN_GPU_ELEMENTS = 500
    _DEFAULT_MIN_GPU_STARS = 50

    def __init__(
        self,
        device_manager: DeviceManager | None = None,
        stage_toggles: dict[str, bool] | None = None,
        batch_size: int = 4096,
        enable_timing: bool = False,
        min_gpu_elements: int = 500,
        min_gpu_stars: int = 50,
    ):
        self.dm = device_manager or DeviceManager.instance()
        self.stage_toggles = stage_toggles or {}
        self.batch_size = int(batch_size)
        self.min_gpu_elements = int(min_gpu_elements)
        self.min_gpu_stars = int(min_gpu_stars)
        self.timer = StageTimer(enabled=enable_timing)
        # Lazily-imported operation modules; only loaded when needed.
        self._ops_cpu = None
        self._ops_gpu = None
        self._torch_ops = None

    @classmethod
    def instance(cls) -> Backend:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls, **kwargs) -> Backend:
        cls._instance = cls(**kwargs)
        return cls._instance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def stage_enabled(self, stage: str) -> bool:
        """Returns True if the GPU path is enabled for the given stage."""
        if not self.dm.is_cuda():
            return False
        flag = self.stage_toggles.get(stage)
        if flag is None:
            # Default: stages 1, 2, 3, 5, 6, 7, 9 opt in; 0, 4, 8 stay CPU
            default_stages = {
                "detection",
                "psf",
                "photometry",
                "systematic",
                "advanced",
                "enhancements",
                "faint",
            }
            return stage in default_stages
        return bool(flag)

    def _large_enough(self, arr, threshold=None) -> bool:
        threshold = threshold if threshold is not None else self.min_gpu_elements
        try:
            return int(np.asarray(arr).size) >= threshold
        except Exception:
            return False

    def _ensure_cpu_ops(self):
        if self._ops_cpu is None:
            from src.gpu import ops_cpu

            self._ops_cpu = ops_cpu
        return self._ops_cpu

    def _ensure_gpu_ops(self):
        if self._ops_gpu is None:
            from src.gpu import ops_gpu

            self._ops_gpu = ops_gpu
        return self._ops_gpu

    def _ensure_torch_ops(self):
        if self._torch_ops is None:
            from src.gpu import torch_ops

            self._torch_ops = torch_ops
        return self._torch_ops

    # ------------------------------------------------------------------
    # Numerical primitives
    # ------------------------------------------------------------------

    def sigma_clipped_stats(self, data, sigma=3.0, maxiters=10):
        """Sigma-clipped mean/median/std with optional CuPy acceleration."""
        op = "sigma_clipped_stats"
        if self.stage_enabled("detection") and self._large_enough(
            data, self.min_gpu_elements
        ):
            try:
                return self._ensure_gpu_ops().sigma_clipped_stats(
                    data, sigma=sigma, maxiters=maxiters, cupy=self.dm.availability.cupy
                )
            except Exception as exc:  # pragma: no cover - GPU path
                self.dm.record_fallback(op, str(exc))
        return self._ensure_cpu_ops().sigma_clipped_stats(
            data, sigma=sigma, maxiters=maxiters
        )

    def batched_aperture_photometry(
        self, image, positions, radius, annulus_inner, annulus_outer
    ):
        """Batched aperture+annulus photometry.

        Returns a dict with ``flux`` and ``bkg_per_pixel`` arrays.
        """
        op = "batched_aperture_photometry"
        if (
            self.stage_enabled("photometry")
            and len(positions) >= self.min_gpu_stars
            and self.dm.availability.cupy is not None
        ):
            try:
                return self._ensure_gpu_ops().batched_aperture_photometry(
                    image,
                    positions,
                    radius,
                    annulus_inner,
                    annulus_outer,
                    cupy=self.dm.availability.cupy,
                    batch_size=self.batch_size,
                )
            except Exception as exc:  # pragma: no cover
                self.dm.record_fallback(op, str(exc))
        return self._ensure_cpu_ops().batched_aperture_photometry(
            image, positions, radius, annulus_inner, annulus_outer
        )

    def ridge_solve(self, X, y, alpha, weights=None):
        """Closed-form Ridge regression solve: (XᵀWX + αI) β = XᵀWy.

        Returns the coefficient vector.  Uses CuPy when available
        and the design matrix is large enough to benefit.
        """
        op = "ridge_solve"
        if (
            self.stage_enabled("systematic")
            and self._large_enough(X, 2 * self.min_gpu_elements)
            and self.dm.availability.cupy is not None
        ):
            try:
                return self._ensure_gpu_ops().ridge_solve(
                    X, y, alpha, weights=weights, cupy=self.dm.availability.cupy
                )
            except Exception as exc:  # pragma: no cover
                self.dm.record_fallback(op, str(exc))
        return self._ensure_cpu_ops().ridge_solve(X, y, alpha, weights=weights)

    def closed_form_loocv_linear(self, X, y, alpha, weights=None):
        """Closed-form leave-one-out CV residuals for Ridge regression.

        For weighted Ridge:

            H = X (Xᵀ W X + αI)^{-1} Xᵀ W
            r_loo_i = (y_i - ŷ_i) / (1 - H_ii)

        Returns the LOO residual array (same length as y).  This
        avoids the O(N²) re-fit loop and scales to large catalogs.
        """
        op = "closed_form_loocv_linear"
        if (
            self.stage_enabled("systematic")
            and self._large_enough(X, self.min_gpu_elements)
            and self.dm.availability.cupy is not None
        ):
            try:
                return self._ensure_gpu_ops().closed_form_loocv_linear(
                    X, y, alpha, weights=weights, cupy=self.dm.availability.cupy
                )
            except Exception as exc:  # pragma: no cover
                self.dm.record_fallback(op, str(exc))
        return self._ensure_cpu_ops().closed_form_loocv_linear(
            X, y, alpha, weights=weights
        )

    def batched_psf_fit(
        self,
        stamps,
        init_params,
        psf_model_fn,
        fit_shape,
    ):
        """Batched nonlinear PSF fitting via PyTorch.

        ``psf_model_fn(params, grid)`` returns predicted stamp values
        given a batch of parameters.  Falls back to a CPU loop when
        PyTorch is unavailable or the batch is small.
        """
        op = "batched_psf_fit"
        if (
            self.stage_enabled("photometry")
            and len(stamps) >= self.min_gpu_stars
            and self.dm.availability.torch is not None
        ):
            try:
                return self._ensure_torch_ops().batched_psf_fit(
                    stamps,
                    init_params,
                    psf_model_fn,
                    fit_shape,
                    torch=self.dm.availability.torch,
                    batch_size=self.batch_size,
                    deterministic=self.dm.deterministic,
                )
            except Exception as exc:  # pragma: no cover
                self.dm.record_fallback(op, str(exc))
        return self._ensure_cpu_ops().batched_psf_fit(
            stamps, init_params, psf_model_fn, fit_shape
        )

    def pairwise_distances(self, A, B, max_radius=None):
        """Pairwise Euclidean distances with optional radius cutoff.

        Used by Stage 4 neighbour search and Stage 7 bootstrap.  Returns
        a dense ``(N, M)`` distance matrix on CPU.
        """
        op = "pairwise_distances"
        if (
            self.stage_enabled("enhancements")
            and (A.shape[0] * B.shape[0]) >= self.min_gpu_elements
            and self.dm.availability.cupy is not None
        ):
            try:
                return self._ensure_gpu_ops().pairwise_distances(
                    A, B, max_radius=max_radius, cupy=self.dm.availability.cupy
                )
            except Exception as exc:  # pragma: no cover
                self.dm.record_fallback(op, str(exc))
        return self._ensure_cpu_ops().pairwise_distances(A, B, max_radius=max_radius)

    def radial_profile_mad(self, psf_data, center, bin_width=0.5, max_radius=None):
        """MAD-clipped radial profile used by PSF azimuthal homogenisation."""
        op = "radial_profile_mad"
        if (
            self.stage_enabled("psf")
            and self._large_enough(psf_data, max(self.min_gpu_elements // 4, 100))
            and self.dm.availability.cupy is not None
        ):
            try:
                return self._ensure_gpu_ops().radial_profile_mad(
                    psf_data,
                    center,
                    bin_width=bin_width,
                    max_radius=max_radius,
                    cupy=self.dm.availability.cupy,
                )
            except Exception as exc:  # pragma: no cover
                self.dm.record_fallback(op, str(exc))
        return self._ensure_cpu_ops().radial_profile_mad(
            psf_data, center, bin_width=bin_width, max_radius=max_radius
        )

    def bin_statistics(self, x, values, bin_edges, statistic="rms"):
        """Binned statistic over arbitrary bin edges.

        Used by magnitude stratification (Stage 7) and faint-star
        precision analysis (Stage 9).
        """
        op = "bin_statistics"
        if (
            self.stage_enabled("faint")
            and len(x) >= self.min_gpu_stars
            and self.dm.availability.cupy is not None
        ):
            try:
                return self._ensure_gpu_ops().bin_statistics(
                    x,
                    values,
                    bin_edges,
                    statistic=statistic,
                    cupy=self.dm.availability.cupy,
                )
            except Exception as exc:  # pragma: no cover
                self.dm.record_fallback(op, str(exc))
        return self._ensure_cpu_ops().bin_statistics(
            x, values, bin_edges, statistic=statistic
        )

    def tps_loocv(self, X, y, smoothing, degree=1):
        """Closed-form leave-one-out CV residuals for TPS (thin-plate
        spline) interpolation.

        X : (N, D) float array — spatial coordinates
        y : (N,) float array  — residual values
        smoothing : float    — regularisation parameter (λ)
        degree : int         — polynomial trend degree (0 or 1)

        Returns the LOO residual array (same length as y).

        This replaces the per-fold re-fit loop with a single
        block-matrix inversion:
            M = [[K + λI   P],
                 [P^T       0]]
            H = K @ M_inv[:N, :N] + P @ M_inv[N:, :N]
            r_loo = (y - H @ y) / (1 - diag(H))
        """
        op = "tps_loocv"
        if (
            self.stage_enabled("advanced")
            and len(X) >= self.min_gpu_stars
            and self.dm.availability.cupy is not None
        ):
            try:
                return self._ensure_gpu_ops().tps_loocv(
                    X,
                    y,
                    smoothing,
                    degree=degree,
                    cupy=self.dm.availability.cupy,
                )
            except Exception as exc:  # pragma: no cover
                self.dm.record_fallback(op, str(exc))
        return self._ensure_cpu_ops().tps_loocv(X, y, smoothing, degree=degree)


# ---------------------------------------------------------------------------
# Environment overrides (used by CLI/config to reconfigure the singleton)
# ---------------------------------------------------------------------------


def configure_backend(
    device: str = "auto",
    deterministic: bool = False,
    batch_size: int = 4096,
    stage_toggles: dict[str, bool] | None = None,
    enable_timing: bool = False,
    min_gpu_elements: int = 500,
    min_gpu_stars: int = 50,
) -> Backend:
    """(Re)creates the singleton DeviceManager and Backend with the
    supplied options.  Called by ``main.py`` once CLI arguments are
    parsed."""
    availability = probe_gpu()
    DeviceManager.reset(
        device=device,
        availability=availability,
        deterministic=deterministic,
    )
    return Backend.reset(
        stage_toggles=stage_toggles or {},
        batch_size=batch_size,
        enable_timing=enable_timing,
        min_gpu_elements=min_gpu_elements,
        min_gpu_stars=min_gpu_stars,
    )


def current_device() -> str:
    """Returns the active device string ("cpu" or "cuda")."""
    return DeviceManager.instance().device
