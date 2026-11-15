"""
Numba CUDA kernels for the PSF-match GPU layer.

These kernels are optional.  When Numba CUDA is unavailable, the
dispatcher uses the CuPy or CPU paths.  Each kernel function is
wrapped in a try/except so a missing ``numba.cuda`` does not break
imports.

Available kernels:

* ``sigma_clip_mask`` — element-wise sigma-clip mask on a 1D array.
* ``pairwise_distance_threshold`` — thresholded pairwise distances.
* ``radial_bin_index`` — per-pixel radial bin index assignment.
* ``batched_aperture_sum`` — batched aperture flux summation.

The kernels are written to be import-safe and called only after the
dispatcher confirms Numba CUDA availability.  Each ``_make_*`` helper
caches the compiled kernel so repeated calls reuse the same compiled
 artefact.
"""

from __future__ import annotations

import numpy as np

from src.logger import get_logger

logger = get_logger("gpu.kernels")

# Optional Numba CUDA import; falls back to None if unavailable.
try:
    from numba import cuda as _cuda  # type: ignore

    _NUMBA_CUDA_AVAILABLE = _cuda.is_available()
except Exception:  # pragma: no cover - import-time fallback
    _cuda = None
    _NUMBA_CUDA_AVAILABLE = False


def is_available() -> bool:
    """Returns True when Numba CUDA is importable and a device is present."""
    return _NUMBA_CUDA_AVAILABLE


# ---------------------------------------------------------------------------
# Kernel cache
# ---------------------------------------------------------------------------

_kernel_cache: dict = {}


def _get_kernel(name: str):
    """Returns a cached compiled kernel by name, or None if unavailable."""
    if not _NUMBA_CUDA_AVAILABLE:
        return None
    if name in _kernel_cache:
        return _kernel_cache[name]
    kernel = _build_kernel(name)
    if kernel is not None:
        _kernel_cache[name] = kernel
    return kernel


def _build_kernel(name: str):
    """Lazily JIT-compiles the requested kernel."""
    if not _NUMBA_CUDA_AVAILABLE:
        return None
    try:
        if name == "sigma_clip_mask":

            @_cuda.jit
            def _sigma_clip_mask(arr, med, sigma, out):
                i = _cuda.grid(1)
                if i < arr.size:
                    out[i] = abs(arr[i] - med) < sigma

            return _sigma_clip_mask

        if name == "pairwise_distance_threshold":

            @_cuda.jit
            def _pwd(A, B, max_r2, out):
                i, j = _cuda.grid(2)
                if i < A.shape[0] and j < B.shape[0]:
                    s = 0.0
                    for k in range(A.shape[1]):
                        d = A[i, k] - B[j, k]
                        s += d * d
                    out[i, j] = s if s <= max_r2 else 1e30

            return _pwd

        if name == "radial_bin_index":

            @_cuda.jit
            def _rbi(r_flat, bin_edges, out):
                i = _cuda.grid(1)
                if i < r_flat.size:
                    r = r_flat[i]
                    nb = bin_edges.shape[0] - 1
                    idx = -1
                    for b in range(nb):
                        if r >= bin_edges[b] and r < bin_edges[b + 1]:
                            idx = b
                            break
                    out[i] = idx

            return _rbi

        if name == "batched_aperture_sum":

            @_cuda.jit
            def _bas(image, positions, r2, out, n_per):
                i = _cuda.grid(1)
                if i < positions.shape[0]:
                    cx = positions[i, 0]
                    cy = positions[i, 1]
                    s = 0.0
                    n = 0
                    ny = image.shape[0]
                    nx = image.shape[1]
                    # Bounding box
                    r_int = int(np.sqrt(r2)) + 1
                    x_lo = max(0, int(cx) - r_int)
                    x_hi = min(nx, int(cx) + r_int + 1)
                    y_lo = max(0, int(cy) - r_int)
                    y_hi = min(ny, int(cy) + r_int + 1)
                    for yy in range(y_lo, y_hi):
                        for xx in range(x_lo, x_hi):
                            dx = xx - cx
                            dy = yy - cy
                            d2 = dx * dx + dy * dy
                            if d2 <= r2:
                                s += image[yy, xx]
                                n += 1
                    out[i] = s
                    n_per[i] = n

            return _bas

    except Exception as exc:  # pragma: no cover - kernel compile failure
        logger.warning("Failed to compile Numba kernel %s: %s", name, exc)
        return None
    return None


# ---------------------------------------------------------------------------
# Public helpers (CPU fallback when Numba unavailable)
# ---------------------------------------------------------------------------


def sigma_clip_mask(arr, median, sigma):
    """Returns a boolean mask: ``|arr - median| < sigma``.

    Uses the Numba kernel when available; otherwise a NumPy fallback.
    """
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if not _NUMBA_CUDA_AVAILABLE:
        return np.abs(arr - median) < sigma
    kernel = _get_kernel("sigma_clip_mask")
    if kernel is None:
        return np.abs(arr - median) < sigma
    out = np.zeros(arr.shape, dtype=np.bool_)
    threads = 256
    blocks = (arr.size + threads - 1) // threads
    kernel[blocks, threads](arr, float(median), float(sigma), out)
    return out


def pairwise_distance_threshold(A, B, max_radius):
    """Thresholded pairwise squared distances.

    Returns an ``(N, M)`` array of squared distances; entries beyond
    ``max_radius`` are set to ``1e30``.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    max_r2 = float(max_radius) ** 2
    if not _NUMBA_CUDA_AVAILABLE:
        diff = A[:, None, :] - B[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        return np.where(d2 <= max_r2, d2, 1e30)
    kernel = _get_kernel("pairwise_distance_threshold")
    if kernel is None:
        diff = A[:, None, :] - B[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        return np.where(d2 <= max_r2, d2, 1e30)
    out = np.full((A.shape[0], B.shape[0]), 1e30, dtype=np.float64)
    threads = (16, 16)
    blocks = (
        (A.shape[0] + threads[0] - 1) // threads[0],
        (B.shape[0] + threads[1] - 1) // threads[1],
    )
    kernel[blocks, threads](
        _cuda.to_device(A),
        _cuda.to_device(B),
        max_r2,
        _cuda.to_device(out),
    )
    return out
