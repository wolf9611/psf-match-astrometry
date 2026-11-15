"""
CPU reference implementations for the GPU backend dispatcher.

These are pure-NumPy implementations of every numerical primitive the
pipeline's GPU layer can accelerate.  The dispatcher in
``src.gpu.backend`` falls back to these when:

* No GPU library is installed.
* The active device is CPU (forced or auto-detected).
* The input is too small to benefit from GPU offload.
* A GPU attempt raised an exception.

Keeping these here (rather than inlining them in the dispatcher) means
the CPU pipeline never imports CuPy/PyTorch, even indirectly.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Sigma-clipped statistics
# ---------------------------------------------------------------------------


def sigma_clipped_stats(data, sigma=3.0, maxiters=10):
    """Sigma-clipped mean/median/std computed on a 1D-flattened copy.

    Mirrors ``astropy.stats.sigma_clipped_stats`` for the median and
    std, but is intentionally lightweight to keep the GPU fallback
    path simple.
    """
    arr = np.asarray(data, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mask = np.isfinite(arr)
    if not np.any(mask):
        return 0.0, 0.0, 0.0
    arr = arr[mask]
    for _ in range(maxiters):
        med = np.median(arr)
        std = np.std(arr)
        if std == 0:
            break
        keep = np.abs(arr - med) < sigma * std
        if keep.all() or keep.sum() < 3:
            break
        arr = arr[keep]
    return float(np.mean(arr)), float(np.median(arr)), float(np.std(arr))


# ---------------------------------------------------------------------------
# Batched aperture photometry
# ---------------------------------------------------------------------------


def batched_aperture_photometry(image, positions, radius, annulus_inner, annulus_outer):
    """CPU reference for batched aperture + annulus photometry.

    ``positions`` is an ``(N, 2)`` array of ``(x, y)`` pixel
    coordinates.  Returns ``{"flux": (N,), "bkg_per_pixel": (N,)}``.
    """
    image = np.asarray(image, dtype=np.float64)
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    n = positions.shape[0]
    flux = np.zeros(n, dtype=np.float64)
    bkg_per_pixel = np.zeros(n, dtype=np.float64)

    ny, nx = image.shape
    yy, xx = np.mgrid[:ny, :nx]
    r2 = xx**2 + yy**2  # placeholder; per-source mask built below

    for i in range(n):
        cx, cy = positions[i, 0], positions[i, 1]
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        # Bounding box clipped to image
        r_max = max(radius, annulus_outer)
        x_lo = max(0, int(np.floor(cx - r_max)))
        x_hi = min(nx, int(np.ceil(cx + r_max)) + 1)
        y_lo = max(0, int(np.floor(cy - r_max)))
        y_hi = min(ny, int(np.ceil(cy + r_max)) + 1)
        if x_hi <= x_lo or y_hi <= y_lo:
            continue
        sub = image[y_lo:y_hi, x_lo:x_hi]
        sub_yy = yy[y_lo:y_hi, x_lo:x_hi] - cy
        sub_xx = xx[y_lo:y_hi, x_lo:x_hi] - cx
        d2 = sub_xx**2 + sub_yy**2
        aperture_mask = d2 <= radius**2
        annulus_mask = (d2 >= annulus_inner**2) & (d2 <= annulus_outer**2)
        flux[i] = float(np.nansum(sub[aperture_mask]))
        n_ann = int(annulus_mask.sum())
        if n_ann > 0:
            bkg_per_pixel[i] = float(np.nansum(sub[annulus_mask])) / n_ann
    return {"flux": flux, "bkg_per_pixel": bkg_per_pixel}


# ---------------------------------------------------------------------------
# Ridge regression solve (weighted)
# ---------------------------------------------------------------------------


def ridge_solve(X, y, alpha, weights=None):
    """Closed-form weighted Ridge regression.

    Solves `` (Xᵀ W X + α I) β = Xᵀ W y`` where ``W`` is a diagonal
    weight matrix.  Returns the coefficient vector.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"ridge_solve: X has {X.shape[0]} rows but y has {y.shape[0]}")
    n_features = X.shape[1]
    if weights is None:
        XtX = X.T @ X
        Xty = X.T @ y
    else:
        w = np.sqrt(np.asarray(weights, dtype=np.float64).ravel())
        if w.shape[0] != X.shape[0]:
            raise ValueError("ridge_solve: weights length mismatch")
        Xw = X * w[:, None]
        XtX = Xw.T @ Xw
        Xty = Xw.T @ (y * w)
    A = XtX + alpha * np.eye(n_features)
    beta = np.linalg.solve(A, Xty)
    return beta


def closed_form_loocv_linear(X, y, alpha, weights=None):
    """Closed-form LOO-CV residuals for weighted Ridge regression.

    Uses the hat-matrix identity:

        H = X (Xᵀ W X + α I)^{-1} Xᵀ W
        r_loo_i = (y_i - ŷ_i) / (1 - H_ii)

    Returns the array of leave-one-out residuals (same shape as ``y``).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n_features = X.shape[1]
    if weights is None:
        W = np.eye(X.shape[0])
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        W = np.diag(w)
    A = X.T @ W @ X + alpha * np.eye(n_features)
    A_inv = np.linalg.inv(A)
    # ŷ = X β = X A_inv Xᵀ W y = H y
    H = X @ A_inv @ X.T @ W
    y_hat = H @ y
    diag_H = np.diag(H)
    denom = 1.0 - diag_H
    # Avoid division by zero
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return (y - y_hat) / denom


# ---------------------------------------------------------------------------
# Batched PSF fitting (CPU fallback)
# ---------------------------------------------------------------------------


def batched_psf_fit(stamps, init_params, psf_model_fn, fit_shape):
    """CPU fallback for batched nonlinear PSF fitting.

    Iterates over stamps and runs a small Levenberg-Marquardt fit per
    source via ``scipy.optimize.least_squares``.  Returns a dict with
    ``params`` ``(N, P)`` and ``success`` ``(N,)`` boolean arrays.
    """
    from scipy.optimize import least_squares

    stamps = np.asarray(stamps, dtype=np.float64)
    init_params = np.atleast_2d(np.asarray(init_params, dtype=np.float64))
    n = len(stamps)
    n_params = init_params.shape[1]
    out_params = np.zeros((n, n_params), dtype=np.float64)
    out_success = np.zeros(n, dtype=bool)

    # Build a grid matching the model function convention
    fy, fx = fit_shape
    yy, xx = np.mgrid[:fy, :fx]
    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float64)

    for i in range(n):
        stamp = stamps[i].ravel()
        p0 = init_params[i]
        try:

            def residual(p, stamp=stamp, grid=grid):
                return psf_model_fn(p, grid).ravel() - stamp

            res = least_squares(residual, p0, max_nfev=200)
            out_params[i] = res.x
            out_success[i] = bool(res.success)
        except Exception:
            out_params[i] = p0
            out_success[i] = False
    return {"params": out_params, "success": out_success}


# ---------------------------------------------------------------------------
# Pairwise distances (CPU)
# ---------------------------------------------------------------------------


def pairwise_distances(A, B, max_radius=None):
    """Dense pairwise Euclidean distance matrix between rows of A and B.

    When ``max_radius`` is set, distances beyond it are set to ``np.inf``
    (still dense, but callers can mask cheaply).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    # (N, M, D) difference tensor
    diff = A[:, None, :] - B[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    if max_radius is not None:
        d[d > max_radius] = np.inf
    return d


# ---------------------------------------------------------------------------
# Radial profile with MAD clipping
# ---------------------------------------------------------------------------


def radial_profile_mad(psf_data, center, bin_width=0.5, max_radius=None):
    """MAD-clipped radial profile.

    Returns a tuple ``(bin_centers, profile, n_per_bin)``.
    """
    psf = np.asarray(psf_data, dtype=np.float64)
    ny, nx = psf.shape
    cy, cx = center
    y_idx, x_idx = np.indices((ny, nx))
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    if max_radius is None:
        max_radius = min(cx, cy) * 0.95
    bin_edges = np.arange(0.0, max_radius + bin_width, bin_width)
    n_bins = len(bin_edges) - 1
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    profile = np.full(n_bins, np.nan)
    n_per_bin = np.zeros(n_bins, dtype=int)

    r_flat = r.ravel()
    v_flat = psf.ravel()
    for i in range(n_bins):
        mask = (r_flat >= bin_edges[i]) & (r_flat < bin_edges[i + 1])
        if not np.any(mask):
            continue
        vals = v_flat[mask]
        n_per_bin[i] = len(vals)
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        sigma_hat = 1.4826 * mad
        if sigma_hat > 0:
            inliers = vals[np.abs(vals - med) <= 3.0 * sigma_hat]
        else:
            inliers = vals
        profile[i] = np.median(inliers) if len(inliers) > 0 else med
    valid = np.isfinite(profile)
    if not np.all(valid) and np.any(valid):
        idx = np.arange(n_bins)
        profile[~valid] = np.interp(idx[~valid], idx[valid], profile[valid])
    return bin_centers, profile, n_per_bin


# ---------------------------------------------------------------------------
# Binned statistics
# ---------------------------------------------------------------------------


def bin_statistics(x, values, bin_edges, statistic="rms"):
    """Binned statistic over arbitrary bin edges.

    ``statistic`` is one of {"mean", "median", "rms", "std", "count"}.
    Returns an array of length ``len(bin_edges) - 1``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()
    n_bins = len(bin_edges) - 1
    out = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if not np.any(mask):
            continue
        v = values[mask]
        if statistic == "mean":
            out[i] = np.mean(v)
        elif statistic == "median":
            out[i] = np.median(v)
        elif statistic == "rms":
            out[i] = np.sqrt(np.mean(v**2))
        elif statistic == "std":
            out[i] = np.std(v)
        elif statistic == "count":
            out[i] = len(v)
        else:
            raise ValueError(f"bin_statistics: unknown statistic {statistic!r}")
    return out


# ---------------------------------------------------------------------------
# Closed-form TPS leave-one-out CV
# ---------------------------------------------------------------------------


def tps_loocv(X, y, smoothing, degree=1):
    """Closed-form leave-one-out CV residuals for thin-plate spline
    interpolation.

    X : (N, D) float array — spatial coordinates
    y : (N,)  float array  — residual values
    smoothing : float     — regularisation parameter (λ)
    degree : int          — polynomial trend degree (0 or 1)

    Returns the LOO residual array (same shape as y).

    Uses the hat-matrix identity:

        A = K + λI
        M = [[A    P  ],
             [P.T  0  ]]
        H = K @ M_inv[:N, :N] + P @ M_inv[N:, :N]
        r_loo = (y - H @ y) / (1 - diag(H))

    where K is the unregularised TPS kernel (r² log r in 2-D)
    and P is the polynomial basis.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n = X.shape[0]
    if n < 3:
        return np.zeros_like(y)

    # Pairwise Euclidean distances
    diff = X[:, None, :] - X[None, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=-1))

    # Unregularised TPS kernel: phi(r) = r² * log(r)   (for r > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        K0 = r * r * np.log(r)
    K0 = np.where(np.isnan(K0), 0.0, K0)

    # Regularised kernel matrix for block system
    K = K0 + smoothing * np.eye(n, dtype=np.float64)

    if degree == 0:
        # Pure RBF, no polynomial trend
        K_inv = np.linalg.inv(K)
        coeffs = K_inv @ y
        diag_inv = np.diag(K_inv)
        denom = np.where(np.abs(diag_inv) < 1e-12, 1e-12, diag_inv)
        return coeffs / denom

    # Degree ≥ 1: polynomial trend P = [1, x₁, x₂, …]
    if degree >= 1:
        P = np.column_stack([np.ones(n, dtype=np.float64), X])
        if degree >= 2:
            # Add interaction + quadratic terms
            for col in range(X.shape[1]):
                P = np.column_stack([P, X[:, col] ** 2])
            if X.shape[1] >= 2:
                P = np.column_stack([P, X[:, 0] * X[:, 1]])

    n_poly = P.shape[1]

    # Block matrix  M = [[K, P], [P.T, 0]]
    M = np.zeros((n + n_poly, n + n_poly), dtype=np.float64)
    M[:n, :n] = K
    M[:n, n:] = P
    M[n:, :n] = P.T

    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        # Fall back to per-fold loop for near-singular matrix
        return _tps_loocv_fallback(X, y, smoothing, degree)

    M_inv_top_n = M_inv[:n, :n]
    M_inv_bottom_n = M_inv[n:, :n]

    # Hat matrix using the unregularised kernel K0
    H = K0 @ M_inv_top_n + P @ M_inv_bottom_n
    y_hat = H @ y
    diag_H = np.diag(H)
    denom = 1.0 - diag_H
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return (y - y_hat) / denom


def _tps_loocv_fallback(X, y, smoothing, degree):
    """Per-fold re-fit fallback for TPS LOO-CV when the block matrix
    is near-singular."""
    from scipy.interpolate import RBFInterpolator

    n = len(X)
    if degree >= 1:
        kernel_kwargs = {
            "kernel": "thin_plate_spline",
            "smoothing": smoothing,
            "degree": 1,
        }
    else:
        kernel_kwargs = {
            "kernel": "thin_plate_spline",
            "smoothing": smoothing,
            "degree": 0,
        }

    errs = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        try:
            tps = RBFInterpolator(X[mask], y[mask], **kernel_kwargs)
            errs[i] = float(y[i] - tps(X[i : i + 1])[0])
        except Exception:
            errs[i] = y[i]
    return errs
