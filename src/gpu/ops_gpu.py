"""
CuPy-accelerated GPU implementations for the backend dispatcher.

Every function in this module accepts a ``cupy`` argument so the
caller can pass the probed CuPy module (allowing tests to swap in a
mock).  Inputs are NumPy arrays on the host; outputs are NumPy arrays
returned to the host.  Host-device transfer is minimised: each
function transfers once on entry and once on exit.

When CuPy is not installed, this module still imports cleanly; the
dispatcher only calls these functions after confirming CuPy is
available.
"""

from __future__ import annotations

import numpy as np


def _to_gpu(cupy, arr, dtype=np.float64):
    """Copies a NumPy array to the GPU, preserving dtype."""
    if isinstance(arr, cupy.ndarray):
        return arr.astype(dtype, copy=False)
    return cupy.asarray(arr, dtype=dtype)


def _to_cpu(cupy, arr):
    """Copies a CuPy array back to the host as a NumPy array."""
    if isinstance(arr, cupy.ndarray):
        return cupy.asnumpy(arr)
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Sigma-clipped statistics
# ---------------------------------------------------------------------------


def sigma_clipped_stats(data, sigma=3.0, maxiters=10, cupy=None):
    """GPU sigma-clipped mean/median/std via CuPy.

    Median is approximated by the 50th percentile (CuPy provides
    ``cupy.percentile``).
    """
    cp = cupy
    arr = _to_gpu(cp, data).ravel()
    mask = cp.isfinite(arr)
    if not bool(cp.any(mask)):
        return 0.0, 0.0, 0.0
    arr = arr[mask]
    for _ in range(maxiters):
        med = cp.median(arr)
        std = cp.std(arr)
        if float(std) == 0.0:
            break
        keep = cp.abs(arr - med) < sigma * std
        if bool(keep.all()) or int(keep.sum()) < 3:
            break
        arr = arr[keep]
    return float(cp.mean(arr)), float(cp.median(arr)), float(cp.std(arr))


# ---------------------------------------------------------------------------
# Batched aperture photometry
# ---------------------------------------------------------------------------


def batched_aperture_photometry(
    image,
    positions,
    radius,
    annulus_inner,
    annulus_outer,
    cupy=None,
    batch_size=4096,
):
    """GPU batched aperture + annulus photometry.

    Extracts per-source stamps, stacks them into a uniform tensor,
    and computes aperture + annulus fluxes in a single GPU pass
    per batch — eliminating per-source kernel-launch overhead.
    """
    cp = cupy
    img_gpu = _to_gpu(cp, image)
    ny, nx = img_gpu.shape
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    n = positions.shape[0]
    flux = np.zeros(n, dtype=np.float64)
    bkg_per_pixel = np.zeros(n, dtype=np.float64)

    r_max = max(radius, annulus_outer)
    r_max_int = int(np.ceil(r_max))
    stamp_sz = 2 * r_max_int + 1

    col_idx = cp.arange(stamp_sz, dtype=cp.float64)
    row_idx = cp.arange(stamp_sz, dtype=cp.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        B = end - start

        stamps_tensor = cp.full((B, stamp_sz, stamp_sz), cp.nan, dtype=cp.float64)
        center_x = cp.full(B, cp.nan, dtype=cp.float64)
        center_y = cp.full(B, cp.nan, dtype=cp.float64)

        for k in range(B):
            cx, cy = positions[start + k]
            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue
            x_lo = max(0, int(np.floor(cx - r_max_int)))
            x_hi = min(nx, int(np.ceil(cx + r_max_int)) + 1)
            y_lo = max(0, int(np.floor(cy - r_max_int)))
            y_hi = min(ny, int(np.ceil(cy + r_max_int)) + 1)
            if x_hi <= x_lo or y_hi <= y_lo:
                continue
            sub = img_gpu[y_lo:y_hi, x_lo:x_hi]
            h = int(y_hi - y_lo)
            w = int(x_hi - x_lo)
            stamps_tensor[k, :h, :w] = sub
            center_x[k] = cx - float(x_lo)
            center_y[k] = cy - float(y_lo)

        dx = col_idx[None, None, :] - center_x[:, None, None]
        dy = row_idx[None, :, None] - center_y[:, None, None]
        d2 = dx * dx + dy * dy

        ap_mask = d2 <= radius**2
        ann_mask = (d2 >= annulus_inner**2) & (d2 <= annulus_outer**2)

        ap_vals = stamps_tensor.copy()
        ap_vals[~ap_mask] = cp.nan
        flux[start:end] = _to_cpu(cp, cp.nansum(ap_vals, axis=(1, 2)))

        ann_vals = stamps_tensor.copy()
        ann_vals[~ann_mask] = cp.nan
        n_ann = cp.nansum(~cp.isnan(ann_vals), axis=(1, 2))
        ann_sum = cp.nansum(ann_vals, axis=(1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            ann_bkg = _to_cpu(cp, cp.where(n_ann > 0, ann_sum / n_ann, 0.0))
        bkg_per_pixel[start:end] = ann_bkg

    return {"flux": flux, "bkg_per_pixel": bkg_per_pixel}


# ---------------------------------------------------------------------------
# Ridge regression solve (CuPy)
# ---------------------------------------------------------------------------


def ridge_solve(X, y, alpha, weights=None, cupy=None):
    """GPU weighted Ridge regression via CuPy linear algebra."""
    cp = cupy
    Xg = _to_gpu(cp, X)
    yg = _to_gpu(cp, y).ravel()
    n_features = Xg.shape[1]
    if weights is None:
        XtX = Xg.T @ Xg
        Xty = Xg.T @ yg
    else:
        w = _to_gpu(cp, weights).ravel()
        Xw = Xg * w[:, None] ** 0.5
        XtX = Xw.T @ Xw
        Xty = Xw.T @ (yg * w**0.5)
    A = XtX + alpha * cp.eye(n_features, dtype=cp.float64)
    beta = cp.linalg.solve(A, Xty)
    return _to_cpu(cp, beta)


def closed_form_loocv_linear(X, y, alpha, weights=None, cupy=None):
    """GPU closed-form LOO-CV for weighted Ridge.

    Mirrors the CPU hat-matrix formula but uses CuPy linear algebra
    throughout.  Memory usage is O(N²); for very large N the caller
    should consider chunking.
    """
    cp = cupy
    Xg = _to_gpu(cp, X)
    yg = _to_gpu(cp, y).ravel()
    n = Xg.shape[0]
    n_features = Xg.shape[1]
    if weights is None:
        W = cp.eye(n, dtype=cp.float64)
    else:
        w = _to_gpu(cp, weights).ravel()
        W = cp.diag(w)
    A = Xg.T @ W @ Xg + alpha * cp.eye(n_features, dtype=cp.float64)
    A_inv = cp.linalg.inv(A)
    H = Xg @ A_inv @ Xg.T @ W
    y_hat = H @ yg
    diag_H = cp.diag(H)
    denom = 1.0 - diag_H
    denom = cp.where(cp.abs(denom) < 1e-12, 1e-12, denom)
    r = (yg - y_hat) / denom
    return _to_cpu(cp, r)


# ---------------------------------------------------------------------------
# Pairwise distances (CuPy)
# ---------------------------------------------------------------------------


def pairwise_distances(A, B, max_radius=None, cupy=None):
    """GPU pairwise Euclidean distance matrix."""
    cp = cupy
    Ag = _to_gpu(cp, A)
    Bg = _to_gpu(cp, B)
    if Ag.ndim == 1:
        Ag = Ag[:, None]
    if Bg.ndim == 1:
        Bg = Bg[:, None]
    # (N, M, D) difference tensor — may be large; fall back to chunked
    # computation when N*M*D exceeds a safe budget.
    N, D = Ag.shape
    M = Bg.shape[0]
    if N * M * D > 50_000_000:  # ~50M float64 entries
        return _pairwise_distances_chunked(Ag, Bg, max_radius, cp)
    diff = Ag[:, None, :] - Bg[None, :, :]
    d = cp.sqrt(cp.sum(diff * diff, axis=-1))
    if max_radius is not None:
        d = cp.where(d > max_radius, cp.inf, d)
    return _to_cpu(cp, d)


def _pairwise_distances_chunked(Ag, Bg, max_radius, cp):
    """Memory-safe chunked pairwise distance computation."""
    N = Ag.shape[0]
    M = Bg.shape[0]
    out = cp.empty((N, M), dtype=cp.float64)
    chunk = max(1, 50_000_000 // (M * Ag.shape[1]))
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        diff = Ag[start:end, None, :] - Bg[None, :, :]
        d = cp.sqrt(cp.sum(diff * diff, axis=-1))
        if max_radius is not None:
            d = cp.where(d > max_radius, cp.inf, d)
        out[start:end] = d
    return _to_cpu(cp, out)


# ---------------------------------------------------------------------------
# Radial profile with MAD clipping (CuPy)
# ---------------------------------------------------------------------------


def radial_profile_mad(psf_data, center, bin_width=0.5, max_radius=None, cupy=None):
    """GPU radial profile via per-bin MAD clipping.

    Pixels are classified into bins in one ``cp.digitize`` pass; per-bin
    MAD statistics then use vectorised CuPy operations — eliminating the
    per-bin spatial-distance mask recomputation.
    """
    cp = cupy
    psf = _to_gpu(cp, psf_data)
    ny, nx = psf.shape
    cy, cx = center
    y_idx, x_idx = cp.mgrid[:ny, :nx]
    y_idx = y_idx.astype(cp.float64)
    x_idx = x_idx.astype(cp.float64)
    r = cp.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    if max_radius is None:
        max_radius = float(min(cx, cy) * 0.95)
    bin_edges = cp.arange(0.0, max_radius + bin_width, bin_width)
    n_bins = int(bin_edges.size) - 1
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    profile = cp.full(n_bins, cp.nan, dtype=cp.float64)
    n_per_bin = cp.zeros(n_bins, dtype=cp.int64)

    r_flat = r.ravel()
    v_flat = psf.ravel()
    bin_idx = cp.digitize(r_flat, bin_edges) - 1
    bin_idx = cp.clip(bin_idx, 0, n_bins - 1)

    for i in range(n_bins):
        mask = bin_idx == i
        n = int(cp.count_nonzero(mask))
        if n == 0:
            continue
        n_per_bin[i] = n
        vals = v_flat[mask]
        med = cp.median(vals)
        mad = cp.median(cp.abs(vals - med))
        sigma_hat = 1.4826 * mad
        if float(sigma_hat) > 0:
            inliers = vals[cp.abs(vals - med) <= 3.0 * sigma_hat]
        else:
            inliers = vals
        profile[i] = cp.median(inliers) if inliers.size > 0 else med
    valid = cp.isfinite(profile)
    if not bool(valid.all()) and bool(valid.any()):
        idx = cp.arange(n_bins)
        profile = cp.where(
            valid,
            profile,
            cp.interp(idx, idx[valid], profile[valid]),
        )
    return _to_cpu(cp, bin_centers), _to_cpu(cp, profile), _to_cpu(cp, n_per_bin)


# ---------------------------------------------------------------------------
# Binned statistics (CuPy)
# ---------------------------------------------------------------------------


def bin_statistics(x, values, bin_edges, statistic="rms", cupy=None):
    """GPU binned statistic over arbitrary bin edges.

    Uses ``cp.digitize`` + ``cp.bincount`` for vectorised per-bin
    reductions (mean, std, rms, count).  Median falls back to a
    per-bin CuPy loop.
    """
    cp = cupy
    xg = _to_gpu(cp, x).ravel()
    vg = _to_gpu(cp, values).ravel()
    edges = _to_gpu(cp, bin_edges)
    n_bins = int(edges.size) - 1
    bin_idx = cp.digitize(xg, edges) - 1
    bin_idx = cp.clip(bin_idx, 0, n_bins - 1)

    out = cp.full(n_bins, cp.nan, dtype=cp.float64)

    if statistic == "median":
        for i in range(n_bins):
            mask = bin_idx == i
            n = int(cp.count_nonzero(mask))
            if n == 0:
                continue
            out[i] = cp.median(vg[mask])
    elif statistic in ("mean", "rms", "std", "count"):
        n_per = cp.bincount(bin_idx, minlength=n_bins).astype(cp.float64)
        mask_n = n_per > 0
        if statistic == "count":
            out[mask_n] = n_per[mask_n]
        else:
            s_per = cp.bincount(bin_idx, weights=vg, minlength=n_bins)
            s2_per = cp.bincount(bin_idx, weights=vg * vg, minlength=n_bins)
            if statistic == "mean":
                out[mask_n] = s_per[mask_n] / n_per[mask_n]
            elif statistic == "rms":
                out[mask_n] = cp.sqrt(s2_per[mask_n] / n_per[mask_n])
            elif statistic == "std":
                mean_sq = (s_per[mask_n] / n_per[mask_n]) ** 2
                var = s2_per[mask_n] / n_per[mask_n] - mean_sq
                out[mask_n] = cp.sqrt(cp.maximum(var, 0))
    else:
        raise ValueError(f"bin_statistics: unknown statistic {statistic!r}")
    return _to_cpu(cp, out)


# ---------------------------------------------------------------------------
# Closed-form TPS leave-one-out CV (CuPy)
# ---------------------------------------------------------------------------


def tps_loocv(X, y, smoothing, degree=1, cupy=None):
    """GPU closed-form leave-one-out CV residuals for thin-plate spline
    interpolation.  Mirrors the CPU reference but uses CuPy linear
    algebra throughout.

    X : (N, D) float array — spatial coordinates
    y : (N,)  float array  — residual values
    smoothing : float     — regularisation parameter (λ)
    degree : int          — polynomial trend degree (0 or 1)

    Returns the LOO residual array (same length as y).
    """
    cp = cupy
    Xg = _to_gpu(cp, X)
    yg = _to_gpu(cp, y).ravel()
    n = Xg.shape[0]
    if n < 3:
        return _to_cpu(cp, cp.zeros_like(yg))

    # Pairwise distances
    diff = Xg[:, None, :] - Xg[None, :, :]
    r = cp.sqrt(cp.sum(diff * diff, axis=-1))

    # Unregularised TPS kernel
    try:
        with cp.errstate(divide="ignore", invalid="ignore"):
            K0 = r * r * cp.log(r)
    except AttributeError:
        K0 = r * r * cp.log(r)
    K0 = cp.where(cp.isnan(K0), 0.0, K0)

    # Regularised kernel for block system
    K = K0 + smoothing * cp.eye(n, dtype=cp.float64)

    if degree == 0:
        K_inv = cp.linalg.inv(K)
        coeffs = K_inv @ yg
        diag_inv = cp.diag(K_inv)
        denom = cp.where(cp.abs(diag_inv) < 1e-12, 1e-12, diag_inv)
        return _to_cpu(cp, coeffs / denom)

    # Degree ≥ 1: polynomial trend P = [1, x₁, x₂, …]
    if degree >= 1:
        n_dim = Xg.shape[1]
        P_list = [cp.ones((n, 1), dtype=cp.float64), Xg]
        if degree >= 2:
            for col in range(n_dim):
                P_list.append((Xg[:, col : col + 1]) ** 2)
            if n_dim >= 2:
                P_list.append(Xg[:, 0:1] * Xg[:, 1:2])
        P = cp.hstack(P_list)

    n_poly = P.shape[1]

    # Block matrix M = [[K, P], [P.T, 0]]
    M_top = cp.hstack([K, P])
    M_bot = cp.hstack([P.T, cp.zeros((n_poly, n_poly), dtype=cp.float64)])
    M = cp.vstack([M_top, M_bot])

    try:
        M_inv = cp.linalg.inv(M)
    except Exception:
        from src.gpu.ops_cpu import tps_loocv as tps_loocv_cpu

        return tps_loocv_cpu(_to_cpu(cp, Xg), _to_cpu(cp, yg), smoothing, degree)

    M_inv_top_n = M_inv[:n, :n]
    M_inv_bottom_n = M_inv[n:, :n]

    # Hat matrix using the unregularised kernel K0
    H = K0 @ M_inv_top_n + P @ M_inv_bottom_n
    y_hat = H @ yg
    diag_H = cp.diag(H)
    denom = 1.0 - diag_H
    denom = cp.where(cp.abs(denom) < 1e-12, 1e-12, denom)
    return _to_cpu(cp, (yg - y_hat) / denom)
