"""
PSF construction including parametric models and empirical ePSF.
"""

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from photutils.psf import EPSFBuilder, MoffatPSF, extract_stars
from scipy.interpolate import PchipInterpolator

from src.config import MODELS_DIR, config
from src.gpu import Backend
from src.logger import get_logger
from src.plotting import analyze_radial_profiles, plot_psf_comparison

logger = get_logger("psf_modeling")


class PSFModeler:
    """
    Constructs and refines PSF models from stellar candidates.
    """

    def __init__(
        self,
        image_clean,
        sources,
        estimated_fwhm,
        saturation_level=60000.0,
        models_dir=MODELS_DIR,
        plots_dir=None,
    ):
        self.image_clean = image_clean
        self.sources = sources
        self.estimated_fwhm = estimated_fwhm
        self.saturation_level = saturation_level
        self.models_dir = models_dir
        self.plots_dir = plots_dir
        self.psf_models = {}

    def build_psf_models(self, psf_size=None, n_psf_stars=None):
        """
        Orchestrates the construction of Moffat and empirical models.
        """
        if n_psf_stars is None:
            n_psf_stars = getattr(config, "epsf_n_stars", 30)
        if psf_size is None:
            psf_size = max(25, int(np.ceil(8 * self.estimated_fwhm)) | 1)

        beta_moffat = 3.0
        alpha_moffat = self.estimated_fwhm / (
            2.0 * np.sqrt(2 ** (1.0 / beta_moffat) - 1)
        )
        self.psf_models["moffat"] = MoffatPSF(alpha=alpha_moffat, beta=beta_moffat)

        self.psf_models["empirical"] = self._build_empirical_psf(psf_size, n_psf_stars)
        return True

    def _build_empirical_psf(self, psf_size, n_psf_stars):
        sorted_sources = self.sources[np.argsort(self.sources["peak"])][::-1]
        h, w = self.image_clean.shape
        border = psf_size

        good_mask = (
            (sorted_sources["x_centroid"] > border)
            & (sorted_sources["x_centroid"] < w - border)
            & (sorted_sources["y_centroid"] > border)
            & (sorted_sources["y_centroid"] < h - border)
            & (sorted_sources["peak"] < self.saturation_level)
        )

        candidates = sorted_sources[good_mask]
        min_sep = 3.0 * self.estimated_fwhm
        selected_indices = []
        selected_x = []
        selected_y = []

        for i, (cx, cy) in enumerate(
            zip(candidates["x_centroid"], candidates["y_centroid"])
        ):
            if selected_x:
                dists = np.sqrt(
                    (np.array(selected_x) - cx) ** 2 + (np.array(selected_y) - cy) ** 2
                )
                if np.min(dists) < min_sep:
                    continue
            selected_indices.append(i)
            selected_x.append(cx)
            selected_y.append(cy)
            if len(selected_indices) >= n_psf_stars:
                break

        psf_stars = candidates[selected_indices].copy()
        # photutils >= 3.0 uses ``x_centroid``; older versions use ``xcentroid``
        for legacy, modern in (
            ("xcentroid", "x_centroid"),
            ("ycentroid", "y_centroid"),
        ):
            if legacy in psf_stars.colnames and modern not in psf_stars.colnames:
                psf_stars.rename_column(legacy, modern)
        if "x_centroid" in psf_stars.colnames:
            psf_stars.rename_column("x_centroid", "x")
        if "y_centroid" in psf_stars.colnames:
            psf_stars.rename_column("y_centroid", "y")

        if len(psf_stars) < 5:
            return None

        analyze_radial_profiles(psf_stars, self.image_clean, plots_dir=self.plots_dir)

        try:
            nddata = NDData(data=self.image_clean)
            star_cutouts = extract_stars(nddata, psf_stars, size=psf_size)

            oversampling = getattr(config, "epsf_oversampling", 4)
            maxiters = getattr(config, "epsf_maxiters", 10)
            epsf_builder = EPSFBuilder(
                oversampling=oversampling, maxiters=maxiters, progress_bar=False
            )
            epsf_model, _ = epsf_builder(star_cutouts)
            logger.info(
                "ePSF built: %d stars, oversampling=%d, maxiters=%d, shape=%s",
                len(psf_stars),
                oversampling,
                maxiters,
                epsf_model.data.shape,
            )

            apply_homog = getattr(config, "apply_psf_homogenization", True)
            if apply_homog:
                epsf_homogenized = self._apply_azimuthal_homogenization(epsf_model)
                plot_psf_comparison(
                    epsf_model, epsf_homogenized, plots_dir=self.plots_dir
                )
                logger.info("Applied azimuthal homogenization to ePSF")
            else:
                epsf_homogenized = epsf_model
                plot_psf_comparison(
                    epsf_model, epsf_homogenized, plots_dir=self.plots_dir
                )
                logger.info(
                    "Skipped azimuthal homogenization (preserves PSF asymmetry)"
                )

            fits.writeto(
                self.models_dir / "psf_model_empirical_original.fits",
                epsf_model.data,
                overwrite=True,
            )
            fits.writeto(
                self.models_dir / "psf_model_empirical.fits",
                epsf_homogenized.data,
                overwrite=True,
            )

            epsf_model.data = epsf_homogenized.data
            return epsf_model

        except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
            logger.error("ePSF construction failed: %s", exc)
            return None

    def _apply_azimuthal_homogenization(self, epsf_model):
        psf_data = epsf_model.data.copy()
        ny, nx = psf_data.shape
        center_y, center_x = (ny - 1) / 2.0, (nx - 1) / 2.0
        y_idx, x_idx = np.indices((ny, nx))
        r = np.sqrt((x_idx - center_x) ** 2 + (y_idx - center_y) ** 2)
        max_radius = min(center_x, center_y) * 0.95

        bin_width = 0.5
        bin_edges = np.arange(0.0, max_radius + bin_width, bin_width)
        n_bins = len(bin_edges) - 1

        # GPU-accelerated MAD-clipped radial profile when available;
        # the dispatcher falls back to the CPU implementation
        # automatically for small PSF arrays or when no GPU is present.
        try:
            backend = Backend.instance()
            bin_centers, radial_profile, _ = backend.radial_profile_mad(
                psf_data,
                center=(center_y, center_x),
                bin_width=bin_width,
                max_radius=max_radius,
            )
            logger.debug("Radial profile built on device: %s", backend.dm.device)
        except Exception:
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            radial_profile = np.full(n_bins, np.nan)
            for i in range(n_bins):
                ann_mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
                if not np.any(ann_mask):
                    continue
                vals = psf_data[ann_mask]
                med = np.median(vals)
                mad = np.median(np.abs(vals - med))
                sigma_hat = 1.4826 * mad
                inliers = (
                    vals[np.abs(vals - med) <= 3.0 * sigma_hat]
                    if sigma_hat > 0
                    else vals
                )
                radial_profile[i] = np.median(inliers) if len(inliers) > 0 else med

        valid = np.isfinite(radial_profile)
        if not np.all(valid) and np.any(valid):
            idx = np.arange(n_bins)
            radial_profile[~valid] = np.interp(
                idx[~valid], idx[valid], radial_profile[valid]
            )

        radial_profile = np.maximum(radial_profile, 0.0)

        cy_int, cx_int = int(round(center_y)), int(round(center_x))
        peak_val = float(
            np.mean(
                psf_data[
                    max(0, cy_int - 1) : cy_int + 2, max(0, cx_int - 1) : cx_int + 2
                ]
            )
        )

        inside = bin_centers < max_radius
        bc = bin_centers[inside]
        rp = radial_profile[inside]

        r_knots = np.concatenate([[0.0], bc, [max_radius, max_radius * 1.05]])
        p_knots = np.concatenate([[peak_val], rp, [0.0, 0.0]])

        _, unique_idx = np.unique(r_knots, return_index=True)
        r_knots, p_knots = r_knots[unique_idx], p_knots[unique_idx]

        interp_func = PchipInterpolator(r_knots, p_knots, extrapolate=False)

        r_flat = r.ravel()
        in_bounds = r_flat <= max_radius * 1.05

        homo_flat = np.zeros(ny * nx)
        homo_flat[in_bounds] = interp_func(r_flat[in_bounds])
        homogenized = np.maximum(0.0, homo_flat.reshape(ny, nx))

        original_flux = np.sum(psf_data)
        if np.sum(homogenized) > 0:
            homogenized *= original_flux / np.sum(homogenized)

        homogenized_model = epsf_model.copy()
        homogenized_model.data = homogenized

        return homogenized_model
