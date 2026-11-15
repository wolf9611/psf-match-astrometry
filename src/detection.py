"""
Source detection using DAOStarFinder with iterative FWHM estimation.
"""

import numpy as np
import pandas as pd
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.optimize import curve_fit

from src.gpu import Backend
from src.logger import get_logger

logger = get_logger("detection")


class SourceDetector:
    """
    Identifies stellar sources and estimates characteristic PSF FWHM.
    """

    def __init__(self, image_data, wcs=None, pixel_scale_arcsec=None):
        self.image_data = image_data
        self.wcs = wcs
        self.pixel_scale_arcsec = pixel_scale_arcsec

        # Estimate background.  Try the GPU-accelerated sigma-clipped
        # stats first; the dispatcher falls back to CPU for small images
        # or when no GPU is available.
        try:
            backend = Backend.instance()
            _, median, std = backend.sigma_clipped_stats(self.image_data, sigma=3.0)
            logger.debug(
                "Background estimation used backend device: %s", backend.dm.device
            )
        except Exception:
            _, median, std = sigma_clipped_stats(self.image_data, sigma=3.0)
        self.bkg_level = median
        self.bkg_rms = std
        self.image_clean = self.image_data - self.bkg_level
        logger.info(
            "Background estimation: median=%.4f ADU, rms=%.4f ADU (sigma-clipped 3σ)",
            self.bkg_level,
            self.bkg_rms,
        )

        self.sources = None
        # Initial FWHM scaled from pixel scale to match ~1 arcsec seeing
        if pixel_scale_arcsec is not None and pixel_scale_arcsec > 0:
            self.estimated_fwhm = max(2.0, 1.0 / pixel_scale_arcsec)  # ~1" seeing
        else:
            self.estimated_fwhm = 3.0

    def detect_sources(self, fwhm_initial=None, threshold_sigma=5.0):
        """
        Runs iterative source detection to converge on a characteristic FWHM.
        """
        if fwhm_initial is None:
            fwhm_initial = self.estimated_fwhm
        current_fwhm = fwhm_initial
        max_iterations = 3

        for iteration in range(max_iterations):
            threshold_val = threshold_sigma * self.bkg_rms
            # Try photutils 3.0+ API first, fall back to legacy keywords
            try:
                daofind = DAOStarFinder(
                    fwhm=current_fwhm,
                    threshold=threshold_val,
                    sharpness_range=(0.3, 0.9),
                    roundness_range=(-0.5, 0.5),
                )
            except TypeError:
                daofind = DAOStarFinder(
                    fwhm=current_fwhm,
                    threshold=threshold_val,
                    sharplo=0.3,
                    sharphi=0.9,
                    roundlo=-0.5,
                    roundhi=0.5,
                )

            self.sources = daofind(self.image_clean)
            if self.sources is None or len(self.sources) < 5:
                logger.warning(
                    "iter %d: only %s sources at FWHM=%.2f, threshold=%.2f ADU",
                    iteration + 1,
                    "0" if self.sources is None else len(self.sources),
                    current_fwhm,
                    threshold_val,
                )
                if iteration == 0:
                    return False
                break

            logger.info(
                "iter %d: FWHM=%.2f px, threshold=%.2f ADU (%.1fσ), N=%d sources",
                iteration + 1,
                current_fwhm,
                threshold_val,
                threshold_sigma,
                len(self.sources),
            )

            new_fwhm = self._estimate_fwhm_from_sources()
            fractional_change = abs(new_fwhm - current_fwhm) / current_fwhm
            current_fwhm = new_fwhm
            logger.info(
                "  → FWHM re-estimate: %.3f px (Δ=%.1f%%)",
                current_fwhm,
                fractional_change * 100.0,
            )

            if fractional_change < 0.05:
                logger.info("  → converged (Δ < 5%%)")
                break

        self.estimated_fwhm = current_fwhm
        return True

    def _estimate_fwhm_from_sources(self, n_stars=20):
        def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
            x, y = xy
            a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
                2 * sigma_y**2
            )
            b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (
                4 * sigma_y**2
            )
            c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
                2 * sigma_y**2
            )
            g = offset + amplitude * np.exp(
                -(
                    a * ((x - x0) ** 2)
                    + 2 * b * (x - x0) * (y - y0)
                    + c * ((y - y0) ** 2)
                )
            )
            return g.ravel()

        sorted_sources = self.sources[np.argsort(self.sources["peak"])][::-1]
        fwhm_values = []
        box_size = 15

        for i in range(min(n_stars, len(sorted_sources))):
            x_center = int(sorted_sources["x_centroid"][i])
            y_center = int(sorted_sources["y_centroid"][i])

            half = box_size // 2
            x_min = max(0, x_center - half)
            x_max = min(self.image_data.shape[1], x_center + half + 1)
            y_min = max(0, y_center - half)
            y_max = min(self.image_data.shape[0], y_center + half + 1)

            if x_max - x_min < 5 or y_max - y_min < 5:
                continue

            sub_data = self.image_clean[y_min:y_max, x_min:x_max]
            y_grid, x_grid = np.mgrid[: sub_data.shape[0], : sub_data.shape[1]]

            cx, cy = sub_data.shape[1] / 2, sub_data.shape[0] / 2
            initial_guess = (sub_data.max(), cx, cy, 2.0, 2.0, 0.0, sub_data.min())

            lower = (0, 0, 0, 0.3, 0.3, -np.pi, -np.inf)
            upper = (
                np.inf,
                sub_data.shape[1],
                sub_data.shape[0],
                half,
                half,
                np.pi,
                np.inf,
            )

            try:
                popt, _ = curve_fit(
                    gaussian_2d,
                    (x_grid, y_grid),
                    sub_data.ravel(),
                    p0=initial_guess,
                    bounds=(lower, upper),
                    maxfev=10000,
                )
                sigma_avg = (abs(popt[3]) + abs(popt[4])) / 2.0
                fwhm = 2.355 * sigma_avg

                if 1.0 < fwhm < 15.0:
                    fwhm_values.append(fwhm)
                    if len(fwhm_values) == 5:
                        box_size = max(15, int(np.ceil(5 * np.median(fwhm_values))) | 1)
            except (ValueError, TypeError, RuntimeError):
                continue

        if len(fwhm_values) < 3:
            return np.median(fwhm_values) if fwhm_values else 3.0

        arr = np.array(fwhm_values)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad > 0:
            good = np.abs(arr - med) < 2.5 * 1.4826 * mad
            arr = arr[good] if np.any(good) else arr

        return float(np.median(arr))

    def save_detected_sources(self, output_file):
        """
        Saves the detected source list to a CSV file.
        """
        save_data = {
            "id": range(1, len(self.sources) + 1),
            "x_pixel": self.sources["x_centroid"],
            "y_pixel": self.sources["y_centroid"],
            "peak_counts": self.sources["peak"],
            "mag_instrumental": -2.5 * np.log10(self.sources["peak"]),
        }

        if self.wcs is not None:
            try:
                ra_coords, dec_coords = self.wcs.all_pix2world(
                    self.sources["x_centroid"], self.sources["y_centroid"], 0
                )
                save_data["ra_deg"] = ra_coords
                save_data["dec_deg"] = dec_coords
            except (ValueError, TypeError, AttributeError):
                pass

        pd.DataFrame(save_data).to_csv(output_file, index=False)
