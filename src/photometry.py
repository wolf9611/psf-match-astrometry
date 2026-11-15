"""
Photometric extraction using aperture and PSF-fitting methods.
"""

import warnings

import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import Table
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.background import LocalBackground, MMMBackground
from photutils.psf import PSFPhotometry

from src.config import CATALOGS_DIR
from src.gpu import Backend
from src.logger import get_logger

logger = get_logger("photometry")


class Photometry:
    """
    Executes stellar photometry and saves resulting coordinates.
    """

    def __init__(
        self,
        image_clean,
        sources,
        estimated_fwhm,
        psf_models,
        wcs,
        catalogs_dir=CATALOGS_DIR,
    ):
        self.image_clean = image_clean
        self.sources = sources
        self.estimated_fwhm = estimated_fwhm
        self.psf_models = psf_models
        self.wcs = wcs
        self.catalogs_dir = catalogs_dir
        self.photometry_results = {}

    def perform_photometry(self, methods=None):
        """
        Runs the specified photometry methods (aperture, moffat, empirical).
        """
        if methods is None:
            methods = ["moffat", "empirical", "aperture"]

        for method in methods:
            if method == "aperture":
                self._aperture_photometry()
            elif method in self.psf_models and self.psf_models[method] is not None:
                self._psf_photometry(method)

        for method in self.photometry_results:
            self._save_coordinates(method)
        return self.photometry_results

    def _psf_photometry(self, method):
        psf_model = self.psf_models[method]
        init_params = Table()
        init_params["x_0"] = self.sources["x_centroid"]
        init_params["y_0"] = self.sources["y_centroid"]
        init_params["flux_0"] = self.sources["peak"] * (
            2.0 * np.pi * (self.estimated_fwhm / 2.355) ** 2
        )

        fit_hw = max(5, int(np.ceil(2.5 * self.estimated_fwhm)))
        fit_size = 2 * fit_hw + 1
        bkg_inner = max(5, int(np.ceil(2.5 * self.estimated_fwhm)))
        bkg_outer = max(bkg_inner + 5, int(np.ceil(5.0 * self.estimated_fwhm)))

        try:
            # photutils 3.0+ API uses local_bkg_estimator; fall back to
            # localbkg_estimator for older versions.
            common_kwargs = dict(
                psf_model=psf_model,
                fit_shape=(fit_size, fit_size),
                fitter=LevMarLSQFitter(),
            )
            local_bkg = LocalBackground(
                inner_radius=bkg_inner,
                outer_radius=bkg_outer,
                bkg_estimator=MMMBackground(),
            )
            try:
                photometry = PSFPhotometry(
                    **common_kwargs, local_bkg_estimator=local_bkg
                )
            except TypeError:
                photometry = PSFPhotometry(
                    **common_kwargs, localbkg_estimator=local_bkg
                )
            # Suppress photutils convergence warnings; outliers are handled
            # by downstream sigma-clipping.
            import logging as _logging

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _logging.getLogger("photutils.psf._components").setLevel(_logging.ERROR)
                result = photometry(data=self.image_clean, init_params=init_params)

            for old, new in [("x_fit", "x"), ("y_fit", "y"), ("flux_fit", "flux")]:
                if old in result.colnames:
                    result.rename_column(old, new)
            self.photometry_results[method] = result
            logger.debug(
                "PSF photometry method=%s completed on CPU (photutils)", method
            )
        except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
            print(f"Error in {method} PSF photometry: {exc}")

    def _aperture_photometry(self):
        aperture_radius = max(3.0, 1.5 * self.estimated_fwhm)
        bkg_inner = max(aperture_radius + 2, 3.0 * self.estimated_fwhm)
        bkg_outer = max(bkg_inner + 5, 5.0 * self.estimated_fwhm)

        positions = list(zip(self.sources["x_centroid"], self.sources["y_centroid"]))

        # Try the batched GPU aperture photometry first; the dispatcher
        # falls back to the photutils CPU implementation when no GPU is
        # available, the catalog is small, or the GPU path raises.
        gpu_used = False
        try:
            backend = Backend.instance()
            if backend.stage_enabled("photometry"):
                positions_arr = np.asarray(positions, dtype=np.float64)
                result = backend.batched_aperture_photometry(
                    self.image_clean,
                    positions_arr,
                    radius=aperture_radius,
                    annulus_inner=bkg_inner,
                    annulus_outer=bkg_outer,
                )
                ap_area = float(np.pi * aperture_radius**2)
                flux = result["flux"] - result["bkg_per_pixel"] * ap_area
                phot_table = Table()
                phot_table["x"] = np.asarray(self.sources["x_centroid"])
                phot_table["y"] = np.asarray(self.sources["y_centroid"])
                phot_table["aperture_sum"] = result["flux"]
                phot_table["flux"] = flux
                self.photometry_results["aperture"] = phot_table
                gpu_used = True
                logger.debug("Aperture photometry ran on device: %s", backend.dm.device)
        except Exception as exc:
            logger.debug("GPU aperture photometry skipped (%s); using CPU", exc)

        if gpu_used:
            return

        apertures = CircularAperture(positions, r=aperture_radius)
        annuli = CircularAnnulus(positions, r_in=bkg_inner, r_out=bkg_outer)

        phot_table = aperture_photometry(self.image_clean, apertures)
        bkg_table = aperture_photometry(self.image_clean, annuli)

        bkg_per_pixel = bkg_table["aperture_sum"] / annuli.area
        bkg_in_aperture = bkg_per_pixel * apertures.area

        phot_table["x"] = self.sources["x_centroid"]
        phot_table["y"] = self.sources["y_centroid"]
        phot_table["flux"] = phot_table["aperture_sum"] - bkg_in_aperture
        self.photometry_results["aperture"] = phot_table

    def _save_coordinates(self, method):
        if method not in self.photometry_results:
            return
        result = self.photometry_results[method]
        filename = self.catalogs_dir / f"xy_radec_{method}.txt"
        try:
            ra, dec = self.wcs.all_pix2world(result["x"], result["y"], 0)
            with open(filename, "w", encoding="utf-8") as f:
                f.writelines(
                    f"{ra_val:12.6f} {dec[i]:12.6f}\n" for i, ra_val in enumerate(ra)
                )
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            print(f"Error saving coordinates for {method}: {exc}")
