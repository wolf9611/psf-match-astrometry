"""
Main orchestrator for the Unified Photometry and Astrometry Pipeline.
"""

import argparse
import glob
import os

import pandas as pd

from src.astrometry import Astrometry
from src.calibration import Calibration
from src.config import (
    CATALOGS_DIR,
    LOGS_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PLOTS_DIR,
    PRESETS,
    OutputDirs,
    config,
)
from src.data_io import DataIO
from src.detection import SourceDetector
from src.gpu import configure_backend, current_device
from src.gpu.batching import resolve_stage_toggles
from src.logger import get_logger, setup_logging
from src.photometry import Photometry
from src.plotting import visualize_fits_image
from src.psf_modeling import PSFModeler

logger = get_logger("main")


def build_parser():
    parser = argparse.ArgumentParser(
        description="PSF-match: unified photometry and astrometry pipeline"
    )
    parser.add_argument(
        "--preset",
        "-p",
        choices=sorted(PRESETS.keys()),
        help="Instrument preset (overrides default input paths and settings)",
    )
    parser.add_argument(
        "--instrument",
        "-i",
        help="Instrument name for output directory naming "
        "(defaults to auto-detected from FITS header)",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input FITS image (overrides config and preset)",
    )
    parser.add_argument(
        "--gaia-catalog",
        type=str,
        help="Path to Gaia reference catalog CSV (overrides config and preset)",
    )
    parser.add_argument(
        "--no-milliquas",
        action="store_true",
        help="Disable Milliquas quasar frame-tie",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip cleanup of old output files",
    )
    parser.add_argument(
        "--extension",
        type=int,
        default=0,
        help="FITS extension number to process (default: 0 = auto-detect)",
    )
    parser.add_argument(
        "--pm-mode",
        choices=["full", "damped", "none"],
        default="damped",
        help="Proper motion handling mode: 'full' (apply all PM), "
        "'damped' (scale down PM for G>=17, default), "
        "or 'none' (epoch positions only, no PM propagation)",
    )
    parser.add_argument(
        "--no-faint-stars",
        action="store_true",
        help="Exclude faint Gaia stars (G>=20) from matching by using strict "
        "RUWE<=1.4 and dropping 2-parameter solutions",
    )
    parser.add_argument(
        "--compare-faint-stars",
        action="store_true",
        help="Run the pipeline twice (with and without faint Gaia stars) and "
        "write a comparison report to the logs directory",
    )
    # --- Optional CUDA GPU support ---
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device: 'auto' (default, use CUDA when available), "
        "'cpu' (force CPU), or 'cuda' (force CUDA, fallback to CPU if "
        "unavailable)",
    )
    parser.add_argument(
        "--no-gpu-fallback",
        action="store_true",
        help="Disable silent GPU-to-CPU fallback (raise on GPU errors "
        "instead).  By default fallback is enabled.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic GPU execution (PyTorch cudnn.deterministic "
        "+ manual_seed).  Useful for reproducible bootstrap/holdout runs.",
    )
    parser.add_argument(
        "--image-precision",
        choices=["float32", "float64"],
        default="float64",
        help="Numerical precision for image arrays (default: float64 to "
        "preserve existing CPU behaviour).",
    )
    parser.add_argument(
        "--astrometry-precision",
        choices=["float32", "float64"],
        default="float64",
        help="Numerical precision for astrometric arrays (default: float64).",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=4096,
        help="Batch size for GPU-batched operations (aperture photometry, "
        "PSF fitting, ridge solves).  Default 4096.",
    )
    parser.add_argument(
        "--gpu-stage",
        action="append",
        metavar="STAGE=BOOL",
        default=[],
        help="Per-stage GPU toggle, e.g. --gpu-stage photometry=true or "
        "--gpu-stage detection=false.  May be repeated.  Recognised "
        "stages: detection, psf, photometry, systematic, advanced, "
        "enhancements, faint.",
    )
    parser.add_argument(
        "--gpu-disable-all",
        action="store_true",
        help="Force-disable GPU acceleration on every stage (equivalent to "
        "--device=cpu but keeps GPU libraries imported for warmup).",
    )
    parser.add_argument(
        "--gpu-timing",
        action="store_true",
        help="Log per-stage device selection and timing information.",
    )
    parser.add_argument(
        "--gpu-workers",
        type=int,
        default=1,
        help="Worker readiness for future multi-GPU MEF processing.  "
        "Values >1 reserve worker slots; the current implementation "
        "remains single-process.  Default 1.",
    )
    return parser


def apply_preset(name):
    """Apply a named preset's config overrides."""
    overrides = PRESETS[name]
    for key, value in overrides.items():
        object.__setattr__(config, key, value)
    print(f"[CONFIG] Applied preset '{name}'")


def cleanup_old_files(dirs: OutputDirs = None):
    """
    Removes artifacts from previous pipeline runs to ensure a clean state.
    """
    if dirs is None:
        dirs = OutputDirs(
            catalogs=CATALOGS_DIR,
            plots=PLOTS_DIR,
            models=MODELS_DIR,
            logs=LOGS_DIR,
        )
    patterns = [
        dirs.catalogs / "matched_catalog_*.csv",
        dirs.plots / "match_*.png",
        dirs.catalogs / "xy_radec_*.txt",
        dirs.plots / "psf_model_*.png",
        dirs.models / "psf_model_*.fits",
        dirs.plots / "radial_profiles_*.png",
        dirs.catalogs / "refined_*.csv",
        dirs.catalogs / "refined_*.txt",
        dirs.plots / "refined_*.png",
        dirs.models / "refined_*.fits",
        dirs.catalogs / "all_detected_sources.csv",
        dirs.logs / "match_statistics.txt",
        dirs.plots / "final_*.png",
        dirs.catalogs / "xy_radec.csv",
        dirs.catalogs / "final_matched_catalog.csv",
        dirs.plots / "rms.png",
        dirs.plots / "residuals_scatter.png",
        dirs.catalogs / "advanced_*.csv",
        dirs.logs / "advanced_*.txt",
        dirs.plots / "advanced_*.png",
        dirs.catalogs / "quasar_matches.csv",
        dirs.catalogs / "quasar_frame_tie.csv",
        dirs.plots / "image_preview.png",
        dirs.plots / "magnitude_stratification.png",
        dirs.plots / "holdout_cv.png",
        dirs.plots / "dcr_diagnostics.png",
        dirs.plots / "spatial_residual_map.png",
        dirs.plots / "ccd_edge_diagnostics.png",
        dirs.catalogs / "enhanced_calibrated_coordinates.csv",
        dirs.catalogs / "faint_star_local_anchor.csv",
        dirs.logs / "enhanced_telemetry_report.md",
    ]
    for pattern in patterns:
        for file in glob.glob(str(pattern)):
            try:
                os.remove(file)
            except OSError:
                pass


def main():
    """
    Executes the end-to-end photometry and astrometry pipeline.

    When --compare-faint-stars is given, runs the pipeline twice (with and
    without faint Gaia stars) and writes a comparison report to the logs
    directory of the "with-faint" run.
    """
    # --- Parse CLI arguments ---
    parser = build_parser()
    args = parser.parse_args()

    # Apply preset first (so CLI args can override individual fields)
    if args.preset:
        apply_preset(args.preset)

    # CLI overrides
    if args.image:
        object.__setattr__(config, "image_file", type(config.image_file)(args.image))
    if args.gaia_catalog:
        object.__setattr__(
            config,
            "gaia_catalog_file",
            type(config.gaia_catalog_file)(args.gaia_catalog),
        )
    if args.no_milliquas:
        object.__setattr__(config, "use_milliquas", False)
    if args.no_clean:
        object.__setattr__(config, "clean_old_outputs", False)
    if args.extension is not None:
        object.__setattr__(config, "extension", args.extension)
    if args.pm_mode:
        object.__setattr__(config, "pm_mode", args.pm_mode)
    if args.no_faint_stars:
        object.__setattr__(config, "include_faint_stars", False)

    # --- Apply GPU-related CLI overrides ---
    object.__setattr__(config, "device", args.device)
    object.__setattr__(config, "gpu_fallback", not args.no_gpu_fallback)
    object.__setattr__(config, "deterministic", args.deterministic)
    object.__setattr__(config, "image_precision", args.image_precision)
    object.__setattr__(config, "astrometry_precision", args.astrometry_precision)
    object.__setattr__(config, "gpu_batch_size", args.gpu_batch_size)
    object.__setattr__(config, "gpu_enable_timing", args.gpu_timing)
    object.__setattr__(config, "gpu_workers", max(1, args.gpu_workers))

    # Parse --gpu-stage STAGE=BOOL pairs into a dict
    stage_overrides: dict[str, bool] = {}
    for spec in args.gpu_stage:
        if "=" not in spec:
            print(f"[WARNING] Ignoring malformed --gpu-stage value: {spec!r}")
            continue
        stage_name, stage_val = spec.split("=", 1)
        stage_name = stage_name.strip().lower()
        stage_val = stage_val.strip().lower()
        if stage_val in ("1", "true", "yes", "on"):
            stage_overrides[stage_name] = True
        elif stage_val in ("0", "false", "no", "off"):
            stage_overrides[stage_name] = False
        else:
            print(f"[WARNING] Ignoring malformed --gpu-stage value: {spec!r}")
    object.__setattr__(config, "gpu_stages", stage_overrides)

    # Configure the GPU backend singleton (re-creates DeviceManager + Backend)
    stage_toggles = resolve_stage_toggles(
        explicit=stage_overrides, disable_all=args.gpu_disable_all
    )
    configure_backend(
        device=config.device,
        deterministic=config.deterministic,
        batch_size=config.gpu_batch_size,
        stage_toggles=stage_toggles,
        enable_timing=config.gpu_enable_timing,
        min_gpu_elements=config.gpu_min_elements,
        min_gpu_stars=config.gpu_min_stars,
    )
    print(
        f"  -> GPU backend configured: device='{current_device()}', "
        f"batch_size={config.gpu_batch_size}, deterministic={config.deterministic}, "
        f"gpu_stages={stage_toggles}"
    )

    override_instrument = args.instrument

    if args.compare_faint_stars:
        _run_comparison_mode(override_instrument)
    elif args.no_faint_stars:
        _execute_pipeline(override_instrument, subdir=f"nofaint_pm_{args.pm_mode}")
    else:
        _execute_pipeline(override_instrument, subdir=f"faint_pm_{args.pm_mode}")


def _execute_pipeline(override_instrument=None, subdir=""):
    """
    Self-contained pipeline run (Stages 0-9).

    Loads the FITS image, runs all stages, and writes outputs under
    ``output/<instrument>/<subdir>/``.  Returns a dict with telemetry
    data for downstream comparison reporting.

    When ``config.process_all_mef_extensions`` is True and the input
    FITS file contains multiple image-like extensions, the function
    iterates over every extension (via ``plan_mef_extensions``) and
    returns a list of per-extension telemetry dicts.  Otherwise a
    single run is performed and a single dict is returned.
    """
    # --- Multi-Extension FITS (MEF) orchestration ---
    # When the user enables MEF processing, build a plan over all
    # image-like extensions and run the single-extension pipeline once
    # per extension.  Outputs land under
    # ``output/<instrument>/<subdir>/ext_<name>/`` so each extension
    # has its own catalog/plots/logs/models tree.
    if config.process_all_mef_extensions:
        from astropy.io import fits

        from src.gpu.batching import plan_mef_extensions

        try:
            with fits.open(config.image_file) as hdul:
                plan = plan_mef_extensions(
                    hdul,
                    process_all=True,
                    requested_extension=config.extension,
                )
        except Exception as exc:
            print(
                f"  [WARNING] MEF planning failed ({exc}); falling back to single extension."
            )
            plan = None

        if plan is not None and len(plan.extension_indices) > 1:
            telemetry_list = []
            for ext_idx, ext_name in zip(plan.extension_indices, plan.extension_names):
                print(
                    f"\n{'=' * 80}\n  MEF extension {ext_idx} ({ext_name})\n{'=' * 80}"
                )
                # Temporarily override the requested extension and
                # redirect outputs to a per-extension subdirectory.
                saved_ext = config.extension
                object.__setattr__(config, "extension", ext_idx)
                ext_subdir = f"{subdir}/ext_{ext_name}" if subdir else f"ext_{ext_name}"
                telemetry = _execute_single_extension_pipeline(
                    override_instrument, subdir=ext_subdir
                )
                object.__setattr__(config, "extension", saved_ext)
                if isinstance(telemetry, dict):
                    telemetry["extension_index"] = ext_idx
                    telemetry["extension_name"] = ext_name
                    telemetry_list.append(telemetry)
            return telemetry_list if telemetry_list else {}

    return _execute_single_extension_pipeline(override_instrument, subdir=subdir)


def _execute_single_extension_pipeline(override_instrument=None, subdir=""):
    """
    Single-extension pipeline run (Stages 0-9).

    Loads the FITS image, runs all stages, and writes outputs under
    ``output/<instrument>/<subdir>/``.  Returns a dict with telemetry
    data for downstream comparison reporting.
    """
    data_io = DataIO()
    print(f"  -> Loading FITS image from: {config.image_file}")
    data_io.load_fits(config.image_file, config.extension)
    print(f"  -> Successfully loaded image. Shape: {data_io.image_data.shape}")

    instrument_name = override_instrument or data_io.instrument
    if not instrument_name or instrument_name.upper() == "UNKNOWN":
        parent = config.image_file.parent.name
        instrument_name = parent if parent.lower() not in ("input", "") else "default"
    base = OUTPUT_DIR / instrument_name.lower().replace(" ", "_")
    if subdir:
        dirs = OutputDirs(
            catalogs=base / subdir / "catalogs",
            plots=base / subdir / "plots",
            models=base / subdir / "models",
            logs=base / subdir / "logs",
        )
    else:
        dirs = OutputDirs(
            catalogs=base / "catalogs",
            plots=base / "plots",
            models=base / "models",
            logs=base / "logs",
        )
    rms_dir = base / "rms"
    for d in [dirs.catalogs, dirs.plots, dirs.models, dirs.logs, rms_dir]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"  -> Instrument: {instrument_name} | Output: {dirs.catalogs.parent}")

    # Reconfigure logging to instrument-specific directory
    setup_logging(dirs.logs)

    global logger
    logger = get_logger("main")

    if config.clean_old_outputs:
        print("[SETUP] Cleaning up old output files from previous runs...")
        cleanup_old_files(dirs)

    if config.process_xy_only:
        print(
            "[INFO] Processing XY file only not fully re-implemented in this script, "
            "please run full pipeline."
        )
        return {}

    print("\n" + "-" * 80)
    print(" 0. LOADING DATA")
    print("-" * 80)

    print(f"  [INPUT] Primary Image: {config.image_file}")
    print(f"  [INPUT] Reference Catalog (Gaia): {config.gaia_catalog_file}")
    if config.use_milliquas:
        print(f"  [INPUT] Quasar Catalog (Milliquas): {config.milliquas_catalog_file}")

    print(
        f"  -> Instrument: {data_io.instrument} | Telescope: {data_io.telescope} | "
        f'Filter: {data_io.filter_name} | Pixel scale: {data_io.pixel_scale_arcsec:.4f}"/px'
    )
    if data_io.airmass:
        print(
            f'  -> Airmass: {data_io.airmass:.3f} | Seeing: {data_io.seeing_arcsec:.2f}"'
        )
    logger.info(
        "Pipeline run: image=%s | instrument=%s | pixel_scale=%.4f | "
        "exptime=%.1fs | airmass=%s | seeing=%s",
        config.image_file.name,
        data_io.instrument,
        data_io.pixel_scale_arcsec,
        data_io.exptime or 0,
        data_io.airmass,
        data_io.seeing_arcsec,
    )

    print(f"  -> Loading Gaia catalog from: {config.gaia_catalog_file}")
    gaia_catalog = data_io.load_gaia_catalog(config.gaia_catalog_file)
    print("  -> Gaia catalog loaded successfully.")

    try:
        gaia_x, gaia_y = data_io.wcs.all_world2pix(
            gaia_catalog["ra"], gaia_catalog["dec"], 0
        )
        ny, nx = data_io.image_data.shape
        mask = (gaia_x >= 0) & (gaia_x < nx) & (gaia_y >= 0) & (gaia_y < ny)
        gaia_sources_plot = {"x": gaia_x[mask], "y": gaia_y[mask]}
    except Exception:
        gaia_sources_plot = None

    if config.use_milliquas:
        print(
            f"  -> Loading Milliquas quasar catalog from: {config.milliquas_catalog_file}"
        )
        milliquas_catalog = data_io.load_milliquas_catalog(
            config.milliquas_catalog_file
        )
        print("  -> Milliquas catalog loaded successfully.")
    else:
        milliquas_catalog = None
        print("  -> Milliquas quasar frame-tie is disabled.")

    print("  -> Generating initial FITS image preview (image_preview_initial.png)...")
    visualize_fits_image(
        data_io.image_data, file_path="image_preview_initial.png", plots_dir=dirs.plots
    )

    print("\n" + "-" * 80)
    print(" 1. SOURCE DETECTION")
    print("-" * 80)
    print("  -> Initializing SourceDetector and estimating local background...")
    detector = SourceDetector(
        data_io.image_data, data_io.wcs, data_io.pixel_scale_arcsec
    )
    print(
        f"  -> Background Median: {detector.bkg_level:.3f} ADU | RMS: {detector.bkg_rms:.3f} ADU"
    )

    print(
        "  -> Running DAOStarFinder iterative detection to estimate characteristic FWHM..."
    )
    if not detector.detect_sources():
        print("  [ERROR] Failed to detect sufficient sources. Exiting pipeline.")
        return {}
    print(f"  -> Detected {len(detector.sources)} isolated sources.")
    print(f"  -> Converged FWHM estimate: {detector.estimated_fwhm:.2f} pixels.")

    print(
        f"  -> Archiving detected sources to: {dirs.catalogs / 'all_detected_sources.csv'}"
    )
    detector.save_detected_sources(dirs.catalogs / "all_detected_sources.csv")

    print("  -> Generating combined GAIA and Detection preview (image_preview.png)...")
    visualize_fits_image(
        data_io.image_data,
        sources=detector.sources,
        gaia_sources=gaia_sources_plot,
        file_path="image_preview.png",
        plots_dir=dirs.plots,
    )

    print("\n" + "-" * 80)
    print(" 2. PSF MODELING")
    print("-" * 80)
    print(
        f"  -> Initializing PSF construction (Saturation Level: {data_io.saturation_level} ADU)..."
    )
    psf_modeler = PSFModeler(
        detector.image_clean,
        detector.sources,
        detector.estimated_fwhm,
        data_io.saturation_level,
        models_dir=dirs.models,
        plots_dir=dirs.plots,
    )
    print("  -> Building parametric (Moffat) and empirical (ePSF) models...")
    if not psf_modeler.build_psf_models():
        print("  [ERROR] Failed to construct PSF models. Exiting pipeline.")
        return {}
    print(f"  -> Successfully constructed and saved PSF models to {dirs.models}/.")

    print("\n" + "-" * 80)
    print(" 3. PHOTOMETRY")
    print("-" * 80)
    print(
        f"  -> Performing photometry using estimators: {', '.join(config.photometry_methods)}"
    )
    photometry = Photometry(
        detector.image_clean,
        detector.sources,
        detector.estimated_fwhm,
        psf_modeler.psf_models,
        data_io.wcs,
        catalogs_dir=dirs.catalogs,
    )
    phot_results = photometry.perform_photometry(config.photometry_methods)
    for method, res in phot_results.items():
        print(f"  -> Method '{method}' extracted {len(res)} sources successfully.")

    print("\n" + "-" * 80)
    print(" 4. CATALOG MATCHING")
    print("-" * 80)
    print("  -> Initializing Astrometry module...")
    astrometry = Astrometry(
        gaia_catalog,
        data_io.obstime,
        catalogs_dir=dirs.catalogs,
        logs_dir=dirs.logs,
        plots_dir=dirs.plots,
    )

    print(
        f"  -> Performing symmetric cross-matching with proper-motion corrected "
        f'Gaia coordinates (Max sep: {config.max_separation_arcsec}")...'
    )
    astrometry.match_catalogs(phot_results, config.max_separation_arcsec)
    print(
        f"  -> Match statistics and residual plots saved to {dirs.logs} and {dirs.plots}."
    )

    print("\n" + "-" * 80)
    print(" 5. SYSTEMATIC ASTROMETRIC CORRECTIONS")
    print("-" * 80)
    print("  -> Preparing Gaia epoch coordinates for systematic modelling...")
    calibration = Calibration(
        data_io.wcs,
        astrometry.get_proper_motion_corrected_gaia(),
        detector.estimated_fwhm,
        data_io.image_data.shape,
        detector.sources,
        milliquas_catalog,
        catalogs_dir=dirs.catalogs,
        logs_dir=dirs.logs,
        plots_dir=dirs.plots,
    )

    if config.enable_systematic_correction:
        print(
            "  -> Applying geometric distortion (SIP) and magnitude-dependent corrections..."
        )
        final_matches = calibration.model_systematic_corrections(
            dirs.catalogs / "matched_catalog_empirical.csv"
        )
        if final_matches is not None:
            print(
                f"  -> Systematic correction converged. Refined {len(final_matches)} pairs."
            )
            print(
                "  -> Converting sub-pixel centroids to refined RA/DEC coordinates..."
            )
            calibration.convert_xy_to_radec()
            print(
                "  -> Performing final validation match against the Gaia reference frame..."
            )
            calibration.match_xy_radec_with_gaia()
            print(f"  -> Final astrometric distribution plots saved to {dirs.plots}/.")
        else:
            print("  [WARNING] Systematic correction returned no valid pairs.")
    else:
        print("  -> Systematic corrections bypassed (disabled in config).")

    enhancement_telemetry = {}
    if config.enable_advanced_calibration:
        print("\n" + "-" * 80)
        print(" 6. ADVANCED CALIBRATION")
        print("-" * 80)
        print("  -> Initializing advanced spatially-correlated residual correction...")
        print("  -> Executing: Sigma Clip + Weighted Polynomial -> TPS")
        if calibration.run_advanced_calibration():
            print("  -> Advanced calibration succeeded. Final coordinates anchored.")
            print(
                f"  -> Advanced reports and figures written to {dirs.logs}/ and {dirs.plots}/."
            )

            print("\n" + "-" * 80)
            print(" 7. RIGOROUS SCIENTIFIC ENHANCEMENTS")
            print("-" * 80)
            enhancement_telemetry = run_rigorous_enhancements(
                calibration, data_io, dirs
            )

            # Write final RMS to dedicated folder
            _final_rms = enhancement_telemetry.get("final_rms_mas")
            _final_n = (
                len(calibration.matched_data)
                if calibration.matched_data is not None
                else 0
            )
            if _final_rms is not None:
                _rms_summary = pd.DataFrame(
                    [
                        {
                            "subdir": subdir if subdir else "default",
                            "pm_mode": config.pm_mode,
                            "include_faint_stars": config.include_faint_stars,
                            "final_rms_mas": _final_rms,
                            "n_stars": _final_n,
                        }
                    ]
                )
                _rms_file = rms_dir / f"final_rms_{subdir.replace('/', '_')}.csv"
                _rms_summary.to_csv(_rms_file, index=False)
                print(f"  -> Final RMS result written to: {_rms_file}")

            print("\n" + "-" * 80)
            print(" 8. INSTRUMENTAL PRECISION ISOLATION")
            print("-" * 80)
            isolate_instrumental_precision(calibration, data_io, dirs.logs)

            if config.enable_faint_precision_analysis:
                print("\n" + "-" * 80)
                print(" 9. FAINT-STAR PRECISION ANALYSIS")
                print("-" * 80)
                run_faint_precision_analysis(calibration, data_io, dirs)
        else:
            print(
                "  [WARNING] Advanced calibration encountered an issue or insufficient pairs."
            )
    else:
        print(
            "  -> Advanced systematic residual calibration bypassed (disabled in config)."
        )

    print("\n" + "==" * 40)
    print("  INTEGRATED PIPELINE COMPLETED SUCCESSFULLY")
    print("==" * 40 + "\n")

    return {
        "dirs": dirs,
        "calibration": calibration,
        "data_io": data_io,
        "holdout_cv": enhancement_telemetry.get("holdout_cv"),
        "mag_strat": enhancement_telemetry.get("magnitude_stratification"),
        "final_rms_mas": enhancement_telemetry.get("final_rms_mas", float("nan")),
        "instrument": instrument_name,
    }


def _run_comparison_mode(override_instrument=None):
    """
    Runs the pipeline twice: once without faint stars, once with, then
    writes a comparison report to ``output/<instrument>/comparison/logs/``.
    """
    object.__setattr__(config, "include_faint_stars", False)
    print("\n" + "==" * 40)
    print("  RUN WITHOUT FAINT STARS")
    print("==" * 40 + "\n")
    nofaint = _execute_pipeline(
        override_instrument, subdir=f"comparison/exclude_faint_pm_{config.pm_mode}"
    )

    object.__setattr__(config, "include_faint_stars", True)
    print("\n" + "==" * 40)
    print("  RUN WITH FAINT STARS")
    print("==" * 40 + "\n")
    faint = _execute_pipeline(
        override_instrument, subdir=f"comparison/include_faint_pm_{config.pm_mode}"
    )

    _write_faint_comparison_report(nofaint, faint)


def _write_faint_comparison_report(nofaint, faint):
    """
    Writes a markdown report comparing the two pipeline runs to
    ``output/<instrument>/comparison/logs/``.
    """
    import numpy as np

    dirs = faint.get("dirs")
    if dirs is None:
        return
    # dirs.catalogs = output/<inst>/comparison/include_faint/catalogs
    # We want      = output/<inst>/comparison/logs
    comp_root = dirs.catalogs.parent.parent  # output/<inst>/comparison
    comp_logs = comp_root / "logs"
    comp_logs.mkdir(parents=True, exist_ok=True)
    report_path = comp_logs / "faint_star_comparison_report.md"

    md_nofaint = nofaint.get("calibration")
    md_faint = faint.get("calibration")

    n_nofaint = len(md_nofaint.matched_data) if md_nofaint is not None else 0
    n_faint = len(md_faint.matched_data) if md_faint is not None else 0

    rms_nofaint = nofaint.get("final_rms_mas", float("nan"))
    rms_faint = faint.get("final_rms_mas", float("nan"))

    ho_nofaint = (nofaint.get("holdout_cv") or {}).get("test_rms_mean_mas")
    ho_faint = (faint.get("holdout_cv") or {}).get("test_rms_mean_mas")

    m_nofaint = nofaint.get("mag_strat") or {}
    m_faint = faint.get("mag_strat") or {}

    bins_nofaint = m_nofaint.get("magnitude_bins", [])
    bins_faint = m_faint.get("magnitude_bins", [])

    base_inst = nofaint.get("instrument", "?")
    lines = [
        "# Faint-Star Comparison Report",
        "",
        f"**Instrument:** {base_inst}",
        f"**Output:** `{comp_root}`",
        "",
        "## Summary",
        "",
        "| Metric | Without G≥21 | With G≥21 | Delta |",
        "|--------|-------------|-----------|-------|",
        f"| Matched stars | {n_nofaint} | {n_faint} | +{n_faint - n_nofaint} |",
        f"| Final RMS (mas) | {rms_nofaint:.3f} | {rms_faint:.3f} | {rms_faint - rms_nofaint:+.3f} |",
    ]
    if ho_nofaint is not None and ho_faint is not None:
        lines.append(
            f"| Holdout test RMS (mas) | {ho_nofaint:.2f} | {ho_faint:.2f} | {ho_faint - ho_nofaint:+.2f} |"
        )
    lines.append("")

    # Count G>=21 stars in faint run
    n_faint_cat = 0
    if md_faint is not None:
        n_faint_cat = int((md_faint.matched_data["gaia_mag"] >= 21).sum())
    lines.append(f"**Faint stars (G≥21) detected in 'with' run:** {n_faint_cat}")
    lines.append("")

    # Bright-star-only comparison (G<20)
    if md_nofaint is not None and md_faint is not None:
        bright_nofaint = md_nofaint.matched_data[
            md_nofaint.matched_data["gaia_mag"] < 20
        ]
        bright_faint = md_faint.matched_data[md_faint.matched_data["gaia_mag"] < 20]
        if len(bright_nofaint) > 1 and len(bright_faint) > 1:
            cos_dec = np.cos(np.radians(bright_nofaint["dec_gaia"]))
            dra = (
                (bright_nofaint["ra_final"] - bright_nofaint["ra_gaia"]).values
                * 3600e3
                * cos_dec
            )
            ddec = (
                bright_nofaint["dec_final"] - bright_nofaint["dec_gaia"]
            ).values * 3600e3
            bright_rms_no = float(np.sqrt(np.mean(dra**2 + ddec**2)))

            cos_dec = np.cos(np.radians(bright_faint["dec_gaia"]))
            dra = (
                (bright_faint["ra_final"] - bright_faint["ra_gaia"]).values
                * 3600e3
                * cos_dec
            )
            ddec = (
                bright_faint["dec_final"] - bright_faint["dec_gaia"]
            ).values * 3600e3
            bright_rms_faint = float(np.sqrt(np.mean(dra**2 + ddec**2)))

            lines.extend(
                [
                    "**Bright-star-only (G<20) RMS:**",
                    "| | Without G≥21 | With G≥21 | Delta |",
                    "|-------|-------------|-----------|-------|",
                    f"| N bright | {len(bright_nofaint)} | {len(bright_faint)} | +{len(bright_faint) - len(bright_nofaint)} |",
                    f"| Bright RMS (mas) | {bright_rms_no:.3f} | {bright_rms_faint:.3f} | {bright_rms_faint - bright_rms_no:+.3f} |",
                    "",
                ]
            )

    # Per-bin magnitude comparison table
    all_lo = sorted(
        set(b["lo"] for b in bins_nofaint) | set(b["lo"] for b in bins_faint)
    )
    if all_lo:
        lines.extend(
            [
                "## Per-Bin Magnitude Comparison",
                "",
                "| G_lo | G_hi | N (no) | RMS (no) | N (with) | RMS (with) | Delta RMS |",
                "|------|------|--------|----------|----------|------------|-----------|",
            ]
        )
        seen = set()
        for lo in all_lo:
            bn = next((b for b in bins_nofaint if b["lo"] == lo), None)
            bf = next((b for b in bins_faint if b["lo"] == lo), None)
            if bn is None and bf is None:
                continue
            key = (lo, (bn or bf)["hi"])
            if key in seen:
                continue
            seen.add(key)
            hi = (bn or bf)["hi"]
            nn = bn["n"] if bn else 0
            rn = bn["rms_tot_mas"] if bn else float("nan")
            nf = bf["n"] if bf else 0
            rf = bf["rms_tot_mas"] if bf else float("nan")
            delta = rf - rn if not (np.isnan(rn) or np.isnan(rf)) else float("nan")
            d_str = f"{delta:+.2f}" if not np.isnan(delta) else "—"
            lines.append(
                f"| {lo:.1f} | {hi:.1f} | {nn} | {rn:.2f} | {nf} | {rf:.2f} | {d_str} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Configuration",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| gaia_ruwe_limit | {config.gaia_ruwe_limit} |",
            f"| gaia_ruwe_faint_limit | {config.gaia_ruwe_faint_limit} |",
            f"| gaia_ruwe_transition_mag | {config.gaia_ruwe_transition_mag} |",
            f"| gaia_faint_limit | {config.gaia_faint_limit} |",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  -> Faint-star comparison report written to: {report_path}")


def run_rigorous_enhancements(calibration, data_io, dirs):
    """
    Runs DCR correction, hold-out CV, magnitude stratification, and CCD
    edge diagnostics on the advanced-calibrated matched data, writing
    telemetry and plots to the specified directories.
    """
    import numpy as np
    from astropy.time import Time

    from src.enhancements import (
        CCDEdgeDiagnostic,
        DCRCorrector,
        HoldoutCV,
        MagnitudeStratifier,
        write_telemetry_report,
    )
    from src.plotting import (
        plot_ccd_edge_diagnostics,
        plot_dcr_diagnostics,
        plot_holdout_cv,
        plot_magnitude_stratification,
        plot_spatial_residual_map,
    )

    if calibration.matched_data is None or len(calibration.matched_data) == 0:
        print("  [WARNING] No matched data; skipping enhancements.")
        return {}

    md = calibration.matched_data
    pixel_scale = data_io.pixel_scale_arcsec or 0.27
    print(
        f"  -> Running enhancements on {len(md)} matched stars "
        f'(pixel scale = {pixel_scale:.4f}"/px)'
    )

    # --- 7a. DCR correction (runs first to feed corrected residuals
    # into downstream diagnostics) ---
    print("\n  [7a] Differential Chromatic Refraction (DCR) correction:")
    obs_time = data_io.obstime or Time.now()
    # Observatory latitude depends on instrument: CTIO for DECam,
    # Paranal for FORS2/ESO.
    inst_upper = str(data_io.instrument).upper()
    if "FORS" in inst_upper or "VLT" in str(data_io.telescope).upper():
        obs_lat = -24.6273  # Paranal
    else:
        obs_lat = -30.17  # CTIO / Cerro Tololo
    ra_center = float(data_io.wcs.wcs.crval[0])
    dec_center = float(data_io.wcs.wcs.crval[1])
    dcr = DCRCorrector(
        matched_data=md,
        obs_time=obs_time,
        ra_center_deg=ra_center,
        dec_center_deg=dec_center,
        observatory_lat_deg=obs_lat,
        pixel_scale_arcsec=pixel_scale,
    )
    dcr_telemetry = dcr.run()
    print(
        f"     -> applied = {dcr_telemetry.get('applied', False)}, "
        f"best-fit a = {dcr_telemetry.get('coef_a_mas', float('nan')):.2f} mas, "
        f"b = {dcr_telemetry.get('coef_b_mas', float('nan')):.2f} mas/mag, "
        f"median zd = {dcr_telemetry.get('median_zd_deg', float('nan')):.1f} deg"
    )
    if dcr_telemetry.get("applied"):
        print(
            f"     -> RMS {dcr_telemetry['rms_before_mas']:.2f} -> "
            f"{dcr_telemetry['rms_after_mas']:.2f} mas "
            f"(LOO-CV {dcr_telemetry['cv_rms_before_mas']:.2f} -> "
            f"{dcr_telemetry['cv_rms_after_mas']:.2f} mas)"
        )
    plot_dcr_diagnostics(md, dcr_telemetry, plots_dir=dirs.plots)

    # --- 7b. CCD edge effects and tree-ring diagnostics ---
    print("\n  [7b] CCD edge effects & tree-ring diagnostics:")
    ccd_edge = CCDEdgeDiagnostic(
        matched_data=md,
        image_shape=data_io.image_data.shape,
        pixel_scale_arcsec=pixel_scale,
    )
    ccd_edge_telemetry = ccd_edge.run(apply_correction=True)
    evc = ccd_edge_telemetry.get("edge_vs_centre", {})
    print(
        f"     -> centre RMS = {evc.get('centre_rms_mas', float('nan')):.2f} mas "
        f"(N={evc.get('centre_n', 0)}), "
        f"edge RMS = {evc.get('edge_rms_mas', float('nan')):.2f} mas "
        f"(N={evc.get('edge_n', 0)})"
    )
    if ccd_edge_telemetry.get("applied"):
        print(
            f"     -> tree-ring/edge correction APPLIED: RMS "
            f"{ccd_edge_telemetry['rms_before_mas']:.2f} -> "
            f"{ccd_edge_telemetry['rms_after_mas']:.2f} mas "
            f"(LOO-CV {ccd_edge_telemetry['cv_rms_before_mas']:.2f} -> "
            f"{ccd_edge_telemetry['cv_rms_after_mas']:.2f} mas)"
        )
    else:
        print(
            f"     -> tree-ring/edge correction skipped (LOO-CV did not improve: "
            f"{ccd_edge_telemetry.get('cv_rms_before_mas', float('nan')):.2f} -> "
            f"{ccd_edge_telemetry.get('cv_rms_after_mas', float('nan')):.2f} mas)"
        )
    plot_spatial_residual_map(md, data_io.image_data.shape, plots_dir=dirs.plots)
    plot_ccd_edge_diagnostics(ccd_edge_telemetry, plots_dir=dirs.plots)

    # --- 7c. Strict train/test hold-out cross-validation ---
    print("\n  [7c] Strict train/test hold-out cross-validation:")
    holdout = HoldoutCV(
        matched_data=md,
        wcs=data_io.wcs,
        train_fraction=0.7,
        n_splits=5,
        random_seed=42,
    )
    holdout_summary = holdout.run()
    if holdout_summary.get("n_splits", 0) > 0:
        print(f"     -> {holdout_summary['n_splits']} splits (70% train / 30% test)")
        print(
            f"     -> TRAIN RMS = {holdout_summary['train_rms_mean_mas']:.2f} +/- "
            f"{holdout_summary['train_rms_std_mas']:.2f} mas"
        )
        print(
            f"     -> TEST  RMS = {holdout_summary['test_rms_mean_mas']:.2f} +/- "
            f"{holdout_summary['test_rms_std_mas']:.2f} mas  "
            f"(unbiased predictive precision)"
        )
        print(
            f"     -> overfit ratio = {holdout_summary['overfit_ratio']:.3f} "
            f"(1.00 = no overfitting)"
        )
    plot_holdout_cv(holdout_summary, plots_dir=dirs.plots)

    # --- 7d. Magnitude stratification and faint-star local anchoring ---
    print("\n  [7d] Magnitude stratification & faint-star local anchoring:")
    mag_stratifier = MagnitudeStratifier(
        matched_data=md,
        wcs=data_io.wcs,
        pixel_scale_arcsec=pixel_scale,
        fwhm_pixels=calibration.estimated_fwhm,
        faint_threshold=21.0,
    )
    mag_strat = mag_stratifier.run()
    bins = mag_strat.get("magnitude_bins", [])
    if bins:
        print(f"     -> {len(bins)} magnitude bins:")
        print(
            f"        {'G_lo':>5} {'G_hi':>5} {'N':>4} {'RMS_tot':>9} "
            f"{'Floor':>7} {'Centroid':>9}"
        )
        for b in bins:
            print(
                f"        {b['lo']:>5.1f} {b['hi']:>5.1f} {b['n']:>4d} "
                f"{b['rms_tot_mas']:>7.2f}m {b['gaia_floor_mas']:>6.2f}m "
                f"{b['centroid_noise_mas']:>7.2f}m"
            )
    if mag_strat.get("photon_break_mag") is not None:
        print(
            f"     -> Photon-noise break magnitude: G = "
            f"{mag_strat['photon_break_mag']:.2f}"
        )
    fs = mag_strat.get("faint_summary")
    if fs:
        print(
            f"     -> Faint-star (G>=21) local anchoring: N = {fs['n_faint']}, "
            f"local RMS = {fs['local_rms_mas']:.2f} mas, "
            f"median sep = {fs['local_median_sep_mas']:.2f} mas, "
            f"mean # anchors = {fs['mean_n_anchor']:.1f}"
        )
    else:
        print(
            "     -> Faint-star local anchoring: no G>=21 stars in this field's "
            "Gaia catalog (catalog limit reached)."
        )
    plot_magnitude_stratification(mag_strat, plots_dir=dirs.plots)

    # Save the faint-anchor catalog if it exists
    if mag_stratifier.faint_anchor_results is not None:
        mag_stratifier.faint_anchor_results.to_csv(
            dirs.catalogs / "faint_star_local_anchor.csv", index=False
        )

    # --- 7e. Save final catalog and write telemetry ---
    md.to_csv(dirs.catalogs / "enhanced_calibrated_coordinates.csv", index=False)

    # Final in-sample RMS for the report
    cos_dec = np.cos(np.radians(md["dec_gaia"]))
    dra = (md["ra_final"] - md["ra_gaia"]).values * 3600e3 * cos_dec
    ddec = (md["dec_final"] - md["dec_gaia"]).values * 3600e3
    final_rms_mas = float(np.sqrt(np.mean(dra**2 + ddec**2)))

    telemetry_path = dirs.logs / "enhanced_telemetry_report.md"
    write_telemetry_report(
        path=telemetry_path,
        instrument=data_io.instrument,
        holdout_cv=holdout_summary,
        mag_strat=mag_strat,
        dcr=dcr_telemetry,
        ccd_edge=ccd_edge_telemetry,
        final_rms_mas=final_rms_mas,
        n_final=len(md),
    )
    print(f"\n  -> Telemetry report written to: {telemetry_path}")
    print(
        f"  -> Final corrected matched catalog: "
        f"{dirs.catalogs / 'enhanced_calibrated_coordinates.csv'}"
    )
    print(
        f"  -> Final in-sample RMS (post-enhancement): {final_rms_mas:.3f} mas "
        f"(N={len(md)})"
    )

    return {
        "holdout_cv": holdout_summary,
        "magnitude_stratification": mag_strat,
        "dcr": dcr_telemetry,
        "ccd_edge": ccd_edge_telemetry,
        "final_rms_mas": final_rms_mas,
    }


def isolate_instrumental_precision(calibration, data_io, logs_dir=None):
    """
    Isolates instrumental astrometric precision via quadrature error budget:

        sigma_total^2 = sigma_inst^2 + sigma_atm^2 + sigma_gaia^2

    The atmospheric component is modelled as Kolmogorov turbulence scaled by
    exposure time and airmass; the Gaia floor is taken from the per-star
    centroid noise model. Writes a report to the logs directory.
    """
    import numpy as np

    if logs_dir is None:
        logs_dir = LOGS_DIR

    if calibration.matched_data is None or len(calibration.matched_data) == 0:
        print("  [WARNING] No matched data; skipping instrumental isolation.")
        return

    # Observed total RMS (from the final cleaned sample)
    cos_dec = np.cos(np.radians(calibration.matched_data["dec_gaia"]))
    dra = (
        (calibration.matched_data["ra_final"] - calibration.matched_data["ra_gaia"])
        * 3600e3
        * cos_dec
    )
    ddec = (
        calibration.matched_data["dec_final"] - calibration.matched_data["dec_gaia"]
    ) * 3600e3
    sigma_total = float(np.sqrt(np.mean(dra**2 + ddec**2)))  # mas

    # Kolmogorov turbulence model: sigma_atm ~ t^{-1/2} * sec(z)^{0.6}
    # Normalised to 15 mas at 30 s, airmass = 1.3.
    base_sigma_atm = config.theoretical_sigma_atmosphere_mas
    # Adjust atmospheric model for VLT/Paranal (better median seeing)
    inst_upper = str(data_io.instrument).upper()
    if (
        "FORS2" in inst_upper
        or "EFOSC" in inst_upper
        or "VLT" in str(data_io.telescope).upper()
    ):
        # Lower Kolmogorov floor for Paranal site quality
        base_sigma_atm = 12.0  # mas at 30s, X=1.3
        logger.info(
            "ESO instrument detected; using σ_atm base=%.1f mas", base_sigma_atm
        )
    base_exptime = 30.0  # seconds
    base_airmass = 1.3
    exptime = data_io.exptime if data_io.exptime else base_exptime
    airmass = data_io.airmass if data_io.airmass else base_airmass
    sigma_atm = (
        base_sigma_atm
        * np.sqrt(base_exptime / exptime)
        * (airmass / base_airmass) ** 0.6
    )

    # Gaia reference floor: median of per-star sigma_centroid in mas
    if "sigma_centroid_mas" in calibration.matched_data.columns:
        sigma_gaia = float(
            np.median(calibration.matched_data["sigma_centroid_mas"].clip(upper=2.0))
        )
    else:
        sigma_gaia = config.theoretical_sigma_gaia_mas

    # Quadrature decomposition
    sigma_inst_sq = sigma_total**2 - sigma_atm**2 - sigma_gaia**2
    if sigma_inst_sq < 0:
        sigma_inst = 0.0
        note = " (clipped to 0 — atmosphere+Gaia budget exceeds observed)"
    else:
        sigma_inst = float(np.sqrt(sigma_inst_sq))
        note = ""

    print(f"  -> σ_total      = {sigma_total:7.3f} mas  (observed, post-calibration)")
    print(
        f"  -> σ_atmosphere = {sigma_atm:7.3f} mas  (Kolmogorov, t={exptime:.0f}s, X={airmass:.3f})"
    )
    print(f"  -> σ_gaia       = {sigma_gaia:7.3f} mas  (reference catalog floor)")
    print(f"  -> σ_instrument = {sigma_inst:7.3f} mas{note}")
    logger.info(
        "Instrumental precision isolation: σ_total=%.3f mas, σ_atm=%.3f mas, "
        "σ_gaia=%.3f mas → σ_inst=%.3f mas (instrument=%s, exptime=%.1fs, airmass=%.3f)",
        sigma_total,
        sigma_atm,
        sigma_gaia,
        sigma_inst,
        data_io.instrument,
        exptime,
        airmass,
    )

    # Write a dedicated report
    report_path = logs_dir / "instrumental_precision_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("INSTRUMENTAL PRECISION ISOLATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Instrument: {data_io.instrument}\n")
        f.write(f"Telescope:  {data_io.telescope}\n")
        f.write(f"Filter:     {data_io.filter_name}\n")
        f.write(f"Pixel scale: {data_io.pixel_scale_arcsec:.4f} arcsec/px\n")
        f.write(f"Exposure time: {exptime:.1f} s\n")
        f.write(f"Airmass: {airmass:.4f}\n")
        if data_io.seeing_arcsec:
            f.write(f"Seeing (DIMM): {data_io.seeing_arcsec:.2f} arcsec\n")
        f.write(f"\nStars used: {len(calibration.matched_data)}\n\n")
        f.write("ERROR BUDGET (quadrature decomposition):\n")
        f.write(f"  σ_total      = {sigma_total:7.3f} mas  (observed)\n")
        f.write(f"  σ_atmosphere = {sigma_atm:7.3f} mas  (Kolmogorov model)\n")
        f.write(f"  σ_gaia       = {sigma_gaia:7.3f} mas  (reference floor)\n")
        f.write(f"  σ_instrument = {sigma_inst:7.3f} mas  (derived){note}\n\n")
        f.write("Formula: σ_inst = sqrt(σ_total² - σ_atm² - σ_gaia²)\n")
        f.write("Atmospheric model: σ_atm = 15mas × sqrt(30s/t) × (X/1.3)^0.6\n")
        f.write("  (Bernstein et al. 2017, PASP, DECam astrometric performance)\n")
    print(f"  -> Report written to: {report_path}")


def run_faint_precision_analysis(calibration, data_io, dirs):
    """
    Stage 9: faint-star precision analysis.

    Extracts stars with G >= 21, compares pipeline-calibrated positions
    to the Gaia reference frame, tabulates per-magnitude-bin precision,
    and generates a multi-panel comparison plot.  Results are saved per
    instrument under the catalogs/ and plots/ subdirectories.

    Requires the matched_data DataFrame produced by the advanced
    calibration pipeline (Stage 7 / 8).
    """
    from src.enhancements import PrecisionAnalyzer
    from src.plotting import plot_faint_precision_comparison

    md = calibration.matched_data
    if md is None or len(md) == 0:
        print("  [WARNING] No matched data available; skipping faint-star analysis.")
        return

    pixel_scale = data_io.pixel_scale_arcsec or 0.27
    print(f"  -> Running faint-star precision analysis on {len(md)} matched stars...")

    analyzer = PrecisionAnalyzer(
        matched_data=md,
        pixel_scale_arcsec=pixel_scale,
        fwhm_pixels=calibration.estimated_fwhm,
        bin_width=config.analysis_mag_bin_width,
        faint_threshold=21.0,
        extra_ranges=config.analysis_mag_ranges,
    )
    result = analyzer.run()

    # --- Save faint-star catalog ---
    faint_cat = result.get("faint_catalog")
    if faint_cat is not None and len(faint_cat) > 0:
        cat_path = dirs.catalogs / "faint_precision_catalog_G21.csv"
        faint_cat.to_csv(cat_path, index=False)
        print(f"  -> Faint-star catalog saved: {cat_path}  ({len(faint_cat)} stars)")
    else:
        print("  -> No faint stars (G>=21) found in the matched catalog.")

    # --- Generate comparison plot ---
    plot_faint_precision_comparison(result, plots_dir=dirs.plots)
    print(
        f"  -> Faint precision comparison plot saved: {dirs.plots / 'faint_precision_comparison.png'}"
    )

    # --- Print per-bin summary to console ---
    hdr = f"        {'G_lo':>5} {'G_hi':>5} {'N':>4} {'Pipe_RMS':>9} {'Gaia_flr':>10} {'Expected':>9} {'Improv':>7}"
    g21_bins = result.get("g21_bins", [])
    if g21_bins:
        print(f"     [G≥21] {len(g21_bins)} magnitude bins:")
        print(hdr)
        for b in g21_bins:
            print(
                f"        {b['lo']:>5.1f} {b['hi']:>5.1f} {b['n']:>4d} "
                f"{b['pipeline_rms_mas']:>8.2f}m {b['gaia_floor_mas']:>9.2f}m "
                f"{b['expected_total_mas']:>8.2f}m {b['improvement_factor']:>6.2f}x"
            )

    range_list = result.get("range_bins_list", [])
    for entry in range_list:
        rlo, rhi = entry["range"]
        bins = entry["bins"]
        if not bins:
            continue
        print(f"     [{rlo:.0f} < G < {rhi:.0f}] {len(bins)} bins:")
        print(hdr)
        for b in bins:
            print(
                f"        {b['lo']:>5.1f} {b['hi']:>5.1f} {b['n']:>4d} "
                f"{b['pipeline_rms_mas']:>8.2f}m {b['gaia_floor_mas']:>9.2f}m "
                f"{b['expected_total_mas']:>8.2f}m {b['improvement_factor']:>6.2f}x"
            )


if __name__ == "__main__":
    main()
