"""
Configuration management for the PSF-match pipeline.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Backward-compatible default output directories (pre-per-instrument)
CATALOGS_DIR = OUTPUT_DIR / "catalogs"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"

# Instrument presets  --  maps instrument name to config overrides
PRESETS = {
    "FORS2": {
        "image_file": INPUT_DIR / "FORS2" / "eso_fors2_single_01.fits",
        "gaia_catalog_file": INPUT_DIR / "FORS2" / "gaia_catalog_fors2.csv",
        "use_milliquas": False,
    },
}


@dataclass(frozen=True)
class OutputDirs:
    """Output directory paths for a pipeline run."""

    catalogs: Path
    plots: Path
    models: Path
    logs: Path


def get_output_dirs(instrument: str = "default") -> OutputDirs:
    """
    Returns output directory paths for the given instrument,
    creating the directories if they don't exist.

    The instrument name is sanitised:
      - lowercased, spaces replaced by underscores
      - empty or "UNKNOWN" maps to "default"
    """
    if not instrument or instrument.upper() == "UNKNOWN":
        instrument = "default"
    base = OUTPUT_DIR / instrument.lower().replace(" ", "_")
    dirs = OutputDirs(
        catalogs=base / "catalogs",
        plots=base / "plots",
        models=base / "models",
        logs=base / "logs",
    )
    for d in [dirs.catalogs, dirs.plots, dirs.models, dirs.logs]:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


@dataclass(frozen=True)
class PipelineConfig:
    """
    Configuration parameters for the PSF photometry and astrometry pipeline.
    """

    image_file: Path = INPUT_DIR / "DECam" / "image.fits"
    gaia_catalog_file: Path = INPUT_DIR / "DECam" / "gaia_catalog.csv"
    auto_download_gaia: bool = True
    gaia_search_radius_deg: float = 0.5
    gain: float = 1.0
    readnoise: float = 10.0
    extension: int = 0

    saturation_level: float = 60000.0  # ADU; overridden by FITS SATURATE header
    gaia_bright_limit: float = 6.0
    gaia_faint_limit: float = 21.5
    gaia_color_blue_limit: float = 0.0  # exclude only pathological blue stars
    gaia_color_red_limit: float = 2.5  # exclude only pathological red stars
    gaia_ruwe_limit: float = 1.4  # exclude astrometric binaries (bright stars)
    gaia_ruwe_faint_limit: float = 2.5  # relaxed for faint stars (G >= transition)
    gaia_ruwe_transition_mag: float = 20.0  # magnitude above which faint limit applies

    max_separation_arcsec: float = 2.0
    photometry_methods: Sequence[str] = ("moffat", "empirical", "aperture")

    # Advanced calibration parameters  --  tuned for sub-10 mas differential astrometry
    advanced_match_radius_arcsec: float = 0.5  # was 2.0; tighter for cleaner sample
    sigma_clip_threshold_sigma: float = 3.0  # was 2.5; looser keeps more stars
    sigma_clip_max_iterations: int = 10  # was 5
    final_sigma_clip_sigma: float = (
        2.3  # per-axis final clip (balanced for 8-10 mas target)
    )
    sip_order: int = 4  # was 3; degree-4 SIP for finer distortion
    sip_n_iterations: int = 5  # was 3
    enable_gaussian_process: bool = True  # GP alternative to TPS
    gp_length_scale_pixels: float = 150.0  # Matern length scale (pixels)

    # PSF construction parameters
    epsf_oversampling: int = 4  # ePSF oversampling factor
    epsf_maxiters: int = 10  # ePSF builder iterations
    epsf_n_stars: int = 30  # PSF stars for ePSF
    apply_psf_homogenization: bool = True  # smooths ePSF grid artifacts

    clean_old_outputs: bool = True
    enable_systematic_correction: bool = True
    enable_advanced_calibration: bool = True

    # --- Proper motion damping & handling ---
    # Damping reduces the applied proper motion for faint/noisy-PM stars
    # where Gaia's PM is uncertain (G >= pm_damping_mag_threshold).
    pm_mode: str = "damped"  # "full", "damped", or "none"
    enable_pm_damping: bool = True
    pm_damping_mag_threshold: float = 17.0
    pm_damping_factor: float = 0.5  # scale factor for damped stars

    # --- Saturated star handling ---
    gaia_saturated_mag_limit: float = 6.0

    # --- Faint-star precision analysis ---
    include_faint_stars: bool = (
        True  # when False, drop 2-param sources and use strict RUWE
    )
    enable_faint_precision_analysis: bool = True
    analysis_mag_bin_width: float = 0.5
    analysis_mag_ranges: tuple = ((19.0, 22.0),)

    process_xy_only: bool = False
    xy_file: Path = INPUT_DIR / "xy.txt"

    use_milliquas: bool = False
    milliquas_catalog_file: Path = INPUT_DIR / "milliquas_catalog.csv"
    auto_download_milliquas: bool = True

    # --- Multi-Extension FITS (MEF) support ---
    # When True, the pipeline processes every image-like extension in a MEF
    # file. When False (default), only the requested extension is processed.
    process_all_mef_extensions: bool = False

    # --- Instrumental precision isolation ---
    # Theoretical error budget components (mas) used to decompose the
    # observed sigma_total into sigma_instrument via quadrature.
    # Defaults follow Bernstein et al. (2017) for DECam; the pipeline
    # overrides these automatically for known ESO instruments.
    theoretical_sigma_atmosphere_mas: float = 15.0  # Kolmogorov (30s exp)
    theoretical_sigma_gaia_mas: float = 0.7  # Gaia DR3 reference floor

    # --- Optional CUDA GPU support ---
    # Device selection: "auto" (use CUDA when available), "cpu"
    # (force CPU), or "cuda" (force CUDA, fallback to CPU if
    # unavailable).
    device: str = "auto"
    # When True, GPU primitives that fail at runtime fall back to
    # their CPU equivalents silently and log a warning.
    gpu_fallback: bool = True
    # When True, configure deterministic GPU execution (PyTorch
    # cudnn.deterministic + manual_seed).  Useful for reproducible
    # bootstrap/holdout runs.
    deterministic: bool = False
    # Numerical precision for image and astrometry arrays.  Allowed
    # values: "float32" or "float64".  Defaults to float64 to
    # preserve existing CPU behaviour.
    image_precision: str = "float64"
    astrometry_precision: str = "float64"
    # Minimum input size before GPU dispatch is considered.  Below
    # these thresholds the overhead of host-device transfer dominates.
    gpu_min_elements: int = 500
    gpu_min_stars: int = 50
    # Default batch size for GPU-batched operations (aperture
    # photometry, PSF fitting, ridge solves).
    gpu_batch_size: int = 4096
    # Per-stage GPU toggles.  Keys are stage names: "detection",
    # "psf", "photometry", "systematic", "advanced", "enhancements",
    # "faint".  When a stage is missing the dispatcher uses its
    # default (see src.gpu.batching.DEFAULT_GPU_STAGES).
    gpu_stages: dict = field(default=None)  # populated by CLI; None means defaults
    # When True, log per-stage device selection and timings.
    gpu_enable_timing: bool = False
    # Worker readiness for future multi-GPU MEF processing.  When
    # greater than 1, MEF extensions may be processed in parallel
    # worker processes (placeholder for future distributed mode).
    gpu_workers: int = 1

    def validate(self):
        """
        Validates that required input files exist.
        """
        missing = []
        if not self.process_xy_only and not self.image_file.exists():
            missing.append(f"FITS image not found: {self.image_file}")
        if not self.gaia_catalog_file.exists() and not self.auto_download_gaia:
            missing.append(f"Gaia catalog not found: {self.gaia_catalog_file}")
        if self.process_xy_only and not self.xy_file.exists():
            missing.append(f"XY file not found: {self.xy_file}")

        if missing:
            raise FileNotFoundError(
                "Required inputs are missing.\n" + "\n".join(f"- {m}" for m in missing)
            )


config = PipelineConfig(use_milliquas=True)
try:
    config.validate()
except FileNotFoundError as e:
    print(f"Warning: {e}")
