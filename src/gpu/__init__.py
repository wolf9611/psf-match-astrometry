"""
Optional CUDA GPU support for the PSF-match pipeline.

This package provides device detection, CPU/CUDA dispatch, and
graceful fallback for every GPU-accelerated pipeline stage.  All
GPU libraries (CuPy, Numba CUDA, PyTorch) are optional: the package
imports cleanly without them.

Public API
----------

* ``Backend``              — central numerical dispatcher.
* ``DeviceManager``        — device detection and selection.
* ``probe_gpu``            — probes the runtime for optional GPU libs.
* ``configure_backend``    — (re)creates the singletons from CLI args.
* ``current_device``       — returns the active device string.
* ``kernels``              — Numba CUDA kernels (import-safe).
* ``batching``             — memory-aware batching and MEF planning.
"""

from src.gpu import batching, kernels  # re-exported for convenience
from src.gpu.backend import (
    Backend,
    DeviceManager,
    GPUAvailability,
    StageTimer,
    configure_backend,
    current_device,
    probe_gpu,
)

__all__ = [
    "Backend",
    "DeviceManager",
    "GPUAvailability",
    "StageTimer",
    "batching",
    "configure_backend",
    "current_device",
    "kernels",
    "probe_gpu",
]
