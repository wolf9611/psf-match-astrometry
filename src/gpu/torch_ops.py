"""
PyTorch-based batched nonlinear PSF fitting.

Used by Stage 3 (Photometry) for batched nonlinear PSF-fitting
photometry.  PyTorch is optional; this module imports cleanly without
it because the caller passes the probed ``torch`` module as an
argument.

Design
------

* Inputs are NumPy arrays on the host.
* The function transfers the whole batch to the GPU once, runs
  L-BFGS optimisation for a small fixed number of iterations, and
  returns the fitted parameters to the host.
* Deterministic mode is honoured when ``deterministic=True``.
"""

from __future__ import annotations

import numpy as np


def batched_psf_fit(
    stamps,
    init_params,
    psf_model_fn,
    fit_shape,
    torch=None,
    batch_size=256,
    deterministic=False,
    max_iters=50,
):
    """Batched nonlinear PSF fitting via PyTorch L-BFGS.

    ``psf_model_fn(params, grid)`` is a Python callable that takes a
    ``torch.Tensor`` parameter batch of shape ``(B, P)`` and a
    ``torch.Tensor`` grid of shape ``(B, H*W, 2)`` and returns a
    predicted stamp batch of shape ``(B, H*W)``.  The function is
    responsible for moving tensors to the correct device internally.
    """
    torch_mod = torch
    stamps = np.asarray(stamps, dtype=np.float64)
    init_params = np.atleast_2d(np.asarray(init_params, dtype=np.float64))
    n = len(stamps)
    n_params = init_params.shape[1]
    out_params = np.zeros((n, n_params), dtype=np.float64)
    out_success = np.zeros(n, dtype=bool)

    if deterministic:
        try:
            torch_mod.manual_seed(0)
            torch_mod.backends.cudnn.deterministic = True
            torch_mod.backends.cudnn.benchmark = False
        except Exception:
            pass

    device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")

    fy, fx = fit_shape
    yy, xx = np.mgrid[:fy, :fx]
    grid_np = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float64)
    grid_t = torch_mod.tensor(grid_np, dtype=torch_mod.float64, device=device)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_stamps = stamps[start:end]
        batch_init = init_params[start:end]
        B = batch_stamps.shape[0]
        # Tile the grid across the batch
        grid_batch = grid_t.unsqueeze(0).expand(B, -1, -1).contiguous()
        stamps_t = torch_mod.tensor(
            batch_stamps.reshape(B, -1), dtype=torch_mod.float64, device=device
        )
        params = torch_mod.tensor(
            batch_init, dtype=torch_mod.float64, device=device, requires_grad=True
        )
        try:
            opt = torch_mod.optim.LBFGS(
                [params], max_iter=max_iters, line_search_fn="strong_wolfe"
            )

            def closure():
                opt.zero_grad()
                pred = psf_model_fn(params, grid_batch)
                loss = torch_mod.mean((pred - stamps_t) ** 2)
                loss.backward()
                return loss

            opt.step(closure)
            out_params[start:end] = params.detach().cpu().numpy()
            out_success[start:end] = True
        except Exception:
            out_params[start:end] = batch_init
            out_success[start:end] = False
    return {"params": out_params, "success": out_success}
