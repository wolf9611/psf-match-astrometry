"""
Diagnostic plotting and visualization utilities.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ImageNormalize, LogStretch, ZScaleInterval
from matplotlib.patches import Ellipse

from src.config import PLOTS_DIR, config

matplotlib.use("Agg")


def visualize_fits_image(
    image_data,
    sources=None,
    gaia_sources=None,
    file_path="image_preview.png",
    plots_dir=PLOTS_DIR,
):
    """
    Generates a preview of the FITS image with optional source overlays.
    """
    norm = ImageNormalize(image_data, interval=ZScaleInterval(), stretch=LogStretch())
    ny, nx = image_data.shape
    fig_w = min(14, nx / 200)
    fig_h = fig_w * ny / nx
    fig, axes = plt.subplots(figsize=(max(fig_w, 6), max(fig_h, 4)))

    axes.imshow(
        image_data, origin="lower", cmap="gray", norm=norm, interpolation="nearest"
    )

    if sources is not None:
        axes.scatter(
            sources["x_centroid"],
            sources["y_centroid"],
            s=30,
            facecolors="none",
            edgecolors="cyan",
            linewidths=0.6,
            label=f"{len(sources)} Detected Sources",
        )

    if gaia_sources is not None:
        axes.scatter(
            gaia_sources["x"],
            gaia_sources["y"],
            s=60,
            facecolors="none",
            edgecolors="#FF00FF",  # High-contrast Magenta
            linewidths=1.2,
            label="GAIA SOURCES",
        )

    fig.suptitle(f"FITS Image Preview — {nx}×{ny} px", fontsize=10, y=0.98)

    if sources is not None or gaia_sources is not None:
        legend = axes.legend(
            fontsize=9,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=2,
            framealpha=0.9,
        )
        for text in legend.get_texts():
            if "GAIA" in text.get_text():
                text.set_fontweight("bold")
                text.set_color("#FF00FF")
    axes.set_xlabel("X (pixels)")
    axes.set_ylabel("Y (pixels)")
    fig.subplots_adjust(top=0.88)
    plt.savefig(plots_dir / file_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_psf_comparison(
    original_psf,
    homogenized_psf,
    filename="psf_model_comparison.png",
    plots_dir=PLOTS_DIR,
):
    """
    Compares the raw empirical PSF with its azimuthally homogenized version.
    """
    _, axes = plt.subplots(2, 3, figsize=(15, 10))

    im1 = axes[0, 0].imshow(
        original_psf.data,
        norm=ImageNormalize(stretch=LogStretch()),
        origin="lower",
        cmap="gray",
    )
    axes[0, 0].set_title("Original ePSF")
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(
        homogenized_psf.data,
        norm=ImageNormalize(stretch=LogStretch()),
        origin="lower",
        cmap="gray",
    )
    axes[0, 1].set_title("Azimuthally Homogenized ePSF")
    plt.colorbar(im2, ax=axes[0, 1])

    diff = original_psf.data - homogenized_psf.data
    im3 = axes[0, 2].imshow(diff, origin="lower", cmap="RdBu_r")
    axes[0, 2].set_title("Difference (Original - Homogenized)")
    plt.colorbar(im3, ax=axes[0, 2])

    center_y, center_x = np.array(original_psf.data.shape) / 2.0
    y_grid, x_grid = np.ogrid[
        : original_psf.data.shape[0], : original_psf.data.shape[1]
    ]
    radial_distances = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
    max_radius = min(center_x, center_y) * 0.9
    radii = np.linspace(0, max_radius, 50)

    original_radial, homogenized_radial = [], []
    for rad in radii:
        mask = (radial_distances >= rad - 0.5) & (radial_distances < rad + 0.5)
        if np.sum(mask) > 0:
            original_radial.append(np.mean(original_psf.data[mask]))
            homogenized_radial.append(np.mean(homogenized_psf.data[mask]))
        else:
            original_radial.append(0)
            homogenized_radial.append(0)

    axes[1, 0].plot(radii, original_radial, "b-", label="Original", alpha=0.7)
    axes[1, 0].plot(radii, homogenized_radial, "r-", label="Homogenized", alpha=0.7)
    axes[1, 0].set_xlabel("Radius (pixels)")
    axes[1, 0].set_ylabel("Intensity")
    axes[1, 0].set_title("Radial Profiles Comparison")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale("log")

    cx_int, cy_int = int(center_x), int(center_y)
    axes[1, 1].plot(original_psf.data[cy_int, :], "b-", label="Original", alpha=0.7)
    axes[1, 1].plot(
        homogenized_psf.data[cy_int, :], "r-", label="Homogenized", alpha=0.7
    )
    axes[1, 1].set_xlabel("X (pixels)")
    axes[1, 1].set_ylabel("Intensity")
    axes[1, 1].set_title("Horizontal Cross-section")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(original_psf.data[:, cx_int], "b-", label="Original", alpha=0.7)
    axes[1, 2].plot(
        homogenized_psf.data[:, cx_int], "r-", label="Homogenized", alpha=0.7
    )
    axes[1, 2].set_xlabel("Y (pixels)")
    axes[1, 2].set_ylabel("Intensity")
    axes[1, 2].set_title("Vertical Cross-section")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_radial_profiles(psf_stars, image_clean, max_stars=15, plots_dir=PLOTS_DIR):
    """
    Plots radial profiles for individual PSF candidate stars.
    """

    def calculate_radial_profile(data, c_x, c_y, max_r=12):
        y_sz, x_sz = data.shape
        y_g, x_g = np.ogrid[:y_sz, :x_sz]
        dist = np.sqrt((x_g - c_x) ** 2 + (y_g - c_y) ** 2)
        radii_bins = np.arange(0.5, max_r, 0.5)
        prof, prof_std = [], []
        for rad in radii_bins:
            mask = (dist >= rad - 0.25) & (dist < rad + 0.25)
            if np.sum(mask) > 0:
                vals = data[mask]
                prof.append(np.mean(vals))
                prof_std.append(np.std(vals))
            else:
                prof.append(np.nan)
                prof_std.append(np.nan)
        return radii_bins, np.array(prof), np.array(prof_std)

    n_stars = min(max_stars, len(psf_stars))
    n_cols, n_rows = 4, int(np.ceil(n_stars / 4))
    _, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_stars):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        star = psf_stars[i]
        x_c, y_c = star["x"], star["y"]
        s_sz = 25
        x_min = max(0, int(x_c - s_sz // 2))
        x_max = min(image_clean.shape[1], int(x_c + s_sz // 2))
        y_min = max(0, int(y_c - s_sz // 2))
        y_max = min(image_clean.shape[0], int(y_c + s_sz // 2))
        stamp = image_clean[y_min:y_max, x_min:x_max]
        rel_x, rel_y = x_c - x_min, y_c - y_min
        rb, prof, ps = calculate_radial_profile(stamp, rel_x, rel_y)
        ax.errorbar(rb, prof, yerr=ps, fmt="o-", markersize=4)
        ax.set_xlabel("Radius (pixels)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Star {i + 1}")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    for i in range(n_stars, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(plots_dir / "radial_profiles_epsf_candidates.png")
    plt.close()


def plot_match_results(matches, method, rms_total, plots_dir=PLOTS_DIR):
    """
    Generates diagnostic histograms and scatter plots for catalog matching.
    """
    plt.style.use("seaborn-v0_8-pastel")
    _, ax = plt.subplots(figsize=(10, 6))
    ax.hist(matches["separation_arcsec"], bins=50, color="royalblue", edgecolor="black")
    ax.axvline(2.0, color="crimson", ls="--", lw=2, label="Match Limit")
    ax.set_xlabel("Separation (arcsec)", fontweight="bold")
    ax.set_ylabel("N Objects", fontweight="bold")
    ax.set_title(
        f"Angular Separation Dist - Method: {method.upper()}", fontweight="bold"
    )
    ax.legend()
    ax.text(
        0.95,
        0.95,
        f'Total RMS: {rms_total:.3f}"',
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        ha="right",
        bbox={"boxstyle": "round,pad=0.5", "fc": "wheat", "alpha": 0.7},
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(plots_dir / f"match_histogram_separation_{method}.png")
    plt.close()

    _, ax = plt.subplots(figsize=(8, 8))
    scat = ax.scatter(
        matches["ra_diff_arcsec"],
        matches["dec_diff_arcsec"],
        c=matches["separation_arcsec"],
        cmap="viridis",
        s=15,
    )
    ax.set_xlabel("ΔRA (arcsec)", fontweight="bold")
    ax.set_ylabel("ΔDEC (arcsec)", fontweight="bold")
    ax.set_title(f"Astrometric Residuals - {method.upper()}", fontweight="bold")
    ax.set_aspect("equal")
    ax.axhline(0, c="k", ls=":")
    ax.axvline(0, c="k", ls=":")
    plt.colorbar(scat, label="Separation (arcsec)")
    plt.tight_layout()
    plt.savefig(plots_dir / f"match_scatter_plot_{method}.png")
    plt.close()


def plot_final_astrometry(
    final_matches,
    estimated_fwhm,
    pixel_scale_arcsec=None,
    xlim_max=0.10,
    plots_dir=PLOTS_DIR,
):
    """
    Plots final astrometric precision after systematic corrections.
    """
    sep = np.sqrt(
        final_matches["ra_diff_final_arcsec"] ** 2
        + final_matches["dec_diff_final_arcsec"] ** 2
    )
    bins_edges = np.arange(0, xlim_max + 0.005, 0.005)
    hist, edges = np.histogram(sep, bins=bins_edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total_rms = np.sqrt(np.mean(sep**2))

    image_fwhm_arcsec = (
        estimated_fwhm * pixel_scale_arcsec if pixel_scale_arcsec else None
    )

    plt.figure(figsize=(10, 6))
    plt.bar(
        centers,
        hist,
        width=0.005,
        color="skyblue",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.8,
    )
    plt.axvline(
        2.0, linestyle="--", color="red", linewidth=2, label="Match Limit: 2.0 arcsec"
    )
    if image_fwhm_arcsec is not None and image_fwhm_arcsec <= xlim_max:
        plt.axvline(
            image_fwhm_arcsec,
            linestyle=":",
            color="green",
            linewidth=2,
            label=f"Image FWHM: {image_fwhm_arcsec:.3f} arcsec",
        )

    plt.title(
        "Angular Separation Dist - Method: SYSTEMATIC CORRECTED",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Separation (arcsec)", fontsize=12, fontweight="bold")
    plt.ylabel("N Objects", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)

    info_txt = (
        f'Image FWHM: {image_fwhm_arcsec:.3f}"\nTotal RMS: {total_rms:.3f}"'
        if image_fwhm_arcsec
        else f'Total RMS: {total_rms:.3f}"'
    )
    plt.text(
        0.95 * xlim_max,
        0.95 * hist.max(),
        info_txt,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.5", "fc": "wheat", "alpha": 0.8},
        fontsize=11,
        fontweight="bold",
    )
    plt.xlim(0, xlim_max)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        plots_dir / "final_astrometric_precision.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def generate_rms_histogram(matches, rms_total, plots_dir=PLOTS_DIR):
    """
    Generates a high-quality RMS histogram for the final catalog.
    """
    separations = matches["separation_arcsec"].values
    plt.figure(figsize=(10, 7))
    max_sep = min(2.5, separations.max() * 1.1)
    bin_size = 0.001 if max_sep <= 0.1 else (0.005 if max_sep <= 0.5 else 0.01)
    bins_arr = np.arange(0, max_sep + bin_size, bin_size)
    if len(bins_arr) < 50:
        bins_arr = np.linspace(0, max_sep, 100)

    plt.hist(
        separations,
        bins=bins_arr,
        color="skyblue",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )

    stats_txt = (
        f'N matches: {len(matches)}\nRMS total: {rms_total:.3f}"\n'
        f'Median sep: {np.median(separations):.3f}"\n'
        f'Mean sep: {np.mean(separations):.3f}"'
    )
    plt.text(
        0.98,
        0.98,
        stats_txt,
        transform=plt.gca().transAxes,
        fontsize=11,
        va="top",
        ha="right",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "wheat", "alpha": 0.8},
        fontweight="bold",
    )
    plt.axvline(
        rms_total,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Total RMS: {rms_total:.3f}"',
    )
    plt.axvline(
        np.median(separations),
        color="green",
        linestyle=":",
        linewidth=2,
        label=f'Median: {np.median(separations):.3f}"',
    )
    plt.xlabel("Separation (arcsec)", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Objects", fontsize=12, fontweight="bold")
    plt.title(
        "Final Astrometric Precision: Refined vs Gaia Catalog",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, max_sep)
    plt.tight_layout()
    plt.savefig(plots_dir / "rms.png", dpi=150, bbox_inches="tight")
    plt.close()


def generate_residuals_plot(matches, plots_dir=PLOTS_DIR):
    """
    Generates a 2D scatter plot of astrometric residuals.
    """
    plt.figure(figsize=(8, 8))
    scat = plt.scatter(
        matches["ra_diff_arcsec"],
        matches["dec_diff_arcsec"],
        c=matches["separation_arcsec"],
        cmap="viridis",
        s=20,
        alpha=0.7,
    )
    cbar = plt.colorbar(scat)
    cbar.set_label("Separation (arcsec)", fontweight="bold")
    plt.axhline(0, color="black", linestyle=":", alpha=0.5)
    plt.axvline(0, color="black", linestyle=":", alpha=0.5)
    rms_ra = np.sqrt(np.mean(matches["ra_diff_arcsec"] ** 2))
    rms_dec = np.sqrt(np.mean(matches["dec_diff_arcsec"] ** 2))
    ellipse = Ellipse(
        (0, 0),
        2 * rms_ra,
        2 * rms_dec,
        fill=False,
        color="red",
        linestyle="--",
        linewidth=2,
    )
    plt.gca().add_patch(ellipse)
    plt.xlabel("ΔRA cos(δ) (arcsec)", fontsize=12, fontweight="bold")
    plt.ylabel("ΔDEC (arcsec)", fontsize=12, fontweight="bold")
    plt.title("Astrometric Residuals: Refined vs Gaia", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(plots_dir / "residuals_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()


def generate_advanced_plots(
    matched_data, rms_history, stage_names=None, plots_dir=PLOTS_DIR
):
    """
    Visualizes the evolution of RMS and spatial residuals during advanced calibration.
    """
    _, axes = plt.subplots(3, 3, figsize=(18, 18))
    ax_hist = axes[0, 0]
    if stage_names is None:
        stages = ["Initial", "Polynomial", "2-pass", "TPS/GP", "Final (3σ clip)"]
        stages_used = stages[: len(rms_history)]
    else:
        stages_used = stage_names
    rms_hist_mas = [r * 1000 for r in rms_history]
    ax_hist.plot(
        range(len(rms_hist_mas)), rms_hist_mas, "o-", linewidth=2, markersize=8
    )
    # Highlight the final stage
    if len(rms_hist_mas) > 1:
        ax_hist.plot(
            len(rms_hist_mas) - 1,
            rms_hist_mas[-1],
            "r*",
            markersize=18,
            zorder=5,
            label=f"Final: {rms_hist_mas[-1]:.2f} mas",
        )
    ax_hist.set_xlabel("Stage")
    ax_hist.set_ylabel("RMS (mas)")
    ax_hist.set_title("Complete RMS Evolution")
    ax_hist.grid(True, alpha=0.3)
    ax_hist.set_xticks(range(len(stages_used)))
    ax_hist.set_xticklabels(stages_used, rotation=45, ha="right")
    ax_hist.axhline(10.0, color="green", ls="--", alpha=0.6, label="Target: 10.0 mas")
    ax_hist.axhline(8.0, color="green", ls=":", alpha=0.4, label="Lower bound: 8.0 mas")
    ax_hist.legend(fontsize=8)

    ra_diff = (
        (matched_data["ra_final"] - matched_data["ra_gaia"])
        * 3600e3
        * np.cos(np.radians(matched_data["dec_gaia"]))
    )
    dec_diff = (matched_data["dec_final"] - matched_data["dec_gaia"]) * 3600e3
    ax_scat = axes[0, 1]
    ax_scat.scatter(ra_diff, dec_diff, alpha=0.7, s=50)
    ax_scat.set_xlabel("Δ RA·cos(δ) (mas)")
    ax_scat.set_ylabel("Δ DEC (mas)")
    ax_scat.set_title(f"Final Residuals\nRMS: {rms_history[-1] * 1000:.2f} mas")
    ax_scat.grid(True, alpha=0.3)
    ax_scat.axis("equal")

    ra_d_init = (
        (matched_data["ra_measured"] - matched_data["ra_gaia"])
        * 3600e3
        * np.cos(np.radians(matched_data["dec_gaia"]))
    )
    dec_d_init = (matched_data["dec_measured"] - matched_data["dec_gaia"]) * 3600e3
    ax_comp = axes[0, 2]
    ax_comp.scatter(ra_d_init, dec_d_init, alpha=0.5, label="Initial", s=30)
    ax_comp.scatter(ra_diff, dec_diff, alpha=0.7, label="Final", s=30)
    ax_comp.set_xlabel("Δ RA·cos(δ) (mas)")
    ax_comp.set_ylabel("Δ DEC (mas)")
    ax_comp.set_title("Initial vs Final Comparison")
    ax_comp.legend()
    ax_comp.grid(True, alpha=0.3)
    ax_comp.axis("equal")

    resids = np.sqrt(ra_diff**2 + dec_diff**2)
    ax_dist = axes[1, 0]
    ax_dist.hist(resids, bins=20, alpha=0.7, density=True, edgecolor="black")
    ax_dist.axvline(
        np.mean(resids),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(resids):.2f} mas",
    )
    ax_dist.axvline(
        np.median(resids),
        color="orange",
        linestyle="--",
        label=f"Median: {np.median(resids):.2f} mas",
    )
    ax_dist.set_xlabel("Residual (mas)")
    ax_dist.set_ylabel("Density")
    ax_dist.set_title("Residual Distribution")
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)

    ax_map = axes[1, 1]
    scat_map = ax_map.scatter(
        matched_data["x_pixel"],
        matched_data["y_pixel"],
        c=resids,
        cmap="viridis",
        alpha=0.7,
        s=50,
    )
    ax_map.set_xlabel("X (pixels)")
    ax_map.set_ylabel("Y (pixels)")
    ax_map.set_title("Spatial Residual Map")
    plt.colorbar(scat_map, ax=ax_map, label="Residual (mas)")

    # Residuals versus Gaia G magnitude
    ax_mag = axes[1, 2]
    ax_mag.scatter(
        matched_data["gaia_mag"],
        ra_diff,
        alpha=0.6,
        s=30,
        c="royalblue",
        label="ΔRA·cosδ",
    )
    ax_mag.scatter(
        matched_data["gaia_mag"], dec_diff, alpha=0.6, s=30, c="crimson", label="ΔDEC"
    )
    ax_mag.axhline(0, c="k", ls=":", alpha=0.4)
    ax_mag.set_xlabel("Gaia G magnitude")
    ax_mag.set_ylabel("Residual (mas)")
    ax_mag.set_title("Residuals vs Magnitude")
    ax_mag.legend(fontsize=9)
    ax_mag.grid(True, alpha=0.3)

    # Residuals as a function of BP-RP colour
    ax_color = axes[2, 0]
    if "bp_rp" in matched_data.columns:
        bp_rp = matched_data["bp_rp"].values
        ax_color.scatter(
            bp_rp, ra_diff, alpha=0.6, s=30, c="royalblue", label="ΔRA·cosδ"
        )
        ax_color.scatter(bp_rp, dec_diff, alpha=0.6, s=30, c="crimson", label="ΔDEC")
        ax_color.set_xlabel("Gaia bp−rp color")
    else:
        ax_color.text(
            0.5,
            0.5,
            "no bp_rp column",
            ha="center",
            va="center",
            transform=ax_color.transAxes,
        )
    ax_color.axhline(0, c="k", ls=":", alpha=0.4)
    ax_color.set_ylabel("Residual (mas)")
    ax_color.set_title("Residuals vs Color (DCR diagnostic)")
    ax_color.legend(fontsize=9)
    ax_color.grid(True, alpha=0.3)

    # Distribution of absolute residual components
    ax_quad = axes[2, 1]
    bins = np.linspace(-3 * np.std(ra_diff), 3 * np.std(ra_diff), 25)
    ax_quad.hist(
        ra_diff,
        bins=bins,
        alpha=0.5,
        color="royalblue",
        edgecolor="black",
        label=f"ΔRA·cosδ (σ={np.std(ra_diff):.2f} mas)",
    )
    ax_quad.hist(
        dec_diff,
        bins=bins,
        alpha=0.5,
        color="crimson",
        edgecolor="black",
        label=f"ΔDEC (σ={np.std(dec_diff):.2f} mas)",
    )
    ax_quad.axvline(0, c="k", ls=":", alpha=0.5)
    ax_quad.set_xlabel("Residual (mas)")
    ax_quad.set_ylabel("Count")
    ax_quad.set_title("RA vs Dec Residual Distribution")
    ax_quad.legend(fontsize=9)
    ax_quad.grid(True, alpha=0.3)

    ax_stats = axes[2, 2]
    final_rms_mas = rms_history[-1] * 1000
    initial_rms_mas = rms_history[0] * 1000
    improvement = (
        ((rms_history[0] - rms_history[-1]) / rms_history[0] * 100)
        if rms_history[0] > 0
        else 0
    )
    rms_txt = (
        f"    RMS Final: {final_rms_mas:.2f} mas\n"
        f"    RMS Initial: {initial_rms_mas:.2f} mas\n"
        f"    Improvement: {improvement:.1f}%"
    )
    in_target = "✓ IN TARGET" if 8.0 <= final_rms_mas <= 10.0 else "✗ OUTSIDE TARGET"
    stats_body = (
        f"\n    Objects: {len(matched_data)}\n    Median: {np.median(resids):.2f} mas\n"
        f"    95th pct: {np.percentile(resids, 95):.2f} mas\n"
        f"    Max: {np.max(resids):.2f} mas\n"
        f"    RA σ: {np.std(ra_diff):.2f} mas | Dec σ: {np.std(dec_diff):.2f} mas\n"
        f"    Status: {in_target}\n\n    Applied Methods:\n"
        f"    • Robust polynomial fitting (degree 3, normalised)\n"
        f"    • Iterative sigma clipping ({config.sigma_clip_threshold_sigma:.1f}σ)\n"
        f"    • SNR-based inverse-variance weighting\n"
        f"    • Differential chromatic refraction (DCR)\n"
        f"    • 2nd-pass colour/magnitude correction\n"
        f"    • TPS / Gaussian Process spatial model (LOO-CV)\n"
        f"    • Final per-axis 3σ clip (iterative)\n"
        f"    • Quasar frame-tie (Milliquas)"
    )
    ax_stats.text(
        0.05,
        0.95,
        "FINAL STATISTICS",
        transform=ax_stats.transAxes,
        fontsize=18,
        va="top",
        fontfamily="monospace",
        fontweight="bold",
    )
    ax_stats.text(
        0.05,
        0.86,
        rms_txt,
        transform=ax_stats.transAxes,
        fontsize=16,
        va="top",
        fontfamily="monospace",
    )
    ax_stats.text(
        0.05,
        0.56,
        stats_body,
        transform=ax_stats.transAxes,
        fontsize=12,
        va="top",
        fontfamily="monospace",
    )
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    ax_stats.axis("off")

    plt.tight_layout()
    plt.savefig(
        plots_dir / "advanced_calibration_complete.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def generate_rms_bootstrap_histogram(matched_data, plots_dir=PLOTS_DIR):
    """
    Computes bootstrap RMS uncertainties and plots the resulting distribution.
    """
    ra_fin = matched_data["ra_final"].values
    dec_fin = matched_data["dec_final"].values
    ra_g = matched_data["ra_gaia"].values
    dec_g = matched_data["dec_gaia"].values
    dec_mid = np.deg2rad(0.5 * (dec_fin + dec_g))

    dra = (ra_fin - ra_g) * 3600.0 * np.cos(dec_mid)
    ddec = (dec_fin - dec_g) * 3600.0
    sep = np.hypot(dra, ddec)
    rms_obs = np.sqrt(np.mean(sep**2))

    rng_gen = np.random.default_rng(42)
    n_boot = 10_000
    n_stars = len(sep)
    rms_boot = np.array(
        [
            np.sqrt(np.mean(rng_gen.choice(sep, size=n_stars, replace=True) ** 2))
            for _ in range(n_boot)
        ]
    )
    rms_std = rms_boot.std(ddof=1)
    ci_lo = np.percentile(rms_boot, 2.5)
    ci_hi = np.percentile(rms_boot, 97.5)

    _, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        rms_boot * 1000,
        bins=60,
        color="#4878CF",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.85,
        label=f"Bootstrap RMS  (N={n_boot:,})",
    )
    ax.axvline(
        rms_obs * 1000,
        color="#D62728",
        lw=2.0,
        label=f"RMS observado = {rms_obs * 1000:.2f} mas",
    )
    ax.axvline(
        ci_lo * 1000,
        color="#FF7F0E",
        lw=1.5,
        ls="--",
        label=f"IC 95%: [{ci_lo * 1000:.2f}, {ci_hi * 1000:.2f}] mas",
    )
    ax.axvline(ci_hi * 1000, color="#FF7F0E", lw=1.5, ls="--")
    ax.axvspan(ci_lo * 1000, ci_hi * 1000, alpha=0.12, color="#FF7F0E")

    ax.set_xlabel("RMS  [mas]", fontsize=13)
    ax.set_ylabel("Contagem de amostras bootstrap", fontsize=12)
    title_str = (
        f"Distribuição bootstrap do RMS final\nRMS = {rms_obs * 1000:.2f} ± "
        f"{rms_std * 1000:.3f} mas  (IC 95%: {ci_lo * 1000:.2f}–{ci_hi * 1000:.2f} mas)"
    )
    ax.set_title(title_str, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "rms_bootstrap_histogram.png", dpi=150)
    plt.close()


def generate_weight_vs_magnitude_plot(matched_data, plots_dir=PLOTS_DIR):
    """
    Plots the astrometric weight assigned to each star as a function of its Gaia magnitude.
    """
    mag = matched_data["gaia_mag"].values
    weights = matched_data["mag_weight"].values

    _, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mag, weights, s=25, alpha=0.7, edgecolors="k", linewidths=0.3)

    # Theoretical SNR-based weight curve
    pixel_scale = 0.1315  # arcsec/pix (DES coadd)
    fwhm_pix = 3.9
    snr_theory = 100.0 * 10.0 ** ((16.0 - mag) / 2.5)
    snr_theory = np.clip(snr_theory, 1.0, 1.0e4)
    sigma_theory_mas = (fwhm_pix * pixel_scale / (2.355 * snr_theory)) * 1000.0
    gaia_floor = 0.05 + 0.5 * np.clip((mag - 16.0) / 5.0, 0.0, 1.0)
    sigma_total = np.sqrt(sigma_theory_mas**2 + gaia_floor**2)
    w_theory = 1.0 / sigma_total**2
    w_theory = w_theory / np.mean(w_theory) * len(w_theory)
    sort_idx = np.argsort(mag)
    ax.plot(
        mag[sort_idx],
        w_theory[sort_idx],
        "r-",
        lw=2,
        label="SNR inverse-variance model",
    )

    ax.set_xlabel("Gaia G magnitude", fontsize=12)
    ax.set_ylabel("Weight (normalised)", fontsize=12)
    ax.set_title("Astrometric Weight vs Magnitude (SNR-based)", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "weight_vs_magnitude.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# New diagnostic plots for the rigorous scientific enhancements
# (magnitude stratification, hold-out CV, DCR, CCD edge / tree-ring)
# =============================================================================


def plot_magnitude_stratification(mag_strat, plots_dir=PLOTS_DIR):
    """
    Plot astrometric RMS as a function of Gaia G magnitude, with the
    theoretical noise floor (Gaia catalog + photon noise) overlaid.
    """
    bins = mag_strat.get("magnitude_bins", [])
    if not bins:
        return
    centers = [0.5 * (b["lo"] + b["hi"]) for b in bins]
    rms_tot = [b["rms_tot_mas"] for b in bins]
    rms_ra = [b["rms_ra_mas"] for b in bins]
    rms_dec = [b["rms_dec_mas"] for b in bins]
    expected = [b["expected_total_mas"] for b in bins]
    n_stars = [b["n"] for b in bins]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Left: RMS vs magnitude (3 components + theoretical floor)
    ax = axes[0]
    ax.plot(
        centers, rms_tot, "o-", color="#222222", lw=2, markersize=8, label="Total RMS"
    )
    ax.plot(
        centers, rms_ra, "s--", color="royalblue", lw=1.5, markersize=6, label="RA RMS"
    )
    ax.plot(
        centers, rms_dec, "^--", color="crimson", lw=1.5, markersize=6, label="Dec RMS"
    )
    ax.plot(
        centers,
        expected,
        "k:",
        lw=2,
        label="Theoretical floor (√(σ²_Gaia + σ²_centroid))",
    )
    if mag_strat.get("photon_break_mag") is not None:
        ax.axvline(
            mag_strat["photon_break_mag"],
            color="orange",
            ls="--",
            alpha=0.6,
            label=f"Photon break ≈ {mag_strat['photon_break_mag']:.1f}",
        )
    ax.axvline(21.0, color="green", ls=":", alpha=0.4, label="G=21 (faint threshold)")
    ax.set_xlabel("Gaia G magnitude", fontsize=12)
    ax.set_ylabel("Residual RMS (mas)", fontsize=12)
    ax.set_title("Astrometric Precision vs Magnitude", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Right: stars per bin + faint-anchor comparison
    ax = axes[1]
    ax.bar(
        centers,
        n_stars,
        width=0.6,
        color="#4878CF",
        alpha=0.7,
        edgecolor="black",
        label="N stars in bin",
    )
    ax.set_xlabel("Gaia G magnitude", fontsize=12)
    ax.set_ylabel("Number of stars", fontsize=12, color="#4878CF")
    ax.tick_params(axis="y", labelcolor="#4878CF")
    ax.set_title("Magnitude-bin population & faint-anchor results", fontsize=13)
    fs = mag_strat.get("faint_summary")
    if fs:
        ax2 = ax.twinx()
        ax2.bar(
            [23.0],
            [fs["local_rms_mas"]],
            width=0.5,
            color="crimson",
            alpha=0.7,
            edgecolor="black",
            label=f"Faint (G≥21) local-anchor RMS = {fs['local_rms_mas']:.1f} mas\n"
            f"(N={fs['n_faint']})",
        )
        ax2.set_ylabel("Faint-anchor local RMS (mas)", fontsize=12, color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson")
        ax2.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.savefig(plots_dir / "magnitude_stratification.png", dpi=200)
    plt.close()


def plot_holdout_cv(holdout_cv_summary, plots_dir=PLOTS_DIR):
    """
    Plot the per-split train vs test RMS, with the mean +/- 1sigma band.
    """
    splits = holdout_cv_summary.get("per_split", [])
    if not splits:
        return
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    train = [s["train_rms_mas"] for s in splits]
    test = [s["test_rms_mas"] for s in splits]
    x_idx = list(range(1, len(splits) + 1))
    ax.plot(
        x_idx,
        train,
        "o-",
        color="royalblue",
        lw=2,
        markersize=8,
        label=f"Train RMS (mean = {holdout_cv_summary['train_rms_mean_mas']:.2f} mas)",
    )
    ax.plot(
        x_idx,
        test,
        "s-",
        color="crimson",
        lw=2,
        markersize=8,
        label=f"TEST RMS (mean = {holdout_cv_summary['test_rms_mean_mas']:.2f} mas)",
    )
    ax.fill_between(
        x_idx,
        [
            holdout_cv_summary["test_rms_mean_mas"]
            - holdout_cv_summary["test_rms_std_mas"]
        ]
        * len(x_idx),
        [
            holdout_cv_summary["test_rms_mean_mas"]
            + holdout_cv_summary["test_rms_std_mas"]
        ]
        * len(x_idx),
        color="crimson",
        alpha=0.15,
        label=f"±1σ band ({holdout_cv_summary['test_rms_std_mas']:.2f} mas)",
    )
    ax.set_xlabel("Split index (random 70/30 partition)", fontsize=12)
    ax.set_ylabel("Residual RMS (mas)", fontsize=12)
    ax.set_title("Strict Train/Test Hold-out Cross-Validation", fontsize=13)
    overfit = holdout_cv_summary.get("overfit_ratio", float("nan"))
    ax.text(
        0.05,
        0.95,
        f"Overfit ratio = {overfit:.3f}\n(1.00 = no overfitting)",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="wheat", ec="black", alpha=0.6),
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.savefig(plots_dir / "holdout_cv.png", dpi=200)
    plt.close()


def plot_dcr_diagnostics(matched_data, dcr_telemetry, plots_dir=PLOTS_DIR):
    """
    Plot DCR diagnostics: residual along zenith direction vs color, before
    and after the DCR correction.
    """
    if dcr_telemetry is None or not dcr_telemetry.get("applied", False):
        # Still produce a color-residual plot for diagnostic purposes
        if matched_data is None or "bp_rp" not in matched_data.columns:
            return
        color = matched_data["bp_rp"].values
        if np.all(np.isnan(color)):
            return
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        cos_dec = np.cos(np.radians(matched_data["dec_gaia"].values))
        dra = (
            (matched_data["ra_final"] - matched_data["ra_gaia"]).values
            * 3600e3
            * cos_dec
        )
        ddec = (matched_data["dec_final"] - matched_data["dec_gaia"]).values * 3600e3
        ax.scatter(color, dra, s=25, alpha=0.6, c="royalblue", label="ΔRA·cosδ")
        ax.scatter(color, ddec, s=25, alpha=0.6, c="crimson", label="ΔDec")
        ax.axhline(0, c="k", ls=":", alpha=0.4)
        ax.set_xlabel("Gaia BP−RP color", fontsize=12)
        ax.set_ylabel("Residual (mas)", fontsize=12)
        ax.set_title(
            "DCR diagnostic (residuals vs colour)\n"
            "Correction not applied (no significant DCR detected)",
            fontsize=12,
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.savefig(plots_dir / "dcr_diagnostics.png", dpi=200)
        plt.close()
        return

    # If correction was applied, plot before/after using the stored coefficients
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    color = matched_data["bp_rp"].values
    color = np.where(np.isnan(color), np.nanmedian(color), color)
    cos_dec = np.cos(np.radians(matched_data["dec_gaia"].values))
    dra = (matched_data["ra_final"] - matched_data["ra_gaia"]).values * 3600e3 * cos_dec
    ddec = (matched_data["dec_final"] - matched_data["dec_gaia"]).values * 3600e3

    # Simulate the "before" residual by re-applying the DCR correction
    a = dcr_telemetry.get("coef_a_mas", 0.0) or 0.0
    b = dcr_telemetry.get("coef_b_mas", 0.0) or 0.0
    tan_zd = dcr_telemetry.get("median_tan_zd", 0.0) or 0.0
    dra_before = dra + (a + b * color) * tan_zd * np.sin(
        np.radians(dcr_telemetry.get("median_parallactic_deg", 0.0) or 0.0)
    )
    ddec_before = ddec + (a + b * color) * tan_zd * np.cos(
        np.radians(dcr_telemetry.get("median_parallactic_deg", 0.0) or 0.0)
    )

    ax = axes[0]
    ax.scatter(
        color, dra_before, s=25, alpha=0.6, c="royalblue", label="ΔRA·cosδ (before)"
    )
    ax.scatter(color, ddec_before, s=25, alpha=0.6, c="crimson", label="ΔDec (before)")
    # Overlay the fitted DCR model
    color_grid = np.linspace(np.nanmin(color), np.nanmax(color), 100)
    dcr_pred = (a + b * color_grid) * tan_zd
    ax.plot(
        color_grid,
        dcr_pred
        * np.sin(np.radians(dcr_telemetry.get("median_parallactic_deg", 0.0) or 0.0)),
        "b-",
        lw=2,
        label=f"DCR model (RA): b={b:.1f} mas/mag",
    )
    ax.plot(
        color_grid,
        dcr_pred
        * np.cos(np.radians(dcr_telemetry.get("median_parallactic_deg", 0.0) or 0.0)),
        "r-",
        lw=2,
        label="DCR model (Dec)",
    )
    ax.axhline(0, c="k", ls=":", alpha=0.4)
    ax.set_xlabel("Gaia BP−RP color", fontsize=12)
    ax.set_ylabel("Residual (mas)", fontsize=12)
    ax.set_title(
        f"BEFORE DCR correction  (zd ≈ {dcr_telemetry.get('median_zd_deg', 0):.1f}°, "
        f"tan(zd) = {tan_zd:.3f})",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(color, dra, s=25, alpha=0.6, c="royalblue", label="ΔRA·cosδ (after)")
    ax.scatter(color, ddec, s=25, alpha=0.6, c="crimson", label="ΔDec (after)")
    ax.axhline(0, c="k", ls=":", alpha=0.4)
    ax.set_xlabel("Gaia BP−RP color", fontsize=12)
    ax.set_ylabel("Residual (mas)", fontsize=12)
    rms_before = dcr_telemetry.get("rms_before_mas", float("nan"))
    rms_after = dcr_telemetry.get("rms_after_mas", float("nan"))
    ax.set_title(
        f"AFTER DCR correction  (RMS {rms_before:.1f} → {rms_after:.1f} mas)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.savefig(plots_dir / "dcr_diagnostics.png", dpi=200)
    plt.close()


def plot_spatial_residual_map(matched_data, image_shape, plots_dir=PLOTS_DIR):
    """
    Plot a 4-panel spatial residual map: quiver (vector field) + heatmap
    of |residual| + radial profile + edge-vs-centre bar chart.
    """
    if matched_data is None or len(matched_data) < 5:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    ny, nx = image_shape
    cx, cy = nx / 2, ny / 2

    x = matched_data["x_pixel"].values
    y = matched_data["y_pixel"].values
    cos_dec = np.cos(np.radians(matched_data["dec_gaia"].values))
    dra = (matched_data["ra_final"] - matched_data["ra_gaia"]).values * 3600e3 * cos_dec
    ddec = (matched_data["dec_final"] - matched_data["dec_gaia"]).values * 3600e3
    r = np.hypot(x - cx, y - cy)
    sep = np.hypot(dra, ddec)

    # Panel 1: quiver (vector field)
    ax = axes[0, 0]
    # Subsample if too many stars
    if len(x) > 80:
        idx = np.random.default_rng(42).choice(len(x), 80, replace=False)
    else:
        idx = np.arange(len(x))
    q = ax.quiver(
        x[idx],
        y[idx],
        dra[idx],
        ddec[idx],
        sep[idx],
        cmap="viridis",
        scale=200,
        scale_units="width",
        width=0.003,
        alpha=0.85,
    )
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title(
        "Residual vector field (quiver)\nColour = |residual| (mas)", fontsize=11
    )
    ax.set_aspect("equal")
    plt.colorbar(q, ax=ax, label="|residual| (mas)")
    # Outline image border
    ax.plot([0, nx, nx, 0, 0], [0, 0, ny, ny, 0], "k-", lw=1, alpha=0.5)

    # Panel 2: heatmap of |residual|
    ax = axes[0, 1]
    scat = ax.scatter(
        x,
        y,
        c=sep,
        cmap="viridis",
        s=40,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
        vmin=0,
        vmax=np.percentile(sep, 90),
    )
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Spatial |residual| heat-map (mas)", fontsize=11)
    ax.set_aspect("equal")
    plt.colorbar(scat, ax=ax, label="|residual| (mas)")

    # Panel 3: radial profile
    ax = axes[1, 0]
    r_max = float(np.hypot(nx / 2, ny / 2))
    r_edges = np.linspace(0, r_max, 7)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    rms_per_bin = []
    n_per_bin = []
    for i in range(len(r_edges) - 1):
        mask = (r >= r_edges[i]) & (r < r_edges[i + 1])
        n_per_bin.append(int(mask.sum()))
        if mask.sum() > 0:
            rms_per_bin.append(float(np.sqrt(np.mean(sep[mask] ** 2))))
        else:
            rms_per_bin.append(float("nan"))
    ax.bar(
        r_centers,
        rms_per_bin,
        width=r_edges[1] - r_edges[0] - 5,
        color="#4878CF",
        alpha=0.7,
        edgecolor="black",
    )
    for rc, n, rms in zip(r_centers, n_per_bin, rms_per_bin):
        if not np.isnan(rms):
            ax.text(rc, rms + 0.3, f"N={n}", ha="center", fontsize=9)
    ax.set_xlabel("Radius from CCD centre (pixels)", fontsize=11)
    ax.set_ylabel("Residual RMS (mas)", fontsize=11)
    ax.set_title(
        "Radial residual profile\n(diagnostic for tree-ring effect)", fontsize=11
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: edge vs centre comparison (per-quadrant)
    ax = axes[1, 1]
    # 4 quadrants + centre + edge = 6 bins
    quadrants = [
        "LL\n(x<nx/2, y<ny/2)",
        "LR\n(x≥nx/2, y<ny/2)",
        "UL\n(x<nx/2, y≥ny/2)",
        "UR\n(x≥nx/2, y≥ny/2)",
        "Centre\n(r<r_max/4)",
        "Edge\n(r>3r_max/4)",
    ]
    masks = [
        (x < nx / 2) & (y < ny / 2),
        (x >= nx / 2) & (y < ny / 2),
        (x < nx / 2) & (y >= ny / 2),
        (x >= nx / 2) & (y >= ny / 2),
        r < r_max / 4,
        r > 3 * r_max / 4,
    ]
    rms_vals = []
    for m in masks:
        if m.sum() > 0:
            rms_vals.append(float(np.sqrt(np.mean(sep[m] ** 2))))
        else:
            rms_vals.append(0.0)
    colors = ["#4878CF", "#4878CF", "#4878CF", "#4878CF", "#2ca02c", "#d62728"]
    ax.bar(quadrants, rms_vals, color=colors, alpha=0.7, edgecolor="black")
    for i, (q, v, m) in enumerate(zip(quadrants, rms_vals, masks)):
        ax.text(i, v + 0.3, f"N={int(m.sum())}", ha="center", fontsize=9)
    ax.set_ylabel("Residual RMS (mas)", fontsize=11)
    ax.set_title("Per-quadrant + edge-vs-centre RMS", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), fontsize=8)

    plt.savefig(plots_dir / "spatial_residual_map.png", dpi=200)
    plt.close()


def plot_ccd_edge_diagnostics(ccd_edge_telemetry, plots_dir=PLOTS_DIR):
    """
    Plot the CCD edge / tree-ring correction summary.
    """
    if not ccd_edge_telemetry:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    # Left: radial profile before/after
    ax = axes[0]
    profile = ccd_edge_telemetry.get("radial_profile", [])
    if profile:
        r_centers = [0.5 * (p["r_lo_px"] + p["r_hi_px"]) for p in profile]
        rms = [p["rms_tot_mas"] for p in profile]
        ax.bar(
            r_centers,
            rms,
            width=r_centers[1] - r_centers[0] - 10 if len(r_centers) > 1 else 50,
            color="#4878CF",
            alpha=0.7,
            edgecolor="black",
        )
        for rc, p in zip(r_centers, profile):
            ax.text(rc, p["rms_tot_mas"] + 0.3, f"N={p['n']}", ha="center", fontsize=9)
        ax.set_xlabel("Radius from CCD centre (pixels)")
        ax.set_ylabel("Residual RMS (mas)")
        ax.set_title("Radial residual profile (tree-ring diagnostic)")
        ax.grid(True, alpha=0.3, axis="y")
    # Right: edge-vs-centre bar chart + RMS before/after
    ax = axes[1]
    evc = ccd_edge_telemetry.get("edge_vs_centre", {})
    labels = ["Centre", "Edge"]
    rms_vals = [evc.get("centre_rms_mas", 0.0), evc.get("edge_rms_mas", 0.0)]
    n_vals = [evc.get("centre_n", 0), evc.get("edge_n", 0)]
    colors = ["#2ca02c", "#d62728"]
    bars = ax.bar(labels, rms_vals, color=colors, alpha=0.7, edgecolor="black")
    for bar, n in zip(bars, n_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"N={n}",
            ha="center",
            fontsize=10,
        )
    ratio = evc.get("edge_rms_mas", 0.0) / max(evc.get("centre_rms_mas", 1e-6), 1e-6)
    ax.set_ylabel("Residual RMS (mas)")
    ax.set_title(f"Edge vs centre RMS\n(edge/centre ratio = {ratio:.2f})")
    ax.grid(True, alpha=0.3, axis="y")

    plt.savefig(plots_dir / "ccd_edge_diagnostics.png", dpi=200)
    plt.close()


def plot_faint_precision_comparison(comparison_data, plots_dir=PLOTS_DIR):
    """
    Produces a multi-panel figure comparing pipeline astrometric precision
    to the Gaia reference floor for faint stars.

    Panel 1: G >= faint_threshold (default G >= 21) — 0.5-mag binned
             pipeline RMS vs Gaia floor vs combined expected.

    Additional panels: one per configured magnitude range (e.g. 19 < G < 22),
             each showing the same three curves for that restricted range.
    """
    g21_bins = comparison_data.get("g21_bins", [])
    range_bins_list = comparison_data.get("range_bins_list", [])
    if not g21_bins and not range_bins_list:
        return

    n_panels = 1 + len(range_bins_list)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(7 * n_panels, 6),
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    # ---- Panel 1: G >= faint_threshold ----
    ax = axes[0]
    _draw_precision_panel(ax, g21_bins, title="G ≥ 21 (faint stars)")

    # ---- Additional panels: extra ranges ----
    for idx, entry in enumerate(range_bins_list):
        ax = axes[1 + idx]
        rlo, rhi = entry["range"]
        _draw_precision_panel(
            ax,
            entry["bins"],
            title=f"{rlo:.0f} < G < {rhi:.0f} magnitude range",
        )

    plt.savefig(plots_dir / "faint_precision_comparison.png", dpi=200)
    plt.close()


def _draw_precision_panel(ax, bins, title):
    """Draws a single precision-comparison panel on *ax*."""
    if not bins:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=12)
        return

    centers = [0.5 * (b["lo"] + b["hi"]) for b in bins]
    pipe_rms = [b["pipeline_rms_mas"] for b in bins]
    gaia_flr = [b["gaia_floor_mas"] for b in bins]
    expected = [b["expected_total_mas"] for b in bins]
    n_stars = [b["n"] for b in bins]
    improv = [b["improvement_factor"] for b in bins]

    # Left y-axis: RMS curves
    ax.plot(
        centers,
        pipe_rms,
        "o-",
        color="#222222",
        lw=2,
        markersize=7,
        label="Pipeline RMS",
    )
    ax.plot(
        centers,
        gaia_flr,
        "s--",
        color="#D62728",
        lw=1.5,
        markersize=6,
        label="Gaia floor",
    )
    ax.plot(
        centers,
        expected,
        "^:",
        color="#2ca02c",
        lw=1.5,
        markersize=6,
        label="Expected (√(floor²+centroid²))",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Gaia G magnitude", fontsize=11)
    ax.set_ylabel("Residual RMS (mas)", fontsize=11, color="#222222")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.3)

    # Right y-axis: N per bin as bars
    ax2 = ax.twinx()
    ax2.bar(
        centers,
        n_stars,
        width=0.35 * (centers[1] - centers[0]) if len(centers) > 1 else 0.3,
        color="#4878CF",
        alpha=0.25,
        edgecolor="#4878CF",
        linewidth=0.5,
        label="N stars",
    )
    ax2.set_ylabel("N stars per bin", fontsize=11, color="#4878CF")
    ax2.tick_params(axis="y", labelcolor="#4878CF")

    # Annotate improvement factor at bin centres
    for c, imp in zip(centers, improv):
        ax.annotate(
            f"{imp:.2f}x",
            (c, pipe_rms[centers.index(c)]),
            textcoords="offset points",
            xytext=(0, -14),
            fontsize=7,
            ha="center",
            color="#222222",
            alpha=0.8,
        )

    # Horizontal line at improvement = 1.0 threshold
    ax.axhline(
        y=np.nan,
        color="gray",
        ls=":",
        alpha=0.5,
        label="improvement factor (number)",
    )
