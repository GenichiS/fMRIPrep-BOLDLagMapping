"""
Visualization Module
--------------------

This module contains all functions related to generating and saving plots and images for
quality control and reporting. It uses `nilearn` for neuroimaging-specific plots
(e.g., overlaying statistical maps on anatomical images, carpet plots) and `matplotlib`
for general-purpose plotting (e.g., line plots of regressors or time series).

The functions are designed to be self-contained and produce publication-quality
figures that summarize the key outputs and intermediate steps of the pipeline.
"""

import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from matplotlib.colors import LinearSegmentedColormap
from nilearn import datasets, plotting
from nilearn.plotting import plot_carpet

logger = logging.getLogger(__name__)


def _stabilize_for_plotting(img):
    """
    Helper function to prevent division-by-zero errors in plotting.
    Finds voxels with zero variance and adds a tiny amount of noise.
    """
    data = img.get_fdata()
    stds = data.std(axis=-1)
    zero_variance_mask = stds == 0
    
    if np.any(zero_variance_mask):
        noise = np.random.normal(scale=1e-10, size=data.shape)
        data[zero_variance_mask, :] += noise[zero_variance_mask, :]
        return nib.Nifti1Image(data, img.affine, img.header)
        
    return img


def save_lag_map_screenshot(lag_map_img, output_prefix, screenshot_z_coord, tracking_method, spatial_fwhm, min_corr_threshold, screenshot_alpha=1.0, screenshot_threshold=0.1, bg_img_path=None):
    """
    Generates and saves a publication-quality screenshot of the lag map overlaid
    on a standard or user-provided anatomical background image.

    Args:
        lag_map_img (Nifti1Image): The lag map image to plot.
        output_prefix (str): The base path for the output file.
        screenshot_z_coord (int): The Z-coordinate (in mm) for the axial slice.
        tracking_method (str): Name of the tracking method used, for the title.
        spatial_fwhm (float): FWHM of the spatial smoothing, for the title.
        min_corr_threshold (float): The correlation threshold used, for the title.
        screenshot_alpha (float): Alpha (transparency) for the lag map overlay.
        screenshot_threshold (float): Threshold to make near-zero lag values transparent.
        bg_img_path (str, optional): Path to a custom background image (e.g., subject's T1w).
                                     Defaults to MNI template.
    """
    try:
        logger.info(f"Generating lag map screenshot for {Path(output_prefix).name}...")
        # Use custom background image if provided and valid, otherwise default to MNI152 template
        bg_img = bg_img_path if bg_img_path and os.path.exists(bg_img_path) else datasets.load_mni152_template(resolution=2)
        
        # Create a custom blue-black-red colormap for visualizing lags
        colors = [(0, 0, 1), (0, 0, 0), (1, 0, 0)]  # Blue -> Black -> Red
        cm = LinearSegmentedColormap.from_list("custom_bbr", colors, N=256)
        
        lag_data = lag_map_img.get_fdata()
        # Set the color bar limits to be symmetric around zero
        vmax = np.nanmax(np.abs(lag_data))
        
        # Create an informative title with key analysis parameters
        title = f"Lag Map (z={screenshot_z_coord}mm) | Track: {tracking_method.capitalize()}, Smooth: {int(spatial_fwhm)}mm, R > {min_corr_threshold}"

        fig, ax = plt.subplots(figsize=(12, 5))
        # Use nilearn's `plot_stat_map` for powerful and easy overlay plotting
        plotting.plot_stat_map(
            lag_map_img, bg_img=bg_img, display_mode="z", cut_coords=[screenshot_z_coord],
            axes=ax, title=title, cmap=cm, vmax=vmax, symmetric_cbar=True, colorbar=True,
            transparency=screenshot_alpha,
            threshold=screenshot_threshold # Values below this (in absolute seconds) will be transparent
        )
        screenshot_path = f"{output_prefix}_lagmap_z{screenshot_z_coord}.png"
        fig.savefig(screenshot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Screenshot saved to: {screenshot_path}")
    except Exception as e:
        logger.error(f"Failed to generate screenshot: {e}", exc_info=True)


def save_regressor_carpet_map(regressor_matrix, regressor_names, output_prefix):
    """
    Generates and saves a carpet map (heatmap) of the nuisance regressors.
    This plot is useful for visually inspecting the temporal structure of confounds.

    Args:
        regressor_matrix (np.ndarray): The matrix of regressors (time x regressors).
        regressor_names (list[str]): A list of names for the regressors.
        output_prefix (str): The base path for the output file.
    """
    if regressor_matrix is None or regressor_matrix.shape[1] == 0:
        return
    try:
        logger.info("Generating nuisance regressor carpet map (standardized)...")
        # Standardize (Z-score) each regressor for consistent color scaling
        regressor_matrix_std = (regressor_matrix - np.mean(regressor_matrix, axis=0)) / (np.std(regressor_matrix, axis=0) + 1e-9)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        vmax = 3 # Set color limits to +/- 3 standard deviations
        im = ax.imshow(regressor_matrix_std.T, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="none")
        fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02).set_label("Z-score")
        ax.set_title("Nuisance Regressors")
        ax.set_xlabel("Time (TR)")
        ax.set_ylabel("Regressors")
        ax.set_yticks(np.arange(len(regressor_names)))
        ax.set_yticklabels(regressor_names, fontsize=8)
        plt.tight_layout()
        carpet_map_path = f"{output_prefix}_regressor_carpet.png"
        fig.savefig(carpet_map_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Regressor carpet map saved to: {carpet_map_path}")
    except Exception as e:
        logger.error(f"Failed to generate regressor carpet map: {e}", exc_info=True)


def save_bold_comparison_carpet_map(raw_bold_img, cleaned_bold_img, post_deperfusion_img, diff_img, mask_img, output_prefix, deperfusion_target, decimation_factor=1):
    """
    Generates and saves a 4-panel carpet map to visually compare the BOLD data
    at different key stages of the pipeline: raw, after nuisance regression, and
    after deperfusion.

    Args:
        raw_bold_img (Nifti1Image): The raw BOLD data.
        cleaned_bold_img (Nifti1Image): BOLD after nuisance regression.
        post_deperfusion_img (Nifti1Image): BOLD after deperfusion.
        diff_img (Nifti1Image): The difference signal (cleaned - deperfusioned).
        mask_img (Nifti1Image): The brain mask.
        output_prefix (str): The base path for the output file.
        deperfusion_target (str): Name of the data used for deperfusion ('raw' or 'cleaned').
        decimation_factor (int): Factor by which to temporally downsample data for plotting.
    """
    try:
        logger.info(f"Generating BOLD comparison carpet plot for {output_prefix}...")

        def decimate_img(img, factor):
            if factor <= 1:
                return img
            logger.info(f"Decimating carpet plot data by factor of {factor} for faster plotting...")
            data = img.get_fdata()
            num_timepoints = data.shape[-1]
            trim_len = (num_timepoints // factor) * factor
            trimmed_data = data[..., :trim_len]
            decimated_data = trimmed_data.reshape(trimmed_data.shape[:-1] + (-1, factor)).mean(axis=-1)
            new_header = img.header.copy()
            new_header.set_zooms(img.header.get_zooms()[:-1] + (img.header.get_zooms()[-1] * factor,))
            return nib.Nifti1Image(decimated_data, img.affine, new_header)

        raw_for_plot = decimate_img(raw_bold_img, decimation_factor)
        cleaned_for_plot = decimate_img(cleaned_bold_img, decimation_factor)
        deperf_for_plot = decimate_img(post_deperfusion_img, decimation_factor)
        diff_for_plot = decimate_img(diff_img, decimation_factor)

        fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
        
        raw_stabilized = _stabilize_for_plotting(raw_for_plot)
        cleaned_stabilized = _stabilize_for_plotting(cleaned_for_plot)
        deperf_stabilized = _stabilize_for_plotting(deperf_for_plot)
        diff_stabilized = _stabilize_for_plotting(diff_for_plot)

        plot_carpet(raw_stabilized, mask_img=mask_img, axes=axes[0], standardize="zscore_sample")
        axes[0].set_title("1. BOLD Data (Raw)", fontsize=16)
        
        plot_carpet(cleaned_stabilized, mask_img=mask_img, axes=axes[1], standardize="zscore_sample")
        axes[1].set_title("2. BOLD Data (Cleaned: Nuisance Regression)", fontsize=16)
        
        plot_carpet(deperf_stabilized, mask_img=mask_img, axes=axes[2], standardize="zscore_sample")
        axes[2].set_title(f'3. BOLD Data After Deperfusion (from "{deperfusion_target}" data)', fontsize=16)
        
        plot_carpet(diff_stabilized, mask_img=mask_img, axes=axes[3], standardize="zscore_sample")
        axes[3].set_title("4. Difference (Removed Signal Component)", fontsize=16)
        
        axes[3].set_xlabel("Time (scans)", fontsize=14)
        fig.suptitle("BOLD Deperfusion Comparison", fontsize=20, y=1.0)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        carpet_map_path = f"{output_prefix}_bold_comparison_carpet.png"
        fig.savefig(carpet_map_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"BOLD comparison carpet map saved to: {carpet_map_path}")
    except Exception as e:
        logger.error(f"Failed to generate BOLD comparison carpet map: {e}", exc_info=True)


def save_tukey_window_plot(window_data, output_prefix, plot_title):
    """
    Generates and saves a simple line plot of the Tukey window used for temporal filtering.

    Args:
        window_data (np.ndarray): The 1D array of window weights.
        output_prefix (str): The base path for the output file.
        plot_title (str): The title for the plot.
    """
    try:
        logger.info(f"Generating plot for: {plot_title}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(window_data)
        ax.set_title(plot_title)
        ax.set_xlabel("Timepoints")
        ax.set_ylabel("Window Weight")
        ax.set_ylim(0, 1.1)
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        filename_suffix = plot_title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        plot_path = f"{output_prefix}_{filename_suffix}.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        logger.info(f"Tukey window plot saved to: {plot_path}")
    except Exception as e:
        logger.error(f"Failed to generate Tukey window plot: {e}", exc_info=True)


def save_seed_signal_plot(seed_signal, tr, output_prefix):
    """
    Generates and saves a line plot of the prototypical Lag=0 seed signal time course.
    This is useful for visually inspecting the primary physiological signal being tracked.

    Args:
        seed_signal (np.ndarray): The 1D time series of the seed signal.
        tr (float): The repetition time, for creating the time axis in seconds.
        output_prefix (str): The base path for the output file.
    """
    try:
        logger.info("Generating plot for the prototypical Lag=0 seed signal...")
        fig, ax = plt.subplots(figsize=(12, 5))
        time_axis_seconds = np.arange(len(seed_signal)) * tr
        ax.plot(time_axis_seconds, seed_signal, color='k', linewidth=1.5)
        ax.set_title("Prototypical Lag=0 Seed Signal Time Course", fontsize=16)
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel("Signal Amplitude (A.U.)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xlim(0, time_axis_seconds[-1])
        plt.tight_layout()
        plot_path = f"{output_prefix}_seed_signal_plot.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        logger.info(f"Seed signal plot saved to: {plot_path}")
    except Exception as e:
        logger.error(f"Failed to generate seed signal plot: {e}", exc_info=True)




def save_slfo_rainbow_plot(seed_time_series, tr_track, output_prefix, run_lengths_track, lim_seconds=7.5):
    """Aso's 'lag structure' view (boldlag viewer.lag_structure_plot).

    Every tracking step's seed (sLFO) is shifted to the voxels it represents - the regressor that
    deperfusion removes from that lag bin - and drawn as one line coloured by its lag; run junctions
    are dashed. Returns the Pearson r between the lag-0 seed shifted by +max and by -max steps: the
    similarity of the two boundary references that Aso's issue-#3 argument is about (near 1 = the
    boundary bins are the same waveform and cannot be told apart). NaN when there are no seeds.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from .lag_estimators_matlab import mean_padded_shift

    keys = sorted(int(k) for k in seed_time_series)
    if not keys:
        return float("nan")
    max_step = max(abs(k) for k in keys)
    s0 = np.asarray(seed_time_series[0] if 0 in seed_time_series else seed_time_series[keys[0]], dtype=float)
    boundary_r = float("nan")
    if max_step > 0:
        a = mean_padded_shift(s0, max_step, np)
        b = mean_padded_shift(s0, -max_step, np)
        if a.std() > 0 and b.std() > 0:
            boundary_r = float(np.corrcoef(a, b)[0, 1])
    norm = colors.Normalize(-lim_seconds, lim_seconds)
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.set_facecolor((0.5, 0.5, 0.5))
    for k in sorted(keys, key=lambda q: -abs(q)):                      # large |lag| first, lag 0 on top
        s = mean_padded_shift(np.asarray(seed_time_series[k], dtype=float), -k, np)   # align to the voxels of lag k
        ax.plot(np.arange(len(s)) * tr_track, s, color=cm.RdBu_r(norm(k * tr_track)), lw=1.1)
    edge = 0
    for n in list(run_lengths_track)[:-1]:
        edge += int(n)
        ax.axvline(edge * tr_track, color="k", ls="--", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("sLFO (std units)")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=ax, pad=0.01,
                 label="lag (s)   red (+): early / upstream    blue (-): late / downstream")   # sign: + leads the seed
    ax.set_title(f"shifted sLFO per lag step (step {tr_track:g} s, {len(keys)} steps); "
                 f"r(seed0 shifted +{max_step}, -{max_step}) = {boundary_r:+.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_slfo_rainbow.png", dpi=100)
    plt.close(fig)
    return boundary_r


def boundary_seed_similarity(seed_time_series):
    """r between the lag-0 seed shifted by +max and by -max tracking steps (no figure).
    Same quantity save_slfo_rainbow_plot returns; used when QC figures are not requested."""
    import numpy as np
    from .lag_estimators_matlab import mean_padded_shift
    keys = sorted(int(k) for k in seed_time_series)
    if not keys or 0 not in seed_time_series:
        return float("nan")
    max_step = max(abs(k) for k in keys)
    if max_step == 0:
        return float("nan")
    s0 = np.asarray(seed_time_series[0], dtype=float)
    a = mean_padded_shift(s0, max_step, np)
    b = mean_padded_shift(s0, -max_step, np)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
