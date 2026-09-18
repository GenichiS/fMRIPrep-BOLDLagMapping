"""
Lag Pre-processing Module
-------------------------

This module contains all functions that are applied *before* the
main lag estimation algorithms are run. This includes nuisance correction,
temporal and spatial filtering, and other data cleaning steps.
"""

import logging
import numpy as np
import nibabel as nib
import os
from nilearn import image as nilearn_image
from scipy.ndimage import (
    gaussian_filter,
    distance_transform_edt,
)
from scipy.signal.windows import tukey
from scipy.ndimage import convolve1d
from tqdm import tqdm

from . import utils

logger = logging.getLogger(__name__)

# Counts of the last spike-detection call (dvars / fd / union / final), copied by
# core.py into the run's stats sidecar right after the call. A module-level stash keeps the public
# signatures of the two detectors unchanged (tests unpack them as 2-tuples).
LAST_SPIKE_COUNTS = {}


def dilate_mask_mm_accurate(mask_img, dilate_mm):
    """
    Dilates a binary mask by a specified distance in millimeters, correctly
    handling anisotropic voxels using a Euclidean distance transform.

    Args:
        mask_img (Nifti1Image): The binary mask image to dilate.
        dilate_mm (float): The distance in millimeters to dilate the mask.

    Returns:
        Nifti1Image: The dilated mask image.
    """
    if not (dilate_mm and dilate_mm > 0):
        return mask_img
        
    logger.info(f"Dilating mask by {dilate_mm}mm using Euclidean distance transform.")
    mask_data = mask_img.get_fdata().astype(bool)
    
    # Get voxel dimensions in mm from the image affine
    voxel_sizes_mm = nib.affines.voxel_sizes(mask_img.affine)
    
    # Calculate the distance from each non-mask voxel to the nearest mask voxel, in mm.
    # The `sampling` parameter correctly handles anisotropic voxels.
    distance_map_mm = distance_transform_edt(~mask_data, sampling=voxel_sizes_mm)
    
    # The new mask includes all original voxels plus any non-mask voxels
    # that are within the specified mm distance of the original mask.
    dilated_mask_data = (distance_map_mm <= dilate_mm)
    
    return nib.Nifti1Image(
        dilated_mask_data.astype(np.uint8), 
        mask_img.affine, 
        mask_img.header
    )


def mask_to_masker_index(mask_img, masker):
    """Return a boolean vector over the masker's voxels marking those inside `mask_img`.

    Used to restrict DVARS spike detection to the undilated brain mask while the analysis masker
    is the dilated one. The mask is resampled (nearest) onto the masker's grid, so a
    mask on a different grid is handled; the result must be non-empty.
    """
    on_grid = nilearn_image.resample_to_img(
        mask_img, masker.mask_img_, interpolation="nearest", force_resample=True, copy_header=True)
    index_1d = np.asarray(masker.transform(on_grid)).flatten() > 0.5
    if index_1d.shape[0] != int(masker.mask_img_.get_fdata().astype(bool).sum()):
        raise ValueError("mask_to_masker_index: transformed length does not match the masker's voxel count")
    if not index_1d.any():
        raise ValueError("mask_to_masker_index: the mask has no voxels inside the analysis mask")
    return index_1d


def calculate_motion_derivatives_and_fd(motion_params_np, motion_names):
    """
    Calculates temporal derivatives of motion parameters and Framewise Displacement (FD).
    This generates a more comprehensive set of motion-related nuisance regressors,
    following the common practice of including derivatives to capture the rate of motion.

    Args:
        motion_params_np (np.ndarray): Array of motion parameters (time x 6).
        motion_names (list[str]): Column names in the SAME order as the array's columns.
            REQUIRED to tell rotations from translations; omitting them is an error, because a
            positional guess (translations 0:3, rotations 3:6) is wrong for the order in which
            io.load_motion_confounds requests fMRIPrep's columns.

    Returns:
        tuple[np.ndarray, list[str]]: A tuple of the full regressor matrix and their names.

    Columns are resolved BY NAME. io.load_motion_confounds requests the fMRIPrep columns in the order
    rot_x, rot_y, rot_z, trans_x, trans_y, trans_z; assuming translations in columns 0:3 would multiply
    the TRANSLATIONS by head_radius_mm, treat radians as millimetres, and change the FD regressor and
    therefore the nuisance regression.
    """
    logger.info("Calculating motion derivatives and Framewise Displacement (FD)...")
    # Calculate backward-looking temporal derivatives (difference from the previous timepoint)
    derivatives_np = np.diff(motion_params_np, axis=0)
    derivatives_backward = np.vstack([np.zeros(motion_params_np.shape[1]), derivatives_np])
    # Calculate forward-looking temporal derivatives (difference to the next timepoint)
    derivatives_forward = np.vstack([derivatives_np, np.zeros(motion_params_np.shape[1])])
    
    # Calculate Framewise Displacement (based on Power et al., 2012)
    head_radius_mm = 50  # An assumed radius of the head in mm, for converting rotations to displacements
    # Resolve which columns are rotations and which are translations BY NAME (see the docstring).
    n_cols = motion_params_np.shape[1]
    if motion_names is None or len(motion_names) != n_cols:
        raise ValueError(
            "motion_names is required and must have one entry per column (got "
            f"{'None' if motion_names is None else len(motion_names)} names for {n_cols} columns); "
            "refusing to guess rotations/translations positionally.")
    basic_names = [str(n) for n in motion_names]
    rot_idx = [i for i, n in enumerate(basic_names) if n.startswith("rot_")]
    tra_idx = [i for i, n in enumerate(basic_names) if n.startswith("trans_")]
    if len(rot_idx) != 3 or len(tra_idx) != 3:
        raise ValueError(
            f"Cannot identify 3 rotation and 3 translation columns from names {basic_names}; "
            "refusing to compute FD from unknown columns.")
    logger.info(f"FD: rotation columns {rot_idx} {[basic_names[i] for i in rot_idx]}, "
                f"translation columns {tra_idx} {[basic_names[i] for i in tra_idx]}.")
    # Convert rotational parameters (in radians) to displacements on the surface of the sphere
    rotational_displacements = np.abs(np.diff(motion_params_np[:, rot_idx] * head_radius_mm, axis=0))
    translational_displacements = np.abs(np.diff(motion_params_np[:, tra_idx], axis=0))

    # FD is the sum of the absolute displacements
    fd = np.sum(translational_displacements, axis=1) + np.sum(rotational_displacements, axis=1)
    # The first timepoint has no preceding volume, so its FD is conventionally set to 0
    fd = np.insert(fd, 0, 0)

    # Combine original motion params, their derivatives, and FD into one comprehensive matrix for regression
    all_motion_regressors = np.hstack([motion_params_np, derivatives_backward, derivatives_forward, fd.reshape(-1, 1)])
    names = basic_names + [f'{n}_deriv_b' for n in basic_names] + [f'{n}_deriv_f' for n in basic_names] + ['framewise_displacement']
    logger.info(f"Generated {all_motion_regressors.shape[1]} motion-based regressors.")
    return all_motion_regressors, names


def compute_dvars_psc(timeseries_raw_np):
    """DVARS on percent-signal-change (scale-invariant), first frame 0.

    Used by the spike detectors when --dvars-definition psc. Each voxel is converted to PSC first so
    DVARS is independent of the input's intensity scale (HCP grand-mean 10000, fMRIPrep, raw EPI).
    See compute_dvars_boldlag for why the default is the brain-mean-normalised definition.
    """
    voxel_means = np.mean(timeseries_raw_np, axis=0)
    safe_means = np.where(np.abs(voxel_means) < 1e-9, 1.0, voxel_means)
    psc = 100.0 * (timeseries_raw_np / safe_means - 1.0)
    dvars = np.sqrt(np.mean(np.square(np.diff(psc, axis=0)), axis=1))
    return np.insert(dvars, 0, 0.0)


def compute_dvars_boldlag(timeseries_raw_np):
    """DVARS as Dr Aso's boldlag defines it (`einsteining.dvars_spikes`, boldlag v0.2.0 commit 09207f4,
    ported verbatim): within the brain -- voxels that are never zero and have a positive mean -- the RMS
    over voxels of the frame-to-frame difference of the RAW intensity, in percent of the mean brain
    signal; first frame 0. The input here is (time x voxels) and already brain-only (undilated mask); the
    never-zero / positive-mean selection is applied to the columns exactly as the original applies it to
    the 4D array (HCP volumes are zero outside the brain). This is the default definition.

    Why not per-voxel PSC (compute_dvars_psc): it divides by each voxel's own mean, so voxels inside the
    brain mask whose mean is far below the brain mean (signal dropout, the edge of the field of view)
    get enormous PSC values and can dominate DVARS^2, so that the series no longer follows head motion.
    Normalising by the mean brain signal, as this definition does, has no such weakness.
    """
    Y = np.asarray(timeseries_raw_np)
    T = Y.shape[0]
    brain = (Y != 0).all(axis=0) & (Y.mean(axis=0) > 0)
    if not brain.any():
        logger.warning("compute_dvars_boldlag: no voxel is never-zero with a positive mean; DVARS set to 0.")
        return np.zeros(T)
    Ym = Y[:, brain].astype(np.float64)                       # original: Y[brain] as (V, T); same numbers
    dvars = np.r_[0.0, np.sqrt(np.mean(np.diff(Ym, axis=0) ** 2, axis=1))] / Ym.mean() * 100
    return dvars


DVARS_DEFINITIONS = {"psc": compute_dvars_psc, "boldlag": compute_dvars_boldlag}


def aso_median_rule_count(dvars, factor=1.5):
    """Frames flagged by Aso's fixed Einsteining rule (issue #2 fix; boldlag einsteining.dvars_spikes):
    DVARS > factor * median(DVARS), frames only (no preceding frame). Always evaluated on
    compute_dvars_boldlag() output, so the logged / sidecar count is the one Aso's rule would give."""
    dvars = np.asarray(dvars, dtype=float)
    if dvars.size == 0:
        return 0
    return int(np.sum(dvars > factor * np.median(dvars)))


def identify_and_create_spike_regressors_aso_median(timeseries_raw_np, num_timepoints, include_preceding_spike,
                                                    fd=None, fd_threshold=0.0, factor=1.5, dvars_definition="boldlag"):
    """Dr Aso's Einsteining spike rule (boldlag `dvars_spikes`, issue #2 fix): a frame is a spike when
    DVARS > factor x median(DVARS); with include_preceding_spike the preceding frame is added, exactly as
    the original does (`np.unique(np.r_[spike, spike - 1])`). The FD rule can be kept on top
    (fd_threshold > 0) or switched off (0) for the original rule. Optional; the default detector is robust-z.
    """
    if dvars_definition not in DVARS_DEFINITIONS:
        raise ValueError(f"dvars_definition must be one of {sorted(DVARS_DEFINITIONS)}, got {dvars_definition!r}")
    logger.info(f"Identifying spikes via Aso's rule: DVARS ({dvars_definition}) > {factor:g} x median"
                + (f" OR FD > {fd_threshold} mm" if fd is not None and fd_threshold and fd_threshold > 0 else ""))
    dvars = DVARS_DEFINITIONS[dvars_definition](timeseries_raw_np)
    median_dvars = np.median(dvars)
    dvars_idx = np.where(dvars > factor * median_dvars)[0] if median_dvars > 0 else np.zeros(0, dtype=int)
    aso_faithful = aso_median_rule_count(compute_dvars_boldlag(timeseries_raw_np) if dvars_definition != "boldlag" else dvars, factor)
    fd_idx = _fd_spike_indices(fd, fd_threshold, num_timepoints)
    out = _finalise_spike_regressors(dvars_idx, fd_idx, fd_threshold, num_timepoints, include_preceding_spike, "aso-median")
    LAST_SPIKE_COUNTS["aso_1p5median"] = int(aso_faithful)
    return out


def _fd_spike_indices(fd, fd_threshold, num_timepoints):
    """Frames whose framewise displacement exceeds fd_threshold (mm). Empty when no FD is given
    or the threshold is <= 0. Spikes = DVARS rule OR FD > threshold, because a brain-only DVARS rule
    alone can miss moderate motion."""
    if fd is None or fd_threshold is None or fd_threshold <= 0:
        return np.zeros(0, dtype=int)
    fd = np.nan_to_num(np.asarray(fd, dtype=float))
    if fd.shape[0] != num_timepoints:
        raise ValueError(f"FD has {fd.shape[0]} frames but the run has {num_timepoints}; refusing to flag spikes from a mispaired FD.")
    return np.where(fd > fd_threshold)[0]


def _finalise_spike_regressors(dvars_idx, fd_idx, fd_threshold, num_timepoints, include_preceding_spike, tag):
    """Union of the DVARS- and FD-flagged frames, optional preceding frame, one-hot regressors."""
    dvars_idx = np.asarray(dvars_idx, dtype=int); fd_idx = np.asarray(fd_idx, dtype=int)
    primary_spike_indices = np.union1d(dvars_idx, fd_idx)
    fd_only = np.setdiff1d(fd_idx, dvars_idx)
    logger.info(f"Spike frames ({tag}): DVARS {len(dvars_idx)}, FD>{fd_threshold} mm {len(fd_idx)} "
                f"(FD-only {len(fd_only)}), union {len(primary_spike_indices)}.")
    if include_preceding_spike and len(primary_spike_indices) > 0:
        final_spike_indices = np.union1d(primary_spike_indices, primary_spike_indices - 1)
    else:
        final_spike_indices = primary_spike_indices
    final_spike_indices = final_spike_indices[(final_spike_indices >= 0) & (final_spike_indices < num_timepoints)]
    num_spikes = len(final_spike_indices)
    LAST_SPIKE_COUNTS.clear()
    LAST_SPIKE_COUNTS.update(dvars=int(len(dvars_idx)), fd=int(len(fd_idx)),
                             union=int(len(primary_spike_indices)), final=int(num_spikes))
    if num_spikes > 0:
        logger.info(f"Creating regressors for {num_spikes} unique volumes.")
        spike_regressors = np.zeros((num_timepoints, num_spikes))
        for i, spike_idx in enumerate(final_spike_indices):
            spike_regressors[spike_idx, i] = 1
        return spike_regressors, [f"spike_TR{idx:03d}" for idx in final_spike_indices]
    return None, []


def identify_and_create_spike_regressors(timeseries_raw_np, num_timepoints, dvars_threshold, include_preceding_spike,
                                         fd=None, fd_threshold=0.0, dvars_definition="psc"):
    """
    Identifies high-motion time points ("spikes" or "scrubbing targets") using DVARS
    (temporal Derivative of timecourses, root mean squared over voxels). Volumes with
    high DVARS are flagged and modeled as separate nuisance regressors.

    Args:
        timeseries_raw_np (np.ndarray): Raw BOLD data (time x voxels) - brain voxels only.
        num_timepoints (int): Total number of timepoints.
        dvars_threshold (float): The threshold for identifying a spike.
        include_preceding_spike (bool): Whether to also flag the volume *before* a spike,
                                        a common practice to account for spin-history effects.
        fd (np.ndarray or None): Framewise displacement per frame (mm), same length as the run.
        fd_threshold (float): Frames with fd > fd_threshold are ALSO flagged (OR rule).
                              0 disables the FD rule.
        dvars_definition (str): "psc" (per-voxel percent signal change, see compute_dvars_boldlag for its
                                weakness) or "boldlag" (Aso's brain-mean-normalised DVARS; the CLI default).

    Returns:
        tuple[np.ndarray or None, list[str]]: Spike regressor matrix and names, or (None, []) if no spikes found.
    """
    if dvars_definition not in DVARS_DEFINITIONS:
        raise ValueError(f"dvars_definition must be one of {sorted(DVARS_DEFINITIONS)}, got {dvars_definition!r}")
    logger.info(f"Identifying spikes via DVARS ({dvars_definition}; robust modified-z cutoff k={dvars_threshold})"
                + (f" OR FD > {fd_threshold} mm" if fd is not None and fd_threshold and fd_threshold > 0 else ""))
    dvars = DVARS_DEFINITIONS[dvars_definition](timeseries_raw_np)

    # Flag spikes with a robust, dataset-adaptive cutoff:
    #     DVARS > median + dvars_threshold * (1.4826 * MAD)
    # 1.4826*MAD is a robust (outlier-resistant) estimator of the standard deviation, so
    # `dvars_threshold` is the number of robust SDs above the median (a "modified z-score").
    # This adapts to each run's own noise level instead of a fixed absolute number.
    median_dvars = np.median(dvars)
    mad_dvars = np.median(np.abs(dvars - median_dvars))
    robust_sd = 1.4826 * mad_dvars
    if robust_sd < 1e-9:
        logger.info("DVARS is essentially constant; no DVARS spikes flagged.")
        dvars_idx = np.zeros(0, dtype=int)
    else:
        cutoff = median_dvars + dvars_threshold * robust_sd
        dvars_idx = np.where(dvars > cutoff)[0]
    # Log what Aso's fixed Einsteining rule (DVARS > 1.5 x median) would flag, for comparison, evaluated
    # on Aso's own DVARS definition (compute_dvars_boldlag).
    aso_faithful = aso_median_rule_count(dvars if dvars_definition == "boldlag" else compute_dvars_boldlag(timeseries_raw_np))
    logger.info(f"Aso 1.5 x median rule (boldlag DVARS) would flag {aso_faithful:d} frames (comparison only, not regressed).")
    fd_idx = _fd_spike_indices(fd, fd_threshold, num_timepoints)
    out = _finalise_spike_regressors(dvars_idx, fd_idx, fd_threshold, num_timepoints, include_preceding_spike, f"robust-z/{dvars_definition}")
    LAST_SPIKE_COUNTS["aso_1p5median"] = int(aso_faithful)
    return out


def identify_and_create_spike_regressors_afyouni_nichols(timeseries_raw_np, num_timepoints,
                                                         an_alpha=0.05, an_dpd_threshold=5.0,
                                                         include_preceding_spike=True,
                                                         fd=None, fd_threshold=0.0):
    """
    Statistical DVARS spike detection following Afyouni & Nichols (2018),
    "Insight and inference for DVARS", NeuroImage 172:291-312
    (doi:10.1016/j.neuroimage.2017.12.098).

    Faithful Python port of the canonical MATLAB `DVARSCalc.m` (asoroosh/DVARS)
    with its default/recommended configuration: chi-square test ('X2'), median
    expected value, half-IQR robust variance estimator ('hIQRd') under a cube-root
    power transform (TransPower = 1/3). A frame is flagged when its DVARS is BOTH
    statistically significant (p < an_alpha / (T-1), Bonferroni across frames) AND
    practically significant (Delta-percent-D-var > an_dpd_threshold). This dual rule
    is exactly the one in the DVARSCalc docstring:
        idx = find(Stat.pvals < 0.05/(T-1) & Stat.DeltapDvar > PracticalSigThr)
    Downstream handling (canonical preceding-volume window + one-hot regressors) is
    identical to identify_and_create_spike_regressors().

    Args:
        timeseries_raw_np (np.ndarray): Raw BOLD data (time x voxels).
        num_timepoints (int): Total number of timepoints (T).
        an_alpha (float): Family-wise significance level (Bonferroni). Default 0.05.
        an_dpd_threshold (float): Practical-significance floor on Delta-%D-var. Default 5.0.
        include_preceding_spike (bool): Also flag the volume preceding each spike.

    Returns:
        tuple[np.ndarray or None, list[str]]: Spike regressor matrix and names, or (None, []).
    """
    from scipy.stats import chi2, norm

    logger.info("Identifying spikes via Afyouni-Nichols DVARS inference "
                "(X2 test, median, hIQR, power=1/3)")
    Y = np.asarray(timeseries_raw_np, dtype=np.float64)            # (T, N)

    # --- Intensity normalisation (canonical DVARSCalc: scale median->100, center per voxel).
    #     pvals and DeltapDvar are scale-invariant; per-voxel centering matters only for Avar. ---
    nonzero = ~np.all(Y == 0, axis=0)
    Y = Y[:, nonzero]
    if Y.shape[1] == 0:
        # A degenerate DVARS must NOT skip the FD rule: the rule is "DVARS OR FD > threshold".
        logger.info("AN: no non-zero voxels; DVARS contributes no spikes (FD rule still applies).")
        return _finalise_spike_regressors(np.array([], dtype=int),
                                          _fd_spike_indices(fd, fd_threshold, num_timepoints),
                                          fd_threshold, num_timepoints, include_preceding_spike,
                                          "Afyouni-Nichols, DVARS degenerate")
    md = np.median(Y.mean(axis=0))
    if md != 0:
        Y = Y / md * 100.0
    Y = Y - Y.mean(axis=0, keepdims=True)                          # center each voxel

    # --- DVARS^2 (mean over voxels of squared temporal difference) ---
    DY = np.diff(Y, axis=0)                                        # (T-1, N)
    DVARS2 = np.mean(DY ** 2, axis=1)                              # (T-1,)

    # --- Null parameters: M_DV2 = median; robust SD via half-IQR with cube-root power transform ---
    M_DV2 = float(np.median(DVARS2))
    dd = 1.0 / 3.0
    Z = DVARS2 ** dd
    M_Z = float(np.median(Z))

    def h_iqr_sd(x):  # half-IQR robust SD: (median - Q1) / 1.349 * 2
        return (np.quantile(x, 0.5) - np.quantile(x, 0.25)) / 1.349 * 2.0

    # S_DV2 = sqrt( (1/dd * M_Z^(1/dd - 1) * H_IQRsd(Z))^2 ) = |3 * M_Z^2 * H_IQRsd(Z)|
    S_DV2 = abs((1.0 / dd) * M_Z ** (1.0 / dd - 1.0) * h_iqr_sd(Z))
    if not np.isfinite(S_DV2) or S_DV2 < 1e-12 or M_DV2 < 1e-12:
        logger.info("AN: DVARS^2 distribution degenerate; DVARS contributes no spikes "
                    "(FD rule still applies).")
        return _finalise_spike_regressors(np.array([], dtype=int),
                                          _fd_spike_indices(fd, fd_threshold, num_timepoints),
                                          fd_threshold, num_timepoints, include_preceding_spike,
                                          "Afyouni-Nichols, DVARS degenerate")

    # --- Chi-square test (DVARS^2 ~ scaled chi-square) ---
    nu = 2.0 * M_DV2 ** 2 / S_DV2 ** 2                             # spatial effective DoF
    x2stat = 2.0 * M_DV2 / S_DV2 ** 2 * DVARS2
    pvals = chi2.sf(x2stat, nu)                                    # upper tail == chi2cdf(...,'upper')

    # --- Practical significance: Delta-%D-var = (DVARS2 - median)/(4*Avar)*100 ---
    avar = float(np.mean(Y ** 2))
    delta_pd_var = (DVARS2 - M_DV2) / (4.0 * avar) * 100.0

    # --- Dual criterion (Bonferroni-significant AND practically-significant) ---
    n_diff = DVARS2.shape[0]                                       # T-1
    flagged_diff = np.where((pvals < an_alpha / n_diff) & (delta_pd_var > an_dpd_threshold))[0]
    # Difference index j (jump between vol j and j+1) flags volume j+1, matching the
    # prepend-zero convention of the robust-z detector (dvars[t] flags volume t).
    primary_spike_indices = flagged_diff + 1
    logger.info(f"AN: {len(primary_spike_indices)} frames flagged "
                f"(p < {an_alpha}/{n_diff} AND Delta-%D-var > {an_dpd_threshold}; "
                f"nu={nu:.1f}, median(DVARS^2)={M_DV2:.4g}).")
    fd_idx = _fd_spike_indices(fd, fd_threshold, num_timepoints)
    return _finalise_spike_regressors(primary_spike_indices, fd_idx, fd_threshold, num_timepoints,
                                      include_preceding_spike, "afyouni-nichols")


def perform_nuisance_regression(timeseries_2d_raw_np, full_confounds_np):
    """
    Regresses out nuisance variables from the raw time series using ordinary least squares.
    The residuals of this regression form the "cleaned" BOLD signal.

    Args:
        timeseries_2d_raw_np (np.ndarray): Raw BOLD data (time x voxels).
        full_confounds_np (np.ndarray or None): Matrix of nuisance regressors.

    Returns:
        np.ndarray: The cleaned BOLD data (residuals after regression).
    """
    if full_confounds_np is None or full_confounds_np.shape[1] == 0:
        logger.info("No nuisance regressors provided. Skipping regression.")
        return timeseries_2d_raw_np
        
    logger.info(f"Regressing out {full_confounds_np.shape[1]} nuisance variables...")
    # Linear regression removes the mean of the data. We must preserve it to add back later.
    original_mean = np.mean(timeseries_2d_raw_np, axis=0)
    
    # Add an intercept (a column of ones) to the design matrix to model the mean
    # Keep the caller's dtype. full_confounds_np comes from pandas as float64; without the cast,
    # hstack + lstsq would promote a float32 BOLD to float64 and every downstream array (PSC,
    # smoothing, filtering, the concatenated matrix, the GPU transfer) would double in size.
    design_matrix = np.hstack([full_confounds_np,
                               np.ones((full_confounds_np.shape[0], 1))]).astype(timeseries_2d_raw_np.dtype, copy=False)
    
    try:
        # Solve the linear system Y = Xb for b (beta coefficients) for all voxels at once
        beta, _, _, _ = np.linalg.lstsq(design_matrix, timeseries_2d_raw_np, rcond=None)
        # Calculate the predicted signal from the confounds: Xb
        predicted_signal = design_matrix @ beta
        # The residuals are the original data minus the predicted signal: Y - Xb
        residuals = timeseries_2d_raw_np - predicted_signal
        # Add the original mean back to the residuals to restore the signal's baseline
        return residuals + original_mean
    except Exception as e:
        # Raise rather than continue with the raw data: a rank-deficient or NaN-containing design
        # would otherwise yield a lag map with NO motion and NO spike correction while the pipeline
        # reported success. Motion is a major confound of lag maps, so this must never pass unnoticed.
        logger.error(f"Nuisance regression failed: {e}", exc_info=True)
        raise RuntimeError(
            f"Nuisance regression failed ({e}); refusing to continue with unregressed data, which "
            "would silently yield a motion-contaminated lag map.") from e


def temporal_filter_fsl_replicated(mapper, data, hp_sigma, lp_sigma):
    """
    Applies a temporal band-pass filter designed to replicate FSL's `fslmaths -bptf`.
    This version is GPU-accelerated and processes data in batches to conserve memory.

    Args:
        mapper (BOLDLagMapper): The main mapper instance.
        data (np.ndarray): Input data (time x voxels).
        hp_sigma (float): High-pass filter sigma in TRs. A value <= 0 disables it.
        lp_sigma (float): Low-pass filter sigma in TRs. A value <= 0 disables it.

    Returns:
        np.ndarray: Filtered data.
    """
    logger.info("Applying FSL-replicated temporal band-pass filter...")
    xp = mapper.xp
    use_gpu = mapper.use_gpu
    num_timepoints, num_voxels = data.shape
    filtered_data = np.copy(data)

    if hp_sigma > 0:
        logger.info(f"Applying high-pass filter with sigma={hp_sigma:.2f} TRs (batch processed)...")
        kernel_radius = int(np.ceil(3 * hp_sigma))
        low_freq_component = np.zeros_like(data)
        hpf_fallbacks = []   # timepoints where the local solve failed (see below)

        for i in tqdm(range(0, num_voxels, mapper.gpu_batch_size), desc="HPF Batch Processing", disable=not mapper.verbose):
            batch_start = i
            batch_end = min(i + mapper.gpu_batch_size, num_voxels)
            data_batch_np = data[:, batch_start:batch_end]
            data_batch = xp.asarray(data_batch_np)
            low_freq_batch = xp.zeros_like(data_batch)

            for t in range(num_timepoints):
                # The window spans t-r .. t+r inclusive (2r+1 samples centred on t); without the +1 on
                # win_end it would be 2r samples centred on t-0.5.
                win_start = max(0, t - kernel_radius)
                win_end = min(num_timepoints, t + kernel_radius + 1)
                window_indices = xp.arange(win_start, win_end)
                window_data = data_batch[win_start:win_end, :]
                weights = xp.exp(-((window_indices - t) ** 2) / (2 * hp_sigma ** 2))
                design_matrix_X = xp.vstack([xp.ones_like(window_indices), window_indices - t]).T
                X_w = design_matrix_X * weights[:, xp.newaxis]
                lhs = X_w.T @ design_matrix_X
                rhs = X_w.T @ window_data

                # BUG FIX #8: Catch specific exceptions and log failures instead of bare except
                try:
                    if use_gpu:
                        beta = utils.cp.linalg.solve(lhs, rhs)
                    else:
                        beta = np.linalg.solve(utils.to_numpy(lhs, use_gpu), utils.to_numpy(rhs, use_gpu))
                        beta = xp.asarray(beta)
                    low_freq_batch[t, :] = beta[0, :]
                except (np.linalg.LinAlgError, ValueError) as e:
                    # A singular local design substitutes an unweighted window mean for that timepoint's
                    # low-frequency estimate for every voxel in the batch. Counted per run and reported
                    # at WARNING below, so it never happens without a record.
                    hpf_fallbacks.append(t)
                    logger.debug(f"HPF solve failed at timepoint t={t}, batch {i//mapper.gpu_batch_size}: {e}. Using mean fallback.")
                    low_freq_batch[t, :] = xp.mean(window_data, axis=0)

            low_freq_component[:, batch_start:batch_end] = utils.to_numpy(low_freq_batch, use_gpu)
            del data_batch, low_freq_batch
            
            # Synchronize GPU before freeing
            if use_gpu and hasattr(xp, 'cuda'):
                xp.cuda.Stream.null.synchronize()
            utils.free_gpu_memory(use_gpu)

        filtered_data = data - low_freq_component
        # Report the fallbacks. A handful at the very edges is expected (truncated windows); a large
        # count means the local design is degenerate and the low-frequency estimate is an unweighted
        # mean for those timepoints.
        if hpf_fallbacks:
            uniq = sorted(set(hpf_fallbacks))
            logger.warning(
                f"High-pass filter fell back to the window mean at {len(uniq)} of {num_timepoints} "
                f"timepoints (indices {uniq[:10]}{'...' if len(uniq) > 10 else ''}). Those timepoints' "
                "low-frequency estimate is an unweighted mean, not a weighted local fit.")

    if lp_sigma > 0:
        # Get LP filter boundary mode from mapper. Default 'reflect' (non-circular, matches
        # FSL -bptf; avoids the run end being convolved onto its start). 'wrap' kept for back-compat.
        lp_mode = getattr(mapper, 'lp_filter_mode', 'reflect')
        if lp_mode == 'wrap' and hasattr(mapper, 'bold_files') and len(mapper.bold_files) > 1:
            logger.warning(
                "LP filter mode is 'wrap' (circular boundary) on multi-run concatenated data. "
                "This means the end of the last run bleeds into the start of the first run. "
                "Consider using --lp-filter-mode nearest for multi-run data to avoid this assumption."
            )
        logger.info(f"Applying low-pass filter with sigma={lp_sigma:.2f} TRs (convolution method, mode={lp_mode})...")

        kernel_radius_lp = int(np.ceil(3 * lp_sigma))
        kernel_size_lp = 2 * kernel_radius_lp + 1

        from scipy.signal.windows import gaussian
        lp_kernel_np = gaussian(kernel_size_lp, std=lp_sigma)
        lp_kernel_np = lp_kernel_np / np.sum(lp_kernel_np)

        if use_gpu:
            try:
                import cupyx.scipy.ndimage
                data_backend = xp.asarray(filtered_data)
                lp_kernel = xp.asarray(lp_kernel_np)
                filtered_data_backend = cupyx.scipy.ndimage.convolve1d(data_backend, lp_kernel, axis=0, mode=lp_mode)
                filtered_data = utils.to_numpy(filtered_data_backend, use_gpu)
                del data_backend, filtered_data_backend
                utils.free_gpu_memory(use_gpu)
            except (ImportError, Exception) as e:
                if 'OutOfMemory' in type(e).__name__ or 'OutOfMemory' in str(e):
                    logger.warning(f"GPU out of memory for LP filter, falling back to CPU: {e}")
                    utils.free_gpu_memory(use_gpu)
                elif isinstance(e, ImportError):
                    logger.warning("CuPyX not found, falling back to CPU for convolution.")
                else:
                    raise
                filtered_data = convolve1d(filtered_data, lp_kernel_np, axis=0, mode=lp_mode)
        else:
            filtered_data = convolve1d(filtered_data, lp_kernel_np, axis=0, mode=lp_mode)

    return filtered_data


def smooth_spm_compat(img_to_smooth, fwhm_mm):
    """
    Applies spatial smoothing using a Gaussian filter.

    Args:
        img_to_smooth (Nifti1Image): The image to be smoothed (can be 3D or 4D).
        fwhm_mm (float): The full-width at half-maximum of the Gaussian kernel in mm.

    Returns:
        Nifti1Image: The smoothed image.
    """
    if fwhm_mm is None or fwhm_mm <= 0:
        return img_to_smooth

    affine = img_to_smooth.affine
    # Get voxel dimensions (in mm) from the diagonal of the affine matrix
    voxel_sizes = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    # Convert FWHM (mm) to sigma (voxel units) using the standard formula
    sigma_vox = (fwhm_mm / voxel_sizes) / np.sqrt(8 * np.log(2))
    # We don't smooth along the time axis, so its sigma is 0
    sigma_per_axis = np.append(sigma_vox, 0) if img_to_smooth.ndim == 4 else sigma_vox
    data = img_to_smooth.get_fdata(dtype=np.float32)

    logger.info(f"Applying spatial smoothing (FWHM={fwhm_mm}mm)...")
    smoothed_data = gaussian_filter(data, sigma=sigma_per_axis, mode="constant", cval=0)

    return nib.Nifti1Image(smoothed_data, affine, img_to_smooth.header)



def create_temporal_window(num_timepoints, max_lag_seconds, tr, taper_width="fixed5pct"):
    """
    Tukey (cosine) taper applied to each run before band-pass filtering, so that the HP edge
    transients and the run junctions are not carried into the shifted cross-correlations.

    taper_width:
      'fixed5pct' (default): alpha = 0.1, i.e. 5 % of the run tapered at each end - the width of the
          original implementation's drMerge4D linear ramp
          ([0:1/((N-1)*.05):1 ones(1,ceil(N*.9)) 1:-1/((N-1)*.05):0 0], Einsteining_v07.m L258, applied
          per run at L271). 12 samples per end for a 240-frame run.
      'maxlag': max_lag_trs samples per end capped at 5 % (2 samples at TR 2.5 s for a 5 s search -
          effectively no protection). Kept only to reproduce outputs of earlier versions.

    Args:
        num_timepoints (int): Length of the run.
        max_lag_seconds (float): Search range; used only by 'maxlag'.
        tr (float): Repetition time; used only by 'maxlag'.
        taper_width (str): 'fixed5pct' or 'maxlag'.

    Returns:
        np.ndarray: The 1D Tukey window.
    """
    if taper_width == "fixed5pct":
        alpha = 0.1
    elif taper_width == "maxlag":
        max_lag_trs = round(max_lag_seconds / tr)
        alpha = min((2 * max_lag_trs) / num_timepoints, 0.1)
    else:
        raise ValueError(f"taper_width must be 'fixed5pct' or 'maxlag', got {taper_width!r}")
    logger.info(f"Tukey taper: mode={taper_width}, alpha={alpha:.4f} "
                f"({alpha * (num_timepoints - 1) / 2:.1f} samples per end) for {num_timepoints} timepoints.")
    return tukey(num_timepoints, alpha=alpha)




def resample_for_tracking(ts_np, tr, step):
    """Resample a (T, V) series from the TR to a tracking step (s) - drLag4Drev7_longTR / boldlag
    lag4d.py:282: resample_poly(Y, round(TR*100), round(reso*100), axis=0, window=('kaiser', 5.0)).

    Applied AFTER band-pass filtering and the amplitude gate and BEFORE std-normalisation, so the
    integer-step estimators see the same 1 s grid at any TR. Returns float32 with
    T' = round(T * tr / step) samples. For a step longer than the TR (e.g. 0.8 s -> 1 s) this is a
    down-sample; the band-pass (<= 0.09 Hz) is far below the new Nyquist (0.5 Hz), so nothing is lost.
    """
    from scipy.signal import resample_poly
    up, down = int(round(float(tr) * 100)), int(round(float(step) * 100))
    if up == down:
        return np.asarray(ts_np, np.float32)
    return resample_poly(np.asarray(ts_np, np.float64), up, down, axis=0,
                         window=('kaiser', 5.0)).astype(np.float32)
