"""
Lag Post-processing Module
--------------------------

This module contains all functions that are applied *after* the main lag map
has been estimated. This includes refining the map by filling holes and the
final deperfusion step, where the identified lag structure is regressed out
from the BOLD signal.
"""

import logging
import numpy as np
from nilearn import image as nilearn_image
from scipy.ndimage import (
    convolve,
    generate_binary_structure,
)
from tqdm import tqdm

from . import lag_estimators_matlab, lag_preprocess  # mean_padded_shift, temporal_filter_fsl_replicated

logger = logging.getLogger(__name__)


def build_deperfusion_regressors(seed_time_series, tr, tr_track, n_tr_frames):
    """One deperfusion regressor per tracking step, on the TR grid.

    Each seed is shifted by its own lag on the grid it was tracked on (a negative step = later signal = shifted
    forward, mean-padded) and only then resampled to the TR and cut or mean-padded to `n_tr_frames`. This is the
    order of Dr Aso's drDeperf_longTR.m (Motodata from the shifted Seeds, L66-75, then resample, L78) and of
    boldlag deperf.py (L72-74). Resampling first and shifting by the nearest whole TR would give a 1 s lag no
    shift and a 2 s lag a 2.5 s one at TR 2.5 s. Without resampling (tr_track == tr) this is the plain integer
    shift.
    """
    resampled = abs(float(tr_track) - float(tr)) > 1e-9
    regressors = {}
    for lag_steps, seed in seed_time_series.items():
        r = lag_estimators_matlab.mean_padded_shift(np.asarray(seed, dtype=float), -int(lag_steps), np, axis=0)
        if resampled:
            from scipy.signal import resample_poly
            r = resample_poly(r, int(round(tr_track * 100)), int(round(tr * 100)), window=('kaiser', 5.0))
            if r.shape[0] < n_tr_frames:
                r = np.concatenate([r, np.full(n_tr_frames - r.shape[0], r.mean())])
            r = r[:n_tr_frames]
        regressors[lag_steps] = r
    return regressors


def lag_steps(lag_map_values, tr_track):
    """(steps, finite): the tracking-step bin of every lag and the mask of finite lags. The ONE place lags are binned
    for deperfusion - float64 division, round half to even. Binning the fixed-seed keys elsewhere with float32 division
    would put a filled lag of 0.4 s at TR 0.8 s in bin 0 there and in bin 1 here, so every caller uses this function."""
    lag = np.asarray(lag_map_values, dtype=np.float64)
    finite = np.isfinite(lag)
    steps = np.zeros(lag.shape, dtype=np.int64)
    steps[finite] = np.rint(lag[finite] / float(tr_track)).astype(np.int64)
    return steps, finite


def fixed_seed_keys(lag_map_values, tr_track):
    """Bins that the fixed / fixed_subtr seed must cover for deperfusion (every bin present in the map), via lag_steps."""
    steps, finite = lag_steps(lag_map_values, tr_track)
    return [int(b) for b in np.unique(steps[finite])]


def group_voxels_by_lag_bin(lag_map_values, regressor_keys, tr_track):
    """Assign each voxel to a lag region (a tracking-step bin that has a deperfusion regressor).

    Returns (groups, summary): groups maps a bin to the voxel indices in it (ordered by lag value, then index),
    summary counts n_voxels / n_regressed / n_no_lag / n_assigned_to_range_edge and gives
    regressor_range_steps [lo, hi].
    As in the original (boldlag deperf.py L91-99, drDeperf_longTR.m L109-128): regions are L = -MaxLag..MaxLag of
    the seed matrix. A voxel without a lag is in no region (temporal mean in the caller). A bin INSIDE the range
    without a regressor cannot occur, since tracking ends at the first seedless step, so it raises.
    A finite lag rounding OUTSIDE the regressor range - only sub-TR lags can, by their fraction past the last tracked
    step (a direction ended early, or --no-boundary-null) - joins the nearest edge bin: the voxel was tracked in that
    region, so it is regressed with that regressor instead of losing its fluctuations. Integer-step maps never round
    outside; the original has no sub-TR lags and therefore no rule for this case.
    Rounding is half to even on float64.
    """
    lag = np.asarray(lag_map_values, dtype=np.float64)
    steps, finite = lag_steps(lag, tr_track)
    if not finite.any():
        raise ValueError("No valid lags in the map: there is no lag region to deperfuse.")
    keys = sorted(int(k) for k in regressor_keys)
    if not keys:
        raise ValueError("No deperfusion regressors (empty seed set): there is nothing to regress.")
    lo, hi = keys[0], keys[-1]
    in_range = finite & (steps >= lo) & (steps <= hi)
    key_set = set(keys)
    gaps = sorted(int(b) for b in np.unique(steps[in_range]) if int(b) not in key_set)
    if gaps:
        n_gap = int(np.isin(steps[in_range], gaps).sum())
        raise RuntimeError(f"Deperfusion: lag bins {gaps} (tracking steps) lie inside the regressor range [{lo}, {hi}] "
                           f"but have no regressor ({n_gap} voxels). Tracking ends at the first step without seed "
                           "voxels, so the lag map and the seeds do not belong together.")
    edge = finite & ~in_range                       # regressed with the nearest edge regressor
    bins = steps.copy()
    bins[edge] = np.clip(steps[edge], lo, hi)
    groups = {}
    for b in keys:
        idx = np.where(finite & (bins == b))[0]
        if idx.size:
            groups[b] = idx[np.argsort(lag[idx], kind='stable')]
    summary = dict(n_voxels=int(lag.size), n_regressed=int(finite.sum()), n_no_lag=int((~finite).sum()),
                   n_assigned_to_range_edge=int(edge.sum()), regressor_range_steps=[lo, hi])
    return groups, summary


def apply_deperfusion_regression_matlab_compat(mapper, data_to_deperfuse_np, run_seed_time_series):
    """
    Applies deperfusion by regressing out the appropriate shifted seed signal from
    groups of voxels with the same lag.

    As in the original (boldlag deperf.py / drDeperf): the output is the temporal mean plus, inside each lag region,
    the regression residual of the high-passed data. A voxel without a lag is in no region and keeps its temporal
    mean - finite, so a bSpline warp to MNI is not poisoned by NaN; a finite lag rounding outside the regressor range
    is regressed with the nearest edge regressor; both are counted in a WARNING. No lags, no regressors, a bin inside
    the range without a regressor, a regressor of the wrong length or a failing regression stop the run (the
    original's fsl_regfilt / pinv errors stop it too).
    """
    logger.info("Applying deperfusion using updated MATLAB-compatible regression method...")
    # Lag bins are in TRACKING steps (tr_track = --tracking-step-seconds when the series was resampled, else
    # the TR), matching the keys of run_seed_time_series. core.py hands over regressors already on the TR grid
    # (build_deperfusion_regressors: shifted on the tracking grid, then resampled), so no resampling here.
    tr_track = float(getattr(mapper, 'tr_track', mapper.tr))
    groups, summary = group_voxels_by_lag_bin(mapper.lag_map_values, run_seed_time_series.keys(), tr_track)
    if summary['n_no_lag'] or summary['n_assigned_to_range_edge']:
        lo, hi = summary['regressor_range_steps']
        logger.warning(f"Deperfusion: {summary['n_no_lag']} voxels without a lag are in no lag region and keep their "
                       f"temporal mean (as drDeperf / boldlag deperf); {summary['n_assigned_to_range_edge']} voxels whose "
                       f"lag rounds outside the regressor range [{lo}, {hi}] (tracking steps) are regressed with the "
                       "nearest edge regressor.")

    t_mean = np.mean(data_to_deperfuse_np, axis=0)
    hp_sigma = 1 / (mapper.final_hp_cutoff_hz * 2.35 * mapper.tr)
    # The data must first be high-pass filtered, just as it was for lag mapping.
    hp_component_np = lag_preprocess.temporal_filter_fsl_replicated(mapper, data_to_deperfuse_np, hp_sigma, -1)
    cleaned_hp_output = np.zeros_like(hp_component_np)      # no region: temporal mean only

    # One lstsq per tracking-step bin instead of per unique float lag value
    for lag_val_trs, voxel_indices in tqdm(groups.items(), desc="Applying deperfusion", disable=not mapper.verbose):
        sLFO_seed = np.asarray(run_seed_time_series[lag_val_trs], dtype=float).reshape(-1)

        voxel_ts_group_hp = hp_component_np[:, voxel_indices]
        n_t = voxel_ts_group_hp.shape[0]
        if sLFO_seed.shape[0] != n_t:
            raise ValueError(f"Deperfusion regressor for lag bin {lag_val_trs} has {sLFO_seed.shape[0]} samples "
                             f"but the run has {n_t} frames; the seed must be on the TR grid of this run.")
        sLFO_seed = sLFO_seed.reshape(-1, 1)
        # De-mean the regressor to precisely replicate fsl_regfilt's behavior
        sLFO_seed_demeaned = sLFO_seed - np.mean(sLFO_seed)
        design_matrix = np.hstack([sLFO_seed_demeaned, np.ones_like(sLFO_seed)])

        beta, _, _, _ = np.linalg.lstsq(design_matrix, voxel_ts_group_hp, rcond=None)
        predicted_signal = design_matrix @ beta
        # The cleaned signal is the residual after subtracting the prediction
        cleaned_hp_output[:, voxel_indices] = voxel_ts_group_hp - predicted_signal

    # Add the original mean back to the high-pass filtered residuals
    return t_mean + cleaned_hp_output


def fill_holes_1by1(lag_map_3d_np, mask_3d_np, max_iterations=1000):
    """
    Fills holes (NaNs) in the lag map by iteratively averaging neighboring voxels.

    MATLAB-MATCHED BEHAVIOR (drErode_Lag):
    - Fills holes that have at least 1 valid neighbor (MATLAB uses nanmean)
    - Iterates until all fillable holes are filled (no iteration limit in MATLAB)
    - Only fills within the brain mask

    Args:
        lag_map_3d_np: 3D lag map with NaN for unassigned voxels
        mask_3d_np: Brain mask
        max_iterations: Maximum iterations (default 1000, MATLAB has no limit)
    """
    logger.info("Iteratively filling holes in the lag map (MATLAB-matched)...")
    filled_lag_map_3d = np.copy(lag_map_3d_np)

    # BUG FIX #9: Explicitly convert mask to boolean
    mask_3d_bool = mask_3d_np.astype(bool)

    # Define a 3D neighborhood structure (connectivity=1, i.e., faces touching)
    structure = generate_binary_structure(3, 1)

    # Count initial assigned voxels
    initial_assigned = np.sum(~np.isnan(lag_map_3d_np) & mask_3d_bool)
    logger.info(f"  Initial assigned voxels: {initial_assigned:,}")

    total_filled = 0
    for iteration in range(max_iterations):
        unassigned_voxels_mask = np.isnan(filled_lag_map_3d) & mask_3d_bool

        # If there are no more holes to fill, we are done
        if not np.any(unassigned_voxels_mask):
            logger.info(f"Hole filling completed in {iteration} iterations. Total filled: {total_filled:,}")
            return filled_lag_map_3d

        # Use convolution to count valid neighbors.
        # A neighbour counts as valid only if it is BOTH non-NaN AND inside the analysis mask.
        # Out-of-mask voxels arrive as 0.0 from masker.inverse_transform, not NaN; counting them would
        # drag every boundary hole toward zero (a hole whose in-mask neighbours are all +5.0 s would be
        # filled with +4.17 s). MATLAB's nanmean over six circshifted copies has the same intent: only
        # real values contribute.
        valid_data = ((~np.isnan(filled_lag_map_3d)) & mask_3d_bool).astype(float)
        neighbor_count = convolve(valid_data, structure, mode="constant", cval=0.0)
        sum_of_neighbors = convolve(np.nan_to_num(filled_lag_map_3d) * valid_data,
                                    structure, mode="constant", cval=0.0)

        mean_neighbor_lag = np.full_like(filled_lag_map_3d, np.nan)
        eligible_for_filling = neighbor_count > 0
        mean_neighbor_lag[eligible_for_filling] = sum_of_neighbors[eligible_for_filling] / neighbor_count[eligible_for_filling]

        # MATLAB-MATCHING: Fill holes with at least 1 valid neighbor
        # MATLAB uses nanmean which fills as long as there's at least one non-NaN neighbor
        fill_mask = unassigned_voxels_mask & (neighbor_count >= 1)

        n_filled_this_iter = np.sum(fill_mask)
        if n_filled_this_iter == 0:
            logger.info(f"No more holes could be filled after {iteration} iterations. Total filled: {total_filled:,}")
            break

        filled_lag_map_3d[fill_mask] = mean_neighbor_lag[fill_mask]
        total_filled += n_filled_this_iter

    logger.info(f"Hole filling stopped after {max_iterations} iterations. Total filled: {total_filled:,}")
    return filled_lag_map_3d


def fill_isolated_holes_single_pass(lag_map_3d_np, mask_3d_np):
    """
    Single-pass hole filling for isolated NaN voxels; replicates MATLAB's `drErode1`.
    Fills voxels that are NaN AND have at least one valid neighbour (MATLAB nanmean).

    `mask_3d_np` is REQUIRED. Without it the neighbour test would be `~np.isnan(lag_map)`, and
    `masker.inverse_transform` writes 0.0 (not NaN) outside the brain, so out-of-mask zeros would count
    as valid neighbours and pull edge holes toward 0 s (a true +5.0 s island would be filled with
    +2.5 s). `fill_holes_1by1` uses the same `& mask` guard, so both passes are consistent.
    A voxel with no measured in-mask neighbour stays NaN (never measured), which is what the
    validity mask reports.
    """
    logger.info("Performing single-pass hole filling on final lag map...")
    filled_map = np.copy(lag_map_3d_np)
    if mask_3d_np is None:
        raise ValueError("fill_isolated_holes_single_pass requires the analysis mask. Without it, "
                         "out-of-mask zeros written by masker.inverse_transform count as valid "
                         "neighbours and pull edge holes toward 0 s. "
                         "There is no safe default, so the argument is mandatory.")
    mask_3d_bool = np.asarray(mask_3d_np).astype(bool)
    if mask_3d_bool.shape != lag_map_3d_np.shape:
        raise ValueError(f"Mask shape {mask_3d_bool.shape} does not match lag map "
                         f"{lag_map_3d_np.shape}; refusing to fill holes against the wrong grid.")

    nan_mask = np.isnan(lag_map_3d_np) & mask_3d_bool
    if not np.any(nan_mask):
        logger.info("No isolated holes to fill.")
        return filled_map

    structure = generate_binary_structure(3, 1)
    # A neighbour counts only if it is IN the mask and not NaN.
    valid_voxels_mask = (~np.isnan(lag_map_3d_np)) & mask_3d_bool
    masked_data = np.where(valid_voxels_mask, np.nan_to_num(lag_map_3d_np), 0.0)
    sum_of_neighbors = convolve(masked_data, structure, mode='constant', cval=0.0)
    neighbor_count = convolve(valid_voxels_mask.astype(float), structure, mode='constant', cval=0.0)

    # MATLAB-MATCHING: fill holes with at least 1 valid neighbour (nanmean behaviour)
    fill_mask = nan_mask & (neighbor_count >= 1)

    neighbor_count_safe = np.where(neighbor_count == 0, 1, neighbor_count)
    mean_neighbor_lag = sum_of_neighbors / neighbor_count_safe

    filled_map[fill_mask] = mean_neighbor_lag[fill_mask]
    n_filled = int(np.sum(fill_mask))
    n_left = int(np.sum(nan_mask & ~fill_mask))
    logger.info(f"Single-pass filled {n_filled:,} isolated holes; {n_left:,} in-mask voxels have no "
                "measured neighbour and stay NaN (never measured).")
    return filled_map



def null_search_boundary(lag_map_3d, mask_bool, max_lag_trs, tr_track, enabled):
    """Identify (and, if enabled, null) the outermost search bin of a lag map.

    drLag4Drev7.m drErode_Lag nulls abs(Y) >= MaxLag (release rev8hcp L354/L536). This is deliberate, as
    discussed in issue #3 of the original repository: the range-linked LP cut-off makes the two boundary bins
    near-identical waveforms, so a peak at the end of the curve is unreliable. The threshold is
    (max_lag_trs - 0.5) * tr_track, i.e. the KNOWN search boundary. For integer maps this is exactly the
    literal rule (the outer bin). For sub-TR maps it is a SECONDS threshold on the phase-corrected value,
    so part of the outer bin survives and part of the next bin can go (TR 2.5 s, 3 TR: |lag| >= 6.25 s
    nulled, < 6.25 s kept).

    Returns (lag_map_3d_out, n_candidates, n_nulled, threshold_s). Candidates are counted even when
    nulling is disabled, so boundary saturation can be reported for every configuration; the input
    array is never modified in place.
    """
    mask_bool = np.asarray(mask_bool, bool)
    threshold_s = float((int(max_lag_trs) - 0.5) * float(tr_track))
    with np.errstate(invalid="ignore"):
        cand = (np.abs(lag_map_3d) >= threshold_s) & mask_bool
    n_cand = int(cand.sum())
    if not enabled or n_cand == 0:
        return lag_map_3d, n_cand, 0, threshold_s
    out = np.array(lag_map_3d, dtype=float, copy=True)
    out[cand] = np.nan
    logger.info(f"Nulling {n_cand:,} search-boundary voxels (|lag| >= {threshold_s:.2f} s) before hole-filling (canonical drErode_Lag).")
    return out, n_cand, n_cand, threshold_s
