"""
Lag Estimation Module (MATLAB Replications)
-------------------------------------------

This module contains the lag estimation algorithms that are designed to be
direct replications of the original MATLAB scripts by Dr. Aso. These methods
typically operate in the time domain, use local peak searches, and do not
provide sub-TR precision, matching the behavior of the original code for
validation and comparison purposes.
"""
import logging
import numpy as np
from tqdm import tqdm

from . import utils

logger = logging.getLogger(__name__)


def _tr_track(mapper):
    """Tracking step in seconds: mapper.tr_track when the series was resampled (--tracking-step-seconds),
    else the TR. Every lag in this module is scaled by this value."""
    return float(getattr(mapper, "tr_track", mapper.tr))


def mean_padded_shift(array, shift, xp, axis=0):
    """
    Performs a non-circular shift on an array, padding with the array's mean value.
    This is a specific operation used by the MATLAB replication methods.
    """
    if shift == 0: return array
    padding_values = xp.mean(array, axis=axis, keepdims=True)
    shifted_array = xp.empty_like(array)
    if shift > 0: # Positive shift moves data "down"
        shifted_array[shift:] = array[:-shift]
        shifted_array[:shift] = padding_values
    else: # Negative shift moves data "up"
        shifted_array[:shift] = array[-shift:]
        shifted_array[shift:] = padding_values
    return shifted_array


def find_local_peak_time_domain(timeseries_backend, reference_ts_backend, xp, window_trs=1):
    """
    Calculates correlation in the time domain within a small local window.

    MATLAB-MATCHED IMPLEMENTATION:
    Uses a fixed-length trimmed window for ALL lags to ensure equal sample sizes.
    MATLAB: YY = cat(3, YY, Y(Lim+Sft+1:end-Lim+Sft, :)) for Sft in Lim:-1:-Lim
    This trims Lim samples from BOTH ends, creating windows of length (n - 2*Lim).

    BUG FIX #10: Added validation for sufficient timepoints before processing
    """
    n_timepoints, n_voxels = timeseries_backend.shape
    lim = window_trs

    # Validate sufficient timepoints
    min_required_timepoints = 2 * lim + 1

    if n_voxels == 0:
        logger.warning("No voxels to process in find_local_peak_time_domain.")
        return xp.array([]), xp.array([])

    if n_timepoints < min_required_timepoints:
        # A correlation that could not be computed is returned as NaN, not 0.0: a neutral-looking zero
        # would be indistinguishable from a real value. The voxels are therefore unambiguously unmeasured.
        logger.warning(
            f"Insufficient timepoints ({n_timepoints}) for window_trs={window_trs}. "
            f"Minimum required: {min_required_timepoints}. Returning NaN correlations "
            "(voxels will be left unassigned, not assigned a lag of 0)."
        )
        return xp.zeros(n_voxels), xp.full(n_voxels, xp.nan)

    # MATLAB-MATCHING: Use trimmed reference signal (same for all lags)
    # MATLAB: XX = repmat( Seed( Lim+1:end-Lim), ...)
    ref_trimmed = reference_ts_backend[lim:-lim] if lim > 0 else reference_ts_backend

    # MATLAB iterates Sft from Lim to -Lim (i.e., [2,1,0,-1,-2] for Lim=2)
    # Index mapping: Sft=Lim -> lag=-Lim, Sft=0 -> lag=0, Sft=-Lim -> lag=+Lim
    # So shifts = [Lim, Lim-1, ..., 0, ..., -Lim+1, -Lim]
    shifts = list(range(lim, -lim - 1, -1))  # e.g., [2, 1, 0, -1, -2] for lim=2

    corrs_in_window = xp.full((len(shifts), n_voxels), -1.0, dtype=xp.float32)

    for i, sft in enumerate(shifts):
        # MATLAB: Y(Lim+Sft+1:end-Lim+Sft, :)
        # Python (0-indexed): Y[lim+sft : n_timepoints-lim+sft, :]
        start_idx = lim + sft
        end_idx = n_timepoints - lim + sft
        ts_window = timeseries_backend[start_idx:end_idx, :]

        # MATLAB-exact correlation: sum(XX.*YY) / (sqrt(sum(XX^2)) * sqrt(sum(YY^2)))
        XX = ref_trimmed[:, xp.newaxis]
        YY = ts_window

        numerator = xp.sum(XX * YY, axis=0)
        xx_sum_sq = xp.sum(XX * XX, axis=0)
        yy_sum_sq = xp.sum(YY * YY, axis=0)
        denominator = xp.sqrt(xx_sum_sq) * xp.sqrt(yy_sum_sq)

        # Avoid division by zero
        denominator = xp.where(denominator == 0, 1e-10, denominator)
        correlation = numerator / denominator

        corrs_in_window[i, :] = correlation

    # Find peak correlation and corresponding lag
    peak_indices_in_window = xp.argmax(corrs_in_window, axis=0)

    # Convert shift index to lag in TRs
    # MATLAB: I==center_index means lag=0
    # shifts[i] = sft, where sft=0 is lag=0, sft>0 is negative lag, sft<0 is positive lag
    # Actually: sft corresponds to how much data is shifted, so lag = -sft
    lags_trs = xp.asarray([-shifts[int(idx)] for idx in peak_indices_in_window])
    max_corrs = corrs_in_window[peak_indices_in_window, xp.arange(n_voxels)]

    return lags_trs, max_corrs
    

def compute_lag_maps_fixed_seed(mapper, timeseries_2d_filtered_backend, global_signal_backend):
    """
    Computes lag maps using a fixed seed with an iterative search loop, replicating
    the `FIXED=1` mode from the MATLAB script `drLag4Drev7`.
    """
    logger.info("--- RUNNING MATLAB-COMPATIBLE FIXED (ITERATIVE & LOCAL) LAG MAPPING ---")
    max_lag_trs = int(round(mapper.max_lag_seconds / _tr_track(mapper)))
    lag_map_values = mapper.xp.full(mapper.num_voxels, mapper.xp.nan)
    max_correlations = mapper.xp.full(mapper.num_voxels, mapper.xp.nan)
    assigned_voxels = mapper.xp.zeros(mapper.num_voxels, dtype=bool)

    initial_lags_trs, initial_corrs = find_local_peak_time_domain(
        timeseries_2d_filtered_backend, global_signal_backend, mapper.xp, window_trs=2
    )
    lag_0_mask = (initial_lags_trs == 0) & (initial_corrs >= mapper.min_corr_threshold)
    lag_0_indices = mapper.xp.where(lag_0_mask)[0]
    if len(lag_0_indices) == 0:
        raise RuntimeError("Fixed tracking failed: No valid lag=0 voxels found.")

    fixed_reference_signal = mapper.xp.mean(timeseries_2d_filtered_backend[:, lag_0_indices], axis=1)
    mapper.seed_time_series[0] = utils.to_numpy(fixed_reference_signal, mapper.use_gpu)
    lag_map_values[lag_0_indices] = 0.0
    max_correlations[lag_0_indices] = initial_corrs[lag_0_indices]
    assigned_voxels[lag_0_indices] = True

    for p in tqdm(range(1, max_lag_trs + 1), desc="Fixed Iterative Search", disable=not mapper.verbose):
        unassigned_mask = ~assigned_voxels
        if not mapper.xp.any(unassigned_mask): break

        ts_shifted = mean_padded_shift(timeseries_2d_filtered_backend, -p, mapper.xp)
        lags_d, corrs_d = find_local_peak_time_domain(ts_shifted[:, unassigned_mask], fixed_reference_signal, mapper.xp, window_trs=1)
        downstream_mask = (lags_d == 0) & (corrs_d >= mapper.min_corr_threshold)
        if mapper.xp.any(downstream_mask):
            newly_found_indices = mapper.xp.where(unassigned_mask)[0][downstream_mask]
            lag_map_values[newly_found_indices] = -p * _tr_track(mapper)
            max_correlations[newly_found_indices] = corrs_d[downstream_mask]
            assigned_voxels[newly_found_indices] = True

        unassigned_mask = ~assigned_voxels
        if not mapper.xp.any(unassigned_mask): break
        ts_shifted = mean_padded_shift(timeseries_2d_filtered_backend, p, mapper.xp)
        lags_u, corrs_u = find_local_peak_time_domain(ts_shifted[:, unassigned_mask], fixed_reference_signal, mapper.xp, window_trs=1)
        upstream_mask = (lags_u == 0) & (corrs_u >= mapper.min_corr_threshold)
        if mapper.xp.any(upstream_mask):
            newly_found_indices = mapper.xp.where(unassigned_mask)[0][upstream_mask]
            lag_map_values[newly_found_indices] = p * _tr_track(mapper)
            max_correlations[newly_found_indices] = corrs_u[upstream_mask]
            assigned_voxels[newly_found_indices] = True

    return utils.to_numpy(lag_map_values, mapper.use_gpu), utils.to_numpy(max_correlations, mapper.use_gpu), utils.to_numpy(lag_0_mask, mapper.use_gpu)


def compute_lag_maps_recursive(mapper, timeseries_2d_filtered_backend, global_signal_backend):
    """
    Computes lag maps using a recursive, iterative approach that replicates the
    `FIXED=0` mode from the MATLAB script `drLag4Drev7`. The seed signal is updated
    at each step, propagating outwards from the initial seed region.

    MATLAB-MATCHED IMPLEMENTATION:
    - Shifts the DATA at each iteration (not the seed)
    - Looks for peak at lag=0 in the local ±1 TR window
    - Updates seed from newly found voxels WITHOUT shifting the seed
    """
    logger.info("--- RUNNING MATLAB-COMPATIBLE RECURSIVE LAG MAPPING (LOCAL PEAK) ---")
    max_lag_trs = int(round(mapper.max_lag_seconds / _tr_track(mapper)))
    lag_map_values = mapper.xp.full(mapper.num_voxels, mapper.xp.nan)
    max_correlations = mapper.xp.full(mapper.num_voxels, mapper.xp.nan)
    assigned_voxels = mapper.xp.zeros(mapper.num_voxels, dtype=bool)

    initial_lags_trs, initial_corrs = find_local_peak_time_domain(
        timeseries_2d_filtered_backend, global_signal_backend, mapper.xp, window_trs=2
    )
    lag_0_mask = (initial_lags_trs == 0) & (initial_corrs >= mapper.min_corr_threshold)
    lag_0_indices = mapper.xp.where(lag_0_mask)[0]
    if len(lag_0_indices) == 0:
        raise RuntimeError("No valid initial seed voxels found.")

    lag_map_values[lag_0_indices] = 0.0
    max_correlations[lag_0_indices] = initial_corrs[lag_0_indices]
    assigned_voxels[lag_0_indices] = True
    seed_0_ts = mapper.xp.mean(timeseries_2d_filtered_backend[:, lag_0_indices], axis=1)
    mapper.seed_time_series[0] = utils.to_numpy(seed_0_ts, mapper.use_gpu)

    # Initialize seeds for downstream/upstream tracking (MATLAB: SeedD = SeedU = Seed0)
    seed_downstream = seed_0_ts.copy()
    seed_upstream = seed_0_ts.copy()

    # Initialize shifted data arrays (MATLAB: Downward = Upward = Y)
    # We'll shift these at each iteration like MATLAB does
    data_downward = timeseries_2d_filtered_backend.copy()
    data_upward = timeseries_2d_filtered_backend.copy()

    logger.info(f"Initial lag=0 voxels: {len(lag_0_indices):,}")

    # Which voxels form the next seed. 'newly_found' averages only the voxels assigned at this step.
    # 'matlab' (default) replicates drLag4Drev7.m L261-268 literally: `SeedD = mean(Downward(:,I==2),2)`
    # is evaluated BEFORE `I(~isnan(Lag)) = 0`, so every voxel whose local peak sits at the centre with
    # R >= THR enters the seed, including voxels assigned at an earlier step. The local peak is therefore
    # computed over ALL voxels (numerically identical per voxel; only the seed membership differs).
    seed_update = getattr(mapper, 'seed_update', 'matlab')
    if seed_update not in ('newly_found', 'matlab'):
        raise ValueError(f"seed_update must be 'newly_found' or 'matlab', got {seed_update!r}")
    mapper.seed_voxel_counts = {0: int(len(lag_0_indices))}
    mapper.newly_found_counts = {0: int(len(lag_0_indices))}

    # A direction ENDS at the first step that yields no seed voxels. drLag4Drev7.m (L265, L289) and boldlag
    # lag4d.py (L148, L157) re-form the seed at every step as the mean of the centre-peaking voxels; with none it
    # is NaN and nothing is assigned in that direction afterwards. Carrying the previous seed over an empty step
    # instead would let a later step fill a lag bin that has no seed, i.e. no deperfusion regressor.
    down_open, up_open = True, True
    for p in tqdm(range(1, max_lag_trs + 1), desc="Recursive Propagation", disable=not mapper.verbose):
        if down_open:
            # --- DOWNSTREAM (negative lags: voxels that lag behind the seed) ---
            # MATLAB: Downward = [ Downward( 2:end,:); nanmean( Downward,1)];
            # This shifts data forward in time (removes first row, adds mean at end)
            data_downward = mean_padded_shift(data_downward, -1, mapper.xp, axis=0)

            unassigned_mask = ~assigned_voxels
            if not mapper.xp.any(unassigned_mask): break

            # Compare shifted data against seed, looking for peak at lag=0 (I==2 in MATLAB)
            lags_d, corrs_d = find_local_peak_time_domain(data_downward, seed_downstream, mapper.xp, window_trs=1)
            centre_mask_d = (lags_d == 0) & (corrs_d >= mapper.min_corr_threshold)
            newly_found_mask_d = centre_mask_d & unassigned_mask
            seed_mask_d = centre_mask_d if seed_update == 'matlab' else newly_found_mask_d

            # Debug: count how many at each local lag (over unassigned voxels, as before)
            n_lag_minus1 = int(mapper.xp.sum((lags_d == -1) & unassigned_mask))
            n_lag_0 = int(mapper.xp.sum((lags_d == 0) & unassigned_mask))
            n_lag_plus1 = int(mapper.xp.sum((lags_d == 1) & unassigned_mask))
            n_above_thr = int(mapper.xp.sum((corrs_d >= mapper.min_corr_threshold) & unassigned_mask))
            logger.info(f"  Downstream p={p}: local lags [-1,0,+1] = [{n_lag_minus1}, {n_lag_0}, {n_lag_plus1}], above_thr={n_above_thr}, found={int(mapper.xp.sum(newly_found_mask_d))}, seed_voxels={int(mapper.xp.sum(seed_mask_d))} ({seed_update})")
            mapper.newly_found_counts[-p] = int(mapper.xp.sum(newly_found_mask_d))

            if mapper.xp.any(newly_found_mask_d):
                newly_found_indices_d = mapper.xp.where(newly_found_mask_d)[0]
                lag_map_values[newly_found_indices_d] = -p * _tr_track(mapper)
                max_correlations[newly_found_indices_d] = corrs_d[newly_found_indices_d]
                assigned_voxels[newly_found_indices_d] = True
            if mapper.xp.any(seed_mask_d):
                # MATLAB: SeedD = mean( Downward( :,I==2),2); - update seed from SHIFTED data
                seed_indices_d = mapper.xp.where(seed_mask_d)[0]
                seed_downstream = mapper.xp.mean(data_downward[:, seed_indices_d], axis=1)
                mapper.seed_time_series[-p] = utils.to_numpy(seed_downstream, mapper.use_gpu)
                mapper.seed_voxel_counts[-p] = int(len(seed_indices_d))
            else:
                down_open = False
                logger.info(f"  Downstream tracking ends at p={p}: no seed voxels (the canonical seed becomes NaN).")

        if up_open:
            # --- UPSTREAM (positive lags: voxels that lead the seed) ---
            # MATLAB: Upward = [ nanmean( Upward,1); Upward( 1:end-1,:)];
            # This shifts data backward in time (adds mean at start, removes last row)
            data_upward = mean_padded_shift(data_upward, 1, mapper.xp, axis=0)

            unassigned_mask = ~assigned_voxels
            if not mapper.xp.any(unassigned_mask): break

            lags_u, corrs_u = find_local_peak_time_domain(data_upward, seed_upstream, mapper.xp, window_trs=1)
            centre_mask_u = (lags_u == 0) & (corrs_u >= mapper.min_corr_threshold)
            newly_found_mask_u = centre_mask_u & unassigned_mask
            seed_mask_u = centre_mask_u if seed_update == 'matlab' else newly_found_mask_u

            n_lag_minus1_u = int(mapper.xp.sum((lags_u == -1) & unassigned_mask))
            n_lag_0_u = int(mapper.xp.sum((lags_u == 0) & unassigned_mask))
            n_lag_plus1_u = int(mapper.xp.sum((lags_u == 1) & unassigned_mask))
            n_above_thr_u = int(mapper.xp.sum((corrs_u >= mapper.min_corr_threshold) & unassigned_mask))
            logger.info(f"  Upstream p={p}: local lags [-1,0,+1] = [{n_lag_minus1_u}, {n_lag_0_u}, {n_lag_plus1_u}], above_thr={n_above_thr_u}, found={int(mapper.xp.sum(newly_found_mask_u))}, seed_voxels={int(mapper.xp.sum(seed_mask_u))} ({seed_update})")
            mapper.newly_found_counts[p] = int(mapper.xp.sum(newly_found_mask_u))

            if mapper.xp.any(newly_found_mask_u):
                newly_found_indices_u = mapper.xp.where(newly_found_mask_u)[0]
                lag_map_values[newly_found_indices_u] = p * _tr_track(mapper)
                max_correlations[newly_found_indices_u] = corrs_u[newly_found_indices_u]
                assigned_voxels[newly_found_indices_u] = True
            if mapper.xp.any(seed_mask_u):
                # MATLAB: SeedU = mean( Upward( :,I==2),2); - update seed from SHIFTED data
                seed_indices_u = mapper.xp.where(seed_mask_u)[0]
                seed_upstream = mapper.xp.mean(data_upward[:, seed_indices_u], axis=1)
                mapper.seed_time_series[p] = utils.to_numpy(seed_upstream, mapper.use_gpu)
                mapper.seed_voxel_counts[p] = int(len(seed_indices_u))
            else:
                up_open = False
                logger.info(f"  Upstream tracking ends at p={p}: no seed voxels (the canonical seed becomes NaN).")

        if not (down_open or up_open):
            break

    return utils.to_numpy(lag_map_values, mapper.use_gpu), utils.to_numpy(max_correlations, mapper.use_gpu), utils.to_numpy(lag_0_mask, mapper.use_gpu)


def _find_local_peak_with_refinement(timeseries_backend, reference_ts_backend, xp, window_trs=2):
    """
    Same as find_local_peak_time_domain but also returns sub-TR offset via least-squares
    quadratic fit on all correlation values in the window.

    Uses a wider window (default ±2 TRs = 5 points) and fits y = a*x^2 + b*x + c to all
    points, giving a more robust sub-TR peak estimate than 3-point parabolic interpolation.

    Returns:
        lags_trs: Integer lag at peak (same as find_local_peak_time_domain)
        max_corrs: Peak correlation values
        offsets: Sub-TR offset in TRs from least-squares quadratic fit
    """
    n_timepoints, n_voxels = timeseries_backend.shape
    lim = window_trs

    if n_voxels == 0:
        return xp.array([]), xp.array([]), xp.array([])

    min_required_timepoints = 2 * lim + 1
    if n_timepoints < min_required_timepoints:
        # Return NaN correlations, not zeros (as find_local_peak_time_domain does): a neutral-looking 0
        # is indistinguishable from a real, genuinely zero correlation.
        nan = xp.full(n_voxels, xp.nan, dtype=xp.float32)
        return xp.zeros(n_voxels), nan, xp.zeros(n_voxels)

    ref_trimmed = reference_ts_backend[lim:-lim] if lim > 0 else reference_ts_backend
    shifts = list(range(lim, -lim - 1, -1))

    corrs_in_window = xp.full((len(shifts), n_voxels), -1.0, dtype=xp.float32)

    for i, sft in enumerate(shifts):
        start_idx = lim + sft
        end_idx = n_timepoints - lim + sft
        ts_window = timeseries_backend[start_idx:end_idx, :]
        XX = ref_trimmed[:, xp.newaxis]
        YY = ts_window
        numerator = xp.sum(XX * YY, axis=0)
        xx_sum_sq = xp.sum(XX * XX, axis=0)
        yy_sum_sq = xp.sum(YY * YY, axis=0)
        denominator = xp.sqrt(xx_sum_sq) * xp.sqrt(yy_sum_sq)
        denominator = xp.where(denominator == 0, 1e-10, denominator)
        corrs_in_window[i, :] = numerator / denominator

    peak_indices = xp.argmax(corrs_in_window, axis=0)
    lags_trs = xp.asarray([-shifts[int(idx)] for idx in peak_indices])
    col_idx = xp.arange(n_voxels)
    max_corrs = corrs_in_window[peak_indices, col_idx]

    # Least-squares quadratic fit: y = a*x^2 + b*x + c → peak at x = -b/(2a)
    # x values are lag positions relative to peak: [-lim-peak_lag, ..., +lim-peak_lag]
    # For each voxel, center the x values on its peak index
    n_shifts = len(shifts)
    # Lag values for each shift index: lag[i] = -(shifts[i]) = i - lim
    lag_values = xp.arange(n_shifts, dtype=xp.float32) - lim  # [-lim, ..., 0, ..., +lim]

    # For each voxel, compute x = lag_values - peak_lag (center on peak)
    peak_lags = lags_trs.astype(xp.float32)  # integer peak lag for each voxel

    # Vectorized least-squares: fit y = a*x^2 + b*x + c for each voxel
    # x[i, v] = lag_values[i] - peak_lags[v]
    x_all = lag_values[:, xp.newaxis] - peak_lags[xp.newaxis, :]  # (n_shifts, n_voxels)
    y_all = corrs_in_window  # (n_shifts, n_voxels)

    # Normal equations for quadratic: [sum(x^4), sum(x^3), sum(x^2)] [a]   [sum(x^2*y)]
    #                                 [sum(x^3), sum(x^2), sum(x)  ] [b] = [sum(x*y)  ]
    #                                 [sum(x^2), sum(x),   n       ] [c]   [sum(y)     ]
    # For symmetric x centered on peak, sum(x) ≈ 0 and sum(x^3) ≈ 0, simplifying.
    # But let's solve it properly for robustness.
    # What is used is the unweighted Savitzky-Golay closed form further down, plus the 3-point
    # parabolic fallback; no general least-squares moments are formed.

    # Offset needs only a and b: offset = -b/(2a).






    # 3-point parabolic on peak+/-1, used as the fallback near the window edge
    y_P = corrs_in_window[peak_indices, col_idx]
    y_L = corrs_in_window[xp.maximum(0, peak_indices - 1), col_idx]
    y_R = corrs_in_window[xp.minimum(n_shifts - 1, peak_indices + 1), col_idx]

    # Standard 3-point parabolic
    denom_3pt = 2.0 * (y_L - 2.0 * y_P + y_R)
    numer_3pt = y_L - y_R
    offsets_3pt = xp.zeros(n_voxels, dtype=xp.float32)
    valid_3pt = xp.abs(denom_3pt) > 1e-9
    offsets_3pt[valid_3pt] = numer_3pt[valid_3pt] / denom_3pt[valid_3pt]
    offsets_3pt = xp.clip(offsets_3pt, -0.5, 0.5)






    # For the ASSIGNMENT peak the 3-point parabolic is already good; the 5-point
    # closed form below helps when the peak is broad.
    # Use 5-point fit when available (peak not at edge), else fall back to 3-point.
    too_close_to_edge = (peak_indices <= 1) | (peak_indices >= n_shifts - 2)

    # For 5-point fit: use all 5 correlation values centered on peak
    # For simplicity and numerical stability, use Savitzky-Golay-like formula:
    # For equally-spaced points x = [-2,-1,0,1,2] with quadratic fit,
    # the peak offset is: offset = (2*(y_{-2} - y_{+2}) + (y_{-1} - y_{+1})) /
    #                              (2*(2*y_{-2} - y_{-1} - 6*y_0 - y_{+1} + 2*y_{+2}))
    # Wait, that's not right. Let me derive it properly.
    # For x = [-2,-1,0,1,2], y = a*x^2 + b*x + c:
    # sum(x^2*y) = a*sum(x^4) + b*sum(x^3) + c*sum(x^2)
    # sum(x*y) = a*sum(x^3) + b*sum(x^2) + c*sum(x)
    # For symmetric x: sum(x)=0, sum(x^3)=0, sum(x^2)=10, sum(x^4)=34
    # So: 34*a + 10*c = sum(x^2*y)
    #     10*b = sum(x*y)  →  b = sum(x*y) / 10
    # offset = -b/(2a). Need a: from first eq, a = (sum(x^2*y) - 10*c)/34
    # From sum(y) = a*sum(x^2) + c*5: 5c = sum(y) - 10a → c = (sum(y)-10a)/5
    # → 34a + 10*(sum(y)-10a)/5 = sum(x^2*y)
    # → 34a + 2*sum(y) - 20a = sum(x^2*y)
    # → 14a = sum(x^2*y) - 2*sum(y)
    # → a = (sum(x^2*y) - 2*sum(y)) / 14
    # → b = sum(x*y) / 10
    # → offset = -b/(2a) = -sum(x*y) / (10 * 2 * (sum(x^2*y) - 2*sum(y)) / 14)
    #          = -7 * sum(x*y) / (10 * (sum(x^2*y) - 2*sum(y)))

    # Compute for voxels with peak not at edge (have full 5 points centered on peak)
    offsets = offsets_3pt.copy()  # Default to 3-point

    has_5pts = ~too_close_to_edge
    if xp.any(has_5pts):
        has_5pts_idx = xp.where(has_5pts)[0]
        peak_idx_5 = peak_indices[has_5pts_idx]

        # Gather 5 correlation values centered on peak
        y_m2 = corrs_in_window[peak_idx_5 - 2, has_5pts_idx]
        y_m1 = corrs_in_window[peak_idx_5 - 1, has_5pts_idx]
        y_0 = corrs_in_window[peak_idx_5, has_5pts_idx]
        y_p1 = corrs_in_window[peak_idx_5 + 1, has_5pts_idx]
        y_p2 = corrs_in_window[peak_idx_5 + 2, has_5pts_idx]

        # x = [-2, -1, 0, 1, 2]: sum(x*y) = -2*y_m2 - y_m1 + y_p1 + 2*y_p2
        sum_xy = -2 * y_m2 - y_m1 + y_p1 + 2 * y_p2
        # sum(x^2*y) = 4*y_m2 + y_m1 + y_p1 + 4*y_p2
        sum_x2y = 4 * y_m2 + y_m1 + y_p1 + 4 * y_p2
        # sum(y) = y_m2 + y_m1 + y_0 + y_p1 + y_p2
        sum_y = y_m2 + y_m1 + y_0 + y_p1 + y_p2

        b_5pt = sum_xy / 10.0
        a_5pt = (sum_x2y - 2.0 * sum_y) / 14.0

        denom_5pt = 2.0 * a_5pt
        offsets_5pt = xp.zeros_like(b_5pt)
        # Require the fitted quadratic to be CONCAVE (a < 0). A convex fit (a > 0) makes -b/(2a) point
        # at a MINIMUM, so the sub-TR offset would take the wrong sign, and clipping to +/-0.5 TR would
        # hide it (example window [0.55, 0.10, 0.60, 0.10, 0.45]: a = +0.043, offset +0.233 TR, wrong
        # direction). A convex fit means the 5 sampled correlations do not describe a peak, so the
        # honest answer is no sub-TR shift: those voxels keep offset 0 and are counted in the log.
        valid_5pt = (a_5pt < 0) & (xp.abs(denom_5pt) > 1e-9)
        offsets_5pt[valid_5pt] = -b_5pt[valid_5pt] / denom_5pt[valid_5pt]
        offsets_5pt = xp.clip(offsets_5pt, -0.5, 0.5)
        n_convex = int(xp.sum((a_5pt >= 0) & (xp.abs(denom_5pt) > 1e-9)))
        if n_convex:
            logger.info(f"5-point sub-TR fit: {n_convex:,} of {int(a_5pt.shape[0]):,} voxels had a "
                        "convex (non-peak) correlation curve; their sub-TR offset is set to 0 "
                        "rather than to the minimum of the parabola.")

        offsets[has_5pts_idx] = offsets_5pt

    return lags_trs, max_corrs, offsets


def compute_lag_maps_recursive_subtr(mapper, timeseries_2d_filtered_backend, global_signal_backend):
    """
    Recursive lag mapping with sub-TR precision via parabolic interpolation.

    Runs the same recursive propagation SCHEME as compute_lag_maps_recursive (MATLAB FIXED=0) - the
    same seed updates and the same spatial propagation order - then refines each voxel's lag to
    sub-TR precision by parabolic interpolation on the local correlation curve.

    The voxel assignments are NOT identical to compute_lag_maps_recursive: that function calls
    find_local_peak_time_domain with window_trs=1 in its propagation loop, this one with window_trs=2.
    The wider window means a different trim length (n-4 vs n-2 samples), different correlation values,
    and a strictly stricter acceptance rule - a voxel whose correlation peaks at +/-2 is rejected under
    lim=2 but accepted under lim=1 - so the assigned voxel sets differ.

    The sub-TR offset is computed from: offset = (y_L - y_R) / (2*(y_L - 2*y_P + y_R))
    where y_L, y_P, y_R are correlations at the peak and its neighbors.

    SEED-PHASE TRACKING (`mapper.subtr_phase_tracking`, default True). The voxels assigned at |p| = 1
    have a mean sub-TR offset that points towards zero, because the true lag distribution is densest
    near 0 so each integer bin is denser on its zero-facing side. The next seed is the mean of exactly
    those voxels and therefore inherits that mean phase; assigning `(-p + offset)` as if the seed sat
    at precisely -(p-1) TR would push every voxel found at |p| >= 2 AWAY from zero, with an error that
    accumulates step by step. With tracking, the seed phase phi (in TR, relative to the initial ROI
    seed) is carried: lag = (-p + phi_{p-1} + offset) * TR and phi_p = phi_{p-1} + mean(offset of the
    voxels that formed the new seed). --no-subtr-phase-tracking assigns `(-p + offset)`. MATLAB has no
    sub-TR step, so this is a property of the Python extension only.

    SEED-UPDATE POPULATION: see compute_lag_maps_recursive (`mapper.seed_update`).

    Args:
        mapper: BOLDLagMapper instance
        timeseries_2d_filtered_backend: Preprocessed timeseries (n_timepoints, n_voxels)
        global_signal_backend: Global reference signal (n_timepoints,)

    Returns:
        Tuple of (lag_map_values, max_correlations, lag0_mask) as numpy arrays
    """
    logger.info("--- RUNNING RECURSIVE LAG MAPPING WITH SUB-TR REFINEMENT ---")
    max_lag_trs = int(round(mapper.max_lag_seconds / _tr_track(mapper)))
    lag_map_values = mapper.xp.full(mapper.num_voxels, mapper.xp.nan)
    max_correlations = mapper.xp.full(mapper.num_voxels, mapper.xp.nan)
    assigned_voxels = mapper.xp.zeros(mapper.num_voxels, dtype=bool)
    seed_update = getattr(mapper, 'seed_update', 'matlab')
    if seed_update not in ('newly_found', 'matlab'):
        raise ValueError(f"seed_update must be 'newly_found' or 'matlab', got {seed_update!r}")
    phase_tracking = bool(getattr(mapper, 'subtr_phase_tracking', True))
    logger.info(f"Sub-TR seed-phase tracking: {'ON' if phase_tracking else 'OFF'}; seed update: {seed_update}")

    # --- Initialize: find lag-0 voxels WITH sub-TR refinement ---
    # Use refined version so lag-0 voxels also get sub-TR offsets
    initial_lags_trs, initial_corrs, initial_offsets = _find_local_peak_with_refinement(
        timeseries_2d_filtered_backend, global_signal_backend, mapper.xp, window_trs=2
    )
    lag_0_mask = (initial_lags_trs == 0) & (initial_corrs >= mapper.min_corr_threshold)
    lag_0_indices = mapper.xp.where(lag_0_mask)[0]
    if len(lag_0_indices) == 0:
        raise RuntimeError("No valid initial seed voxels found.")

    # Apply sub-TR offset to lag-0 voxels (offset * tr gives lag in seconds)
    lag_map_values[lag_0_indices] = initial_offsets[lag_0_indices] * _tr_track(mapper)
    max_correlations[lag_0_indices] = initial_corrs[lag_0_indices]
    assigned_voxels[lag_0_indices] = True
    seed_0_ts = mapper.xp.mean(timeseries_2d_filtered_backend[:, lag_0_indices], axis=1)
    mapper.seed_time_series[0] = utils.to_numpy(seed_0_ts, mapper.use_gpu)

    seed_downstream = seed_0_ts.copy()
    seed_upstream = seed_0_ts.copy()
    # phase of the current seed in TR, relative to the initial ROI seed (0 when tracking is off)
    phase_0 = float(mapper.xp.mean(initial_offsets[lag_0_indices])) if phase_tracking else 0.0
    phase_d, phase_u = phase_0, phase_0
    mapper.seed_phase_trs = {0: phase_0}
    mapper.seed_voxel_counts = {0: int(len(lag_0_indices))}
    mapper.newly_found_counts = {0: int(len(lag_0_indices))}

    data_downward = timeseries_2d_filtered_backend.copy()
    data_upward = timeseries_2d_filtered_backend.copy()

    logger.info(f"Initial lag=0 voxels: {len(lag_0_indices):,} (seed phase {phase_0:+.4f} TR)")

    # A direction ends at the first step without seed voxels, as in compute_lag_maps_recursive
    # (see the comment there; the original's seed becomes NaN).
    down_open, up_open = True, True
    for p in tqdm(range(1, max_lag_trs + 1), desc="Recursive + Sub-TR", disable=not mapper.verbose):
        if down_open:
            # --- DOWNSTREAM (negative lags) ---
            data_downward = mean_padded_shift(data_downward, -1, mapper.xp, axis=0)

            unassigned_mask = ~assigned_voxels
            if not mapper.xp.any(unassigned_mask):
                break

            # +/-2 window for the 5-point quadratic fit, over ALL voxels (per-voxel result is
            # independent of the others; the seed membership rule below is what differs by mode)
            lags_d, corrs_d, offsets_d = _find_local_peak_with_refinement(
                data_downward, seed_downstream, mapper.xp, window_trs=2
            )
            centre_mask_d = (lags_d == 0) & (corrs_d >= mapper.min_corr_threshold)
            newly_found_mask_d = centre_mask_d & unassigned_mask
            seed_mask_d = centre_mask_d if seed_update == 'matlab' else newly_found_mask_d

            n_found_d = int(mapper.xp.sum(newly_found_mask_d))
            mapper.newly_found_counts[-p] = n_found_d
            logger.info(f"  Downstream p={p}: found={n_found_d}, seed_voxels={int(mapper.xp.sum(seed_mask_d))} "
                        f"({seed_update}), seed phase {phase_d:+.4f} TR")

            if n_found_d:
                newly_found_indices_d = mapper.xp.where(newly_found_mask_d)[0]
                refined_offsets_d = offsets_d[newly_found_indices_d]
                lag_map_values[newly_found_indices_d] = (-p + phase_d + refined_offsets_d) * _tr_track(mapper)
                max_correlations[newly_found_indices_d] = corrs_d[newly_found_indices_d]
                assigned_voxels[newly_found_indices_d] = True
            if mapper.xp.any(seed_mask_d):
                seed_indices_d = mapper.xp.where(seed_mask_d)[0]
                seed_downstream = mapper.xp.mean(data_downward[:, seed_indices_d], axis=1)
                if phase_tracking:
                    phase_d += float(mapper.xp.mean(offsets_d[seed_indices_d]))
                mapper.seed_time_series[-p] = utils.to_numpy(seed_downstream, mapper.use_gpu)
                mapper.seed_voxel_counts[-p] = int(len(seed_indices_d))
                mapper.seed_phase_trs[-p] = phase_d
            else:
                down_open = False
                logger.info(f"  Downstream tracking ends at p={p}: no seed voxels (the canonical seed becomes NaN).")

        if up_open:
            # --- UPSTREAM (positive lags) ---
            data_upward = mean_padded_shift(data_upward, 1, mapper.xp, axis=0)

            unassigned_mask = ~assigned_voxels
            if not mapper.xp.any(unassigned_mask):
                break

            lags_u, corrs_u, offsets_u = _find_local_peak_with_refinement(
                data_upward, seed_upstream, mapper.xp, window_trs=2
            )
            centre_mask_u = (lags_u == 0) & (corrs_u >= mapper.min_corr_threshold)
            newly_found_mask_u = centre_mask_u & unassigned_mask
            seed_mask_u = centre_mask_u if seed_update == 'matlab' else newly_found_mask_u

            n_found_u = int(mapper.xp.sum(newly_found_mask_u))
            mapper.newly_found_counts[p] = n_found_u
            logger.info(f"  Upstream   p={p}: found={n_found_u}, seed_voxels={int(mapper.xp.sum(seed_mask_u))} "
                        f"({seed_update}), seed phase {phase_u:+.4f} TR")

            if n_found_u:
                newly_found_indices_u = mapper.xp.where(newly_found_mask_u)[0]
                refined_offsets_u = offsets_u[newly_found_indices_u]
                lag_map_values[newly_found_indices_u] = (p + phase_u + refined_offsets_u) * _tr_track(mapper)
                max_correlations[newly_found_indices_u] = corrs_u[newly_found_indices_u]
                assigned_voxels[newly_found_indices_u] = True
            if mapper.xp.any(seed_mask_u):
                seed_indices_u = mapper.xp.where(seed_mask_u)[0]
                seed_upstream = mapper.xp.mean(data_upward[:, seed_indices_u], axis=1)
                if phase_tracking:
                    phase_u += float(mapper.xp.mean(offsets_u[seed_indices_u]))
                mapper.seed_time_series[p] = utils.to_numpy(seed_upstream, mapper.use_gpu)
                mapper.seed_voxel_counts[p] = int(len(seed_indices_u))
                mapper.seed_phase_trs[p] = phase_u
            else:
                up_open = False
                logger.info(f"  Upstream   tracking ends at p={p}: no seed voxels (the canonical seed becomes NaN).")

        if not (down_open or up_open):
            break

    n_assigned = int(mapper.xp.sum(assigned_voxels))
    logger.info(f"Recursive sub-TR complete: {n_assigned:,}/{mapper.num_voxels:,} "
                f"voxels assigned ({n_assigned / mapper.num_voxels * 100:.1f}%)")

    # D/U ratio logging (monitoring only — no behavioral change)
    lag_values_np = utils.to_numpy(lag_map_values, mapper.use_gpu)
    # This QC tally counts the SIGN of the final lag, so a step-0 voxel whose sub-TR offset came out
    # slightly negative is counted as downstream. The integer step is reported as well so the two are
    # distinguishable.
    n_downstream = int(np.sum(lag_values_np < 0))
    n_upstream = int(np.sum(lag_values_np > 0))
    du_ratio = n_downstream / max(n_upstream, 1)
    n_zero_step = int(np.sum(np.abs(lag_values_np) < (_tr_track(mapper) / 2.0)))
    logger.info(f"D/U ratio: {du_ratio:.3f} (downstream={n_downstream:,}, upstream={n_upstream:,}; "
                f"{n_zero_step:,} voxels are within +/-TR/2 of zero, i.e. step 0 with a sub-TR offset)")

    return (utils.to_numpy(lag_map_values, mapper.use_gpu),
            utils.to_numpy(max_correlations, mapper.use_gpu),
            utils.to_numpy(lag_0_mask, mapper.use_gpu))

def compute_lag_maps_fixed_subtr(mapper, timeseries_2d_filtered_backend, global_signal_backend):
    """MATLAB FIXED=1 propagation (one seed for every step) + the 5-point sub-TR refinement of
    recursive_subtr.

    The seed is the mean of the lag-0 voxels and is never updated, so no seed-phase error can
    accumulate step by step: the only phase term is that of the lag-0 seed itself (phase_0, the mean
    sub-TR offset of the voxels that formed it), applied to every voxel. Lags in seconds are
    (+/-p + phase_0 + offset) * tr_track. Deperfusion uses the single seed for every lag bin.
    """
    xp = mapper.xp
    tr_track = _tr_track(mapper)
    logger.info("--- RUNNING FIXED-SEED LAG MAPPING WITH SUB-TR REFINEMENT (fixed_subtr) ---")
    max_lag_trs = int(round(mapper.max_lag_seconds / tr_track))
    lag_map_values = xp.full(mapper.num_voxels, xp.nan)
    max_correlations = xp.full(mapper.num_voxels, xp.nan)
    assigned = xp.zeros(mapper.num_voxels, dtype=bool)

    lags0, corrs0, offs0 = _find_local_peak_with_refinement(
        timeseries_2d_filtered_backend, global_signal_backend, xp, window_trs=2)
    lag_0_mask = (lags0 == 0) & (corrs0 >= mapper.min_corr_threshold)
    idx0 = xp.where(lag_0_mask)[0]
    if len(idx0) == 0:
        raise RuntimeError("fixed_subtr: no valid lag=0 voxels found.")
    phase_0 = float(xp.mean(offs0[idx0]))
    lag_map_values[idx0] = offs0[idx0] * tr_track          # lag-0 voxels keep their own sub-TR offset
    max_correlations[idx0] = corrs0[idx0]
    assigned[idx0] = True
    ref = xp.mean(timeseries_2d_filtered_backend[:, idx0], axis=1)
    mapper.seed_time_series[0] = utils.to_numpy(ref, mapper.use_gpu)
    mapper.seed_phase_trs = {0: phase_0}
    mapper.seed_voxel_counts = {0: int(len(idx0))}
    mapper.newly_found_counts = {0: int(len(idx0))}
    logger.info(f"Initial lag=0 voxels: {len(idx0):,} (seed phase {phase_0:+.4f} steps of {tr_track:g} s)")

    for p in tqdm(range(1, max_lag_trs + 1), desc="Fixed + Sub-TR", disable=not mapper.verbose):
        for sign, shift in ((-1, -p), (+1, p)):          # downstream (negative lags), then upstream
            unassigned = ~assigned
            if not xp.any(unassigned):
                break
            ts_shifted = mean_padded_shift(timeseries_2d_filtered_backend, shift, xp)
            lags, corrs, offs = _find_local_peak_with_refinement(ts_shifted[:, unassigned], ref, xp, window_trs=2)
            found = (lags == 0) & (corrs >= mapper.min_corr_threshold)
            n_found = int(xp.sum(found))
            mapper.newly_found_counts[sign * p] = n_found
            logger.info(f"  {'Downstream' if sign < 0 else 'Upstream  '} p={p}: found={n_found}")
            if n_found:
                idx = xp.where(unassigned)[0][found]
                lag_map_values[idx] = (sign * p + phase_0 + offs[found]) * tr_track
                max_correlations[idx] = corrs[found]
                assigned[idx] = True

    n_assigned = int(xp.sum(assigned))
    logger.info(f"fixed_subtr complete: {n_assigned:,}/{mapper.num_voxels:,} voxels assigned "
                f"({n_assigned / mapper.num_voxels * 100:.1f}%)")
    return (utils.to_numpy(lag_map_values, mapper.use_gpu),
            utils.to_numpy(max_correlations, mapper.use_gpu),
            utils.to_numpy(lag_0_mask, mapper.use_gpu))
