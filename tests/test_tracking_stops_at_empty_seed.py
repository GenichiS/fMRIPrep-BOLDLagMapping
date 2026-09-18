"""Recursive tracking ends in a direction at the first step that yields no seed voxels.

Original (boldlag lag4d.py L147-151 / L156-160, drLag4Drev7.m L265-269 / L289-296): the seed is re-formed at every
step as the mean of the centre-peaking voxels; with none it is NaN, and nothing is assigned in that direction from
then on. Carrying the previous seed over the empty step would let a later step assign voxels while the empty step
had no seed - a lag bin without a deperfusion regressor. Data: a lag-0 group and a group delayed by 2 samples,
nothing at 1 sample, so step 1 has no seed voxels on either side.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bold_lag_mapper import lag_estimators_matlab as lem  # noqa: E402

N_T, N_EACH = 400, 20


def _data(seed=0):
    rng = np.random.default_rng(seed)
    s = np.convolve(rng.normal(size=N_T + 2), [0.25, 0.5, 0.25], mode='valid')      # short autocorrelation
    lag0 = s[:, None] + rng.normal(0, 0.05, (N_T, N_EACH))
    late2 = np.r_[np.full(2, s.mean()), s[:-2]][:, None] + rng.normal(0, 0.05, (N_T, N_EACH))
    return np.hstack([lag0, late2]).astype(np.float64), s


def _mapper(n_vox):
    return SimpleNamespace(xp=np, use_gpu=False, num_voxels=n_vox, max_lag_seconds=3.0, tr=1.0,
                           min_corr_threshold=0.2, verbose=False, seed_time_series={}, seed_update='matlab',
                           subtr_phase_tracking=True)


@pytest.mark.parametrize("func", [lem.compute_lag_maps_recursive, lem.compute_lag_maps_recursive_subtr])
def test_direction_ends_at_first_step_without_seed(func):
    Y, s = _data()
    m = _mapper(Y.shape[1])
    lag, corr, lag0_mask = func(m, Y, s)
    assert np.all(lag0_mask[:N_EACH]) and not np.any(lag0_mask[N_EACH:]), "test data: the lag-0 group seeds step 0"
    assert set(m.seed_time_series) == {0}, f"no seed can exist beyond the empty step 1, got {sorted(m.seed_time_series)}"
    assert np.all(np.isnan(lag[N_EACH:])), "voxels beyond the empty step stay unassigned (canonical: NaN seed)"


@pytest.mark.parametrize("func", [lem.compute_lag_maps_recursive, lem.compute_lag_maps_recursive_subtr])
def test_seed_keys_are_contiguous_on_ordinary_data(func):
    rng = np.random.default_rng(1)
    s = np.convolve(rng.normal(size=N_T + 2), [0.25, 0.5, 0.25], mode='valid')
    cols = [np.r_[np.full(d, s.mean()), s[:N_T - d]] if d >= 0 else np.r_[s[-d:], np.full(-d, s.mean())]
            for d in (-2, -1, 0, 1, 2) for _ in range(N_EACH)]
    Y = np.stack(cols, 1) + rng.normal(0, 0.05, (N_T, 5 * N_EACH))
    m = _mapper(Y.shape[1])
    func(m, Y, s)
    keys = sorted(m.seed_time_series)
    assert keys == list(range(keys[0], keys[-1] + 1)), f"seed keys must be contiguous, got {keys}"
    assert keys[0] <= -2 and keys[-1] >= 2, "ordinary data still tracks both ways"
