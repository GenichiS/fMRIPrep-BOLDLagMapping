"""What deperfusion does with voxels it cannot regress.

Original (boldlag deperf.py L85-99, drDeperf_longTR.m L100-139): the output starts as the temporal mean and the
regression residual is added inside each lag region L = -MaxLag..MaxLag; a voxel in no region (no lag, or a lag that
rounds outside the regressor range) keeps its temporal mean; a failing fsl_regfilt / pinv stops the run.
Here: voxels without a lag keep their temporal mean (finite, so a bSpline warp to MNI is not poisoned - a single NaN
can turn a whole ANTs bSpline output NaN), with a WARNING and a count; a bin INSIDE the range without a regressor
cannot happen once tracking ends at the first seedless step, so it raises; a failed regression, no regressors and a
map without lags raise. A finite lag that rounds OUTSIDE the regressor range (only sub-TR lags can, by their fraction
past the last tracked step) is regressed with the nearest, edge regressor instead of keeping its temporal mean."""
import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bold_lag_mapper import lag_postprocess  # noqa: E402

N_T = 120


def _mapper(lag_map_values):
    return SimpleNamespace(xp=np, use_gpu=False, gpu_batch_size=10000, verbose=False, final_hp_cutoff_hz=0.008,
                           tr=2.5, tr_track=1.0, lag_map_values=np.asarray(lag_map_values, dtype=float))


def _data(n_vox, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N_T) * 2.5
    slfo = np.sin(2 * np.pi * 0.03 * t)
    return 1000.0 + 10.0 * slfo[:, None] + rng.normal(0, 1.0, (N_T, n_vox)), slfo


def test_lag_less_voxels_keep_their_mean_and_outside_range_voxels_use_the_edge_regressor(caplog):
    data, slfo = _data(6)
    late = np.roll(slfo, 3)                             # a different regressor for bin 1, so the edge used is visible
    # bins: 0, 0, 1, NaN (no lag), 5 (above the regressor range [0, 1]), -3 (below it)
    m = _mapper([0.0, 0.2, 1.0, np.nan, 5.0, -3.0])
    data[:, 4] = data[:, 2]                             # voxel 4 carries voxel 2's data (bin 1)
    data[:, 5] = data[:, 0]                             # voxel 5 carries voxel 0's data (bin 0)
    with caplog.at_level(logging.WARNING, logger='bold_lag_mapper.lag_postprocess'):
        out = lag_postprocess.apply_deperfusion_regression_matlab_compat(m, data, {0: slfo, 1: late})
    assert np.isfinite(out).all(), "never NaN: the output is warped with bSpline"
    assert np.allclose(out[:, 3], data[:, 3].mean()), "a voxel without a lag keeps its temporal mean"
    assert np.allclose(out[:, 4], out[:, 2]), "a lag above the range is regressed with the upper edge regressor (bin 1)"
    assert np.allclose(out[:, 5], out[:, 0]), "a lag below the range is regressed with the lower edge regressor (bin 0)"
    assert out[:, 4].std() > 0 and out[:, 5].std() > 0, "edge voxels are regressed, not flattened"
    msg = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "1 voxels without a lag" in msg and "2 voxels" in msg and "edge" in msg and "[0, 1]" in msg


def test_bin_inside_the_range_without_regressor_raises():
    data, slfo = _data(3, seed=1)
    m = _mapper([0.0, 1.0, 2.0])                       # bin 1 lies between the regressors 0 and 2
    with pytest.raises(RuntimeError, match="no regressor"):
        lag_postprocess.apply_deperfusion_regression_matlab_compat(m, data, {0: slfo, 2: slfo})


def test_failed_regression_raises():
    data, slfo = _data(4, seed=2)
    m = _mapper([0.0, 0.0, 1.0, 1.0])
    with patch.object(lag_postprocess.np.linalg, 'lstsq', side_effect=np.linalg.LinAlgError("forced")):
        with pytest.raises(np.linalg.LinAlgError):
            lag_postprocess.apply_deperfusion_regression_matlab_compat(m, data, {0: slfo, 1: slfo})


def test_no_regressors_raises():
    data, _ = _data(3, seed=3)
    with pytest.raises(ValueError, match="regressor"):
        lag_postprocess.apply_deperfusion_regression_matlab_compat(_mapper([0.0, 1.0, 2.0]), data, {})


def test_map_without_lags_raises():
    data, slfo = _data(3, seed=4)
    with pytest.raises(ValueError, match="No valid lags"):
        lag_postprocess.apply_deperfusion_regression_matlab_compat(_mapper([np.nan] * 3), data, {0: slfo})


def test_bin_summary_counts_every_voxel():
    groups, s = lag_postprocess.group_voxels_by_lag_bin([0.0, 0.4, 0.6, np.nan, 9.0, -2.0], [0, 1], 1.0)
    assert sorted(groups) == [0, 1]
    assert list(groups[0]) == [5, 0, 1], "bin 0: the -2 s voxel (below the range) joins it, ordered by lag"
    assert list(groups[1]) == [2, 4], "bin 1: the 9 s voxel (above the range) joins it, ordered by lag"
    assert s == dict(n_voxels=6, n_regressed=5, n_no_lag=1, n_assigned_to_range_edge=2, regressor_range_steps=[0, 1])


def test_rounding_matches_the_previous_grouping():
    # the old code grouped by Python round(float(lag) / tr_track) over float32 map values (round half to even)
    vals = np.array([0.5, 1.5, 2.5, -0.5, -1.5, 3.4999, 3.5001], dtype=np.float32)
    groups, _ = lag_postprocess.group_voxels_by_lag_bin(vals, range(-5, 6), 1.0)
    got = {int(i): b for b, idx in groups.items() for i in idx}
    assert got == {i: int(round(float(v) / 1.0)) for i, v in enumerate(vals)}


def test_mapper_writes_the_deperfusion_sidecar(tmp_path):
    from bold_lag_mapper import core  # noqa: F401
    from test_resampled_step_consumers import _lagged_run
    from test_save_cleaned_bold import _mapper_from_cli
    bold, mask, conf = _lagged_run(tmp_path, seed=5)
    m = _mapper_from_cli(tmp_path, bold, mask, conf, ['--tracking-step-seconds', '1.0', '--save-deperfusioned-bold', 'raw'])
    m.process_runs([bold], [conf], mask, str(tmp_path / 'out'))
    side = list((tmp_path / 'out').glob('*_desc-deperfusion.json'))
    assert len(side) == 1, sorted(p.name for p in (tmp_path / 'out').iterdir())
    s = json.load(open(side[0]))
    assert s['n_voxels'] == s['n_regressed'] + s['n_no_lag'] == m.num_voxels
    # independent of the grouping code: counts and range read from the map and the seeds themselves
    lag = np.asarray(m.lag_map_values, float)
    assert s['n_no_lag'] == int(np.isnan(lag).sum())
    assert s['regressor_range_steps'] == [min(m.seed_time_series), max(m.seed_time_series)]
    assert s['n_assigned_to_range_edge'] == 0 and s['n_regressed'] == int(np.isfinite(lag).sum()) > 0, \
        "integer 1 s-grid maps are filled within the tracked range"
    m2 = _mapper_from_cli(tmp_path, bold, mask, conf, ['--tracking-step-seconds', '1.0'])
    m2.process_runs([bold], [conf], mask, str(tmp_path / 'out2'))
    assert not list((tmp_path / 'out2').glob('*_desc-deperfusion.json')), "no deperfusion, no deperfusion sidecar"


def test_lag_steps_divides_in_float64():
    # 0.4f / 0.8 is exactly 0.5 in float32 (bin 0) but 0.5000000075 in float64 (bin 1)
    v = np.array([0.4, -0.4, np.nan, 2.4], dtype=np.float32)
    steps, finite = lag_postprocess.lag_steps(v, 0.8)
    assert list(finite) == [True, True, False, True]
    assert [int(x) for x in steps[finite]] == [1, -1, 3]
    assert int(np.round(v / 0.8)[0]) == 0, "a float32 division would give bin 0"


def test_fixed_seed_keys_cover_every_bin_the_grouping_uses():
    rng = np.random.default_rng(6)
    v = (0.4 * rng.integers(-15, 16, 5000)).astype(np.float32)      # every lag / 0.8 is a whole or exact half step
    keys = lag_postprocess.fixed_seed_keys(v, 0.8)
    _, s = lag_postprocess.group_voxels_by_lag_bin(v, keys, 0.8)
    assert s['n_regressed'] == s['n_voxels'] == v.size


def test_fixed_path_builds_its_keys_with_the_shared_binning(tmp_path):
    from bold_lag_mapper import core
    from test_resampled_step_consumers import _lagged_run
    from test_save_cleaned_bold import _mapper_from_cli
    bold, mask, conf = _lagged_run(tmp_path, seed=7)
    m = _mapper_from_cli(tmp_path, bold, mask, conf, ['--tracking-method', 'fixed', '--save-deperfusioned-bold', 'raw'])
    calls = []
    orig = lag_postprocess.fixed_seed_keys

    def spy(values, tr_track):
        calls.append(np.asarray(values).copy())
        return orig(values, tr_track)

    with patch.object(core.lag_postprocess, 'fixed_seed_keys', side_effect=spy):
        m.process_runs([bold], [conf], mask, str(tmp_path / 'out'))
    assert m.tracking_method == 'fixed' and len(calls) == 1, "core.py must build the fixed-seed keys with fixed_seed_keys"
    assert set(m.seed_time_series) >= set(int(k) for k in orig(calls[0], m.tr_track))
