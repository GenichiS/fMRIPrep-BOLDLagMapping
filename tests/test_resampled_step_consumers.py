"""Consumers of the seeds when the series was resampled to a tracking step (--tracking-step-seconds) must work on
that step, not on the acquisition TR.

(1) The lag-0 seed plot must use the tracking grid as its time axis (otherwise 600 samples x 2.5 s = 1500 s for a
    600 s run).
(2) Deperfusion must shift each seed on the tracking grid first and resample the shifted regressors to the TR
    afterwards, as Dr Aso's drDeperf_longTR.m (L66-78) and boldlag deperf.py (L72-74) do; resampling first and
    shifting by the nearest whole TR would turn a 1 s or 2 s lag into 0 or 2.5 s.
The cross-check against boldlag itself runs when boldlag is installed.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bold_lag_mapper import core, lag_estimators_matlab, lag_postprocess, viz  # noqa: E402
from test_save_cleaned_bold import _mapper_from_cli, TR  # noqa: E402


def _lagged_run(tmp_path, n=120, shape=(8, 8, 6), seed=0):
    """Five voxel slabs carrying the same sLFO at -2, -1, 0, +1 and +2 s, so every 1 s tracking step has seed voxels
    (tracking ends in a direction at the first step without seed voxels, so the lags must be contiguous)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * TR
    data = np.full((*shape, n), 800.0, dtype=np.float32)
    mask = np.zeros(shape, np.uint8)
    for k, d in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0)):
        x0 = 1 + k
        slfo = np.sin(2 * np.pi * 0.03 * (t - d)) + 0.5 * np.sin(2 * np.pi * 0.017 * (t - d) + 1.0)
        data[x0:x0 + 1, 1:7, 1:5, :] = (1000.0 + 10.0 * slfo[None, None, None, :]
                                        + rng.normal(0, 1.0, (1, 6, 4, n)).astype(np.float32))
        mask[x0:x0 + 1, 1:7, 1:5] = 1
    img = nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))
    img.header.set_zooms((2.0, 2.0, 2.0, TR)); img.header.set_xyzt_units('mm', 'sec')
    bold = tmp_path / 'sub-01_task-rest_run-01_bold.nii.gz'
    nib.save(img, str(bold))
    mask_p = tmp_path / 'sub-01_mask.nii.gz'
    nib.save(nib.Nifti1Image(mask, img.affine), str(mask_p))
    cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'framewise_displacement']
    conf = tmp_path / 'sub-01_task-rest_run-01_desc-confounds_timeseries.tsv'
    with open(conf, 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for _ in range(n):
            f.write('\t'.join(f'{v:.6f}' for v in rng.normal(0, 1e-3, 6).tolist() + [0.05]) + '\n')
    return str(bold), str(mask_p), str(conf)


def _canonical_order(seed, lag_steps, tr, step, n_frames):
    """Shift on the tracking grid, then resample to the TR grid (drDeperf_longTR / boldlag deperf)."""
    r = lag_estimators_matlab.mean_padded_shift(np.asarray(seed, float), -int(lag_steps), np, axis=0)
    r = resample_poly(r, int(round(step * 100)), int(round(tr * 100)), window=('kaiser', 5.0))
    return r[:n_frames]


def test_seed_signal_plot_gets_the_tracking_step(tmp_path):
    bold, mask, conf = _lagged_run(tmp_path)
    m = _mapper_from_cli(tmp_path, bold, mask, conf, ['--tracking-step-seconds', '1.0', '--save-carpet-map'])
    seen = {}
    orig = viz.save_seed_signal_plot

    def spy(seed_signal, tr, output_prefix):
        seen['n'], seen['step'] = len(seed_signal), tr
        return orig(seed_signal, tr, output_prefix)

    with patch.object(core.viz, 'save_seed_signal_plot', side_effect=spy):
        m.process_runs([bold], [conf], mask, str(tmp_path / 'out'))
    assert 'step' in seen, "the lag-0 seed plot was not drawn"
    assert abs(seen['step'] - 1.0) < 1e-9, f"seed plot time base {seen['step']} s, expected the 1 s tracking step"
    assert (seen['n'] - 1) * seen['step'] <= 120 * TR, "the plotted time axis is longer than the run"


def test_deperfusion_regressors_are_shifted_on_the_tracking_grid(tmp_path):
    bold, mask, conf = _lagged_run(tmp_path, seed=1)
    m = _mapper_from_cli(tmp_path, bold, mask, conf, ['--tracking-step-seconds', '1.0', '--save-deperfusioned-bold', 'raw'])
    seen = {}
    orig = lag_postprocess.apply_deperfusion_regression_matlab_compat

    def spy(mapper, data, run_regressors):
        seen['reg'] = {k: np.asarray(v, float).copy() for k, v in run_regressors.items()}
        seen['n'] = data.shape[0]
        return orig(mapper, data, run_regressors)

    with patch.object(core.lag_postprocess, 'apply_deperfusion_regression_matlab_compat', side_effect=spy):
        m.process_runs([bold], [conf], mask, str(tmp_path / 'out'))
    assert abs(m.tr_track - 1.0) < 1e-9 and abs(m.tr - TR) < 1e-9
    nonzero = [k for k in seen['reg'] if k != 0 and k in m.seed_time_series]
    assert nonzero, f"test data must produce non-zero tracking steps, got {sorted(seen['reg'])}"
    for k, reg in seen['reg'].items():
        exp = _canonical_order(m.seed_time_series[k], k, m.tr, m.tr_track, seen['n'])
        assert reg.shape == (seen['n'],)
        assert np.allclose(reg, exp, rtol=1e-6, atol=1e-9), f"lag step {k}: regressor not shifted on the tracking grid"


def test_regressors_match_boldlag_deperf_order():
    pytest.importorskip("boldlag", reason="install boldlag to run the cross-check against Dr Aso's port")
    from boldlag.deperf import shifted_seeds                     # the original Motodata construction
    max_lag, n_step, tr, step = 3, 300, 2.5, 1.0
    rng = np.random.default_rng(3)
    seeds = {}
    for k in range(-max_lag, max_lag + 1):                        # zero-mean, so mean padding == boldlag's zero padding
        x = np.convolve(rng.normal(size=n_step + 20), np.ones(20) / 20, mode='valid')[:n_step]
        seeds[k] = x - x.mean()
    S = np.stack([seeds[max_lag - j] for j in range(2 * max_lag + 1)], 1)   # boldlag column j = lag (max_lag - j)
    moto = resample_poly(shifted_seeds(S, max_lag), int(step * 100), int(tr * 100), axis=0, window=('kaiser', 5.0))
    ours = lag_postprocess.build_deperfusion_regressors(seeds, tr, step, moto.shape[0])
    for k in range(-max_lag, max_lag + 1):
        assert np.allclose(ours[k], moto[:, k + max_lag], atol=1e-9), f"lag step {k} differs from boldlag deperf"


def test_regressors_without_resampling_are_the_integer_shift():
    rng = np.random.default_rng(4)
    seeds = {k: rng.normal(size=200) for k in (-2, -1, 0, 1, 2)}
    ours = lag_postprocess.build_deperfusion_regressors(seeds, 0.8, 0.8, 200)
    for k, s in seeds.items():
        assert np.array_equal(ours[k], lag_estimators_matlab.mean_padded_shift(s, -k, np, axis=0))
