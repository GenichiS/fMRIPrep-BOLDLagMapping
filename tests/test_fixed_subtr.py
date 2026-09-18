"""Fixed-seed tracking (MATLAB FIXED=1 propagation) with the same 5-point sub-TR
refinement as recursive_subtr.

Synthetic truth as in test_subtr_phase_tracking: voxels are copies of one sLFO shifted by known
continuous lags (densest at 0). A fixed seed cannot accumulate phase error step by step, so the
|p| = 2 bin must be unbiased without any tracking machinery."""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bold_lag_mapper import lag_estimators_matlab as est  # noqa: E402
from bold_lag_mapper.core import resolve_effective_tracking_method  # noqa: E402

TR, N = 2.5, 240


class _Mapper:
    def __init__(self, n_vox, **kw):
        self.xp = np; self.use_gpu = False; self.verbose = False
        self.num_voxels = n_vox; self.tr = TR; self.max_lag_seconds = 7.5
        self.min_corr_threshold = 0.2; self.seed_time_series = {}
        for k, v in kw.items():
            setattr(self, k, v)


def _synthetic(seed=0, n_vox=3000, noise=0.25):
    rng = np.random.default_rng(seed)
    dt = 0.05
    t_fine = np.arange(-30.0, N * TR + 30.0, dt)
    base = gaussian_filter1d(rng.standard_normal(t_fine.size), sigma=4.0 / dt)      # ~0.04 Hz content
    base -= gaussian_filter1d(base, sigma=60.0 / dt)
    base /= base.std()
    lag_tr = rng.triangular(-2.6, 0.0, 2.6, n_vox)
    t_k = np.arange(N) * TR
    X = np.stack([np.interp(t_k + L * TR, t_fine, base) for L in lag_tr], axis=1)
    X += noise * rng.standard_normal(X.shape)
    X = (X - X.mean(0)) / X.std(0)
    roi_seed = X[:, np.abs(lag_tr) < 0.5].mean(1)
    return X, roi_seed, lag_tr


def test_fixed_subtr_recovers_continuous_lags_without_step_bias():
    X, seed, truth = _synthetic()
    m = _Mapper(X.shape[1])
    lag_s, corr, lag0 = est.compute_lag_maps_fixed_subtr(m, X.copy(), seed.copy())
    e = lag_s / TR
    assert np.isfinite(e).mean() > 0.8, "too few voxels assigned"
    sel = np.isfinite(e) & (np.abs(truth) <= 2.5)
    assert np.mean(np.abs(e[sel] - truth[sel])) < 0.30
    sel2 = np.isfinite(e) & (np.abs(np.round(truth)) == 2)
    assert sel2.sum() > 100
    signed = np.mean((e[sel2] - truth[sel2]) * np.sign(truth[sel2]))
    assert abs(signed) < 0.06, f"|p|=2 bias {signed:+.3f} TR"
    assert set(m.seed_time_series) == {0}
    assert 0 in m.seed_phase_trs and m.seed_voxel_counts[0] == int(lag0.sum())
    assert np.isfinite(corr[np.isfinite(e)]).all()


def test_fixed_subtr_lags_stay_within_the_search_range():
    X, seed, _ = _synthetic(seed=4, n_vox=1500)
    m = _Mapper(X.shape[1])
    lag_s, _, _ = est.compute_lag_maps_fixed_subtr(m, X.copy(), seed.copy())
    max_lag_trs = round(m.max_lag_seconds / TR)                         # 3
    assert np.nanmax(np.abs(lag_s)) <= (max_lag_trs + 0.5) * TR + 1e-6


def test_tr_gate_maps_fixed_subtr_to_fixed():
    assert resolve_effective_tracking_method("fixed_subtr", 0.8, 1.5) == "fixed"
    assert resolve_effective_tracking_method("fixed_subtr", 2.5, 1.5) == "fixed_subtr"
    assert resolve_effective_tracking_method("recursive_subtr", 0.8, 1.5) == "recursive"
