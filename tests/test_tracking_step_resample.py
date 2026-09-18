"""Optional resampling of the filtered series to a fixed tracking step
(drLag4Drev7_longTR / boldlag lag4d --reso, scipy resample_poly with a Kaiser window). Integer
tracking on a 1 s grid must recover continuous lags at TR 2.5 to within the 1 s quantisation, and the
TR gate must be evaluated on the tracking STEP."""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bold_lag_mapper import lag_estimators_matlab as est  # noqa: E402
from bold_lag_mapper.core import resolve_effective_tracking_method  # noqa: E402

TR, N = 2.5, 240


def resample_for_tracking(*a, **k):
    # imported lazily so a missing implementation fails these tests only, not the whole collection
    from bold_lag_mapper.lag_preprocess import resample_for_tracking as _f
    return _f(*a, **k)


class _Mapper:
    def __init__(self, n_vox, tr_track):
        self.xp = np; self.use_gpu = False; self.verbose = False; self.num_voxels = n_vox
        self.tr = TR; self.tr_track = tr_track; self.max_lag_seconds = 7.0
        self.min_corr_threshold = 0.2; self.seed_time_series = {}; self.seed_update = "matlab"


def _synthetic(seed=0, n_vox=2000, noise=0.25):
    rng = np.random.default_rng(seed)
    dt = 0.05
    t_fine = np.arange(-30.0, N * TR + 30.0, dt)
    base = gaussian_filter1d(rng.standard_normal(t_fine.size), sigma=5.0 / dt)      # <= ~0.06 Hz content
    base -= gaussian_filter1d(base, sigma=60.0 / dt)
    base /= base.std()
    lag_s = rng.triangular(-6.0, 0.0, 6.0, n_vox)
    t_k = np.arange(N) * TR
    X = np.stack([np.interp(t_k + L, t_fine, base) for L in lag_s], axis=1)     # positive lag = leads
    X += noise * rng.standard_normal(X.shape)
    X = (X - X.mean(0)) / X.std(0)
    return X, X[:, np.abs(lag_s) < 1.0].mean(1), lag_s


def test_resample_length_and_dtype():
    X = np.random.default_rng(1).standard_normal((240, 7)).astype(np.float32)
    Y = resample_for_tracking(X, 2.5, 1.0)
    assert Y.shape == (600, 7) and Y.dtype == np.float32
    assert resample_for_tracking(X, 0.8, 1.0).shape == (192, 7)          # short TR: a down-sample


def test_resample_preserves_a_band_limited_signal():
    t = np.arange(240) * 2.5
    x = np.sin(2 * np.pi * 0.05 * t)[:, None]
    y = resample_for_tracking(x, 2.5, 1.0)[:, 0]
    t1 = np.arange(len(y)) * 1.0
    ref = np.sin(2 * np.pi * 0.05 * t1)
    inner = slice(30, len(y) - 30)                                       # away from the edges
    assert np.corrcoef(y[inner], ref[inner])[0, 1] > 0.999


def test_recursive_on_1s_grid_recovers_lags_within_quantisation():
    X, seed, truth = _synthetic()
    Xr = resample_for_tracking(X, TR, 1.0)
    seedr = resample_for_tracking(seed[:, None], TR, 1.0)[:, 0]
    Xr = (Xr - Xr.mean(0)) / Xr.std(0)
    m = _Mapper(X.shape[1], tr_track=1.0)
    lag_s, corr, _ = est.compute_lag_maps_recursive(m, Xr, seedr / seedr.std())
    ok = np.isfinite(lag_s) & (np.abs(truth) <= 5.0)
    assert ok.mean() > 0.6
    assert np.mean(np.abs(lag_s[ok] - truth[ok]) <= 1.0) > 0.7          # 1 s step -> |err| <= 1 s for most voxels
    assert np.nanmax(np.abs(lag_s)) <= 7.0 + 1e-6                        # 7 steps of 1 s


def test_gate_uses_the_tracking_step():
    assert resolve_effective_tracking_method("recursive_subtr", 1.0, 1.5) == "recursive"
    assert resolve_effective_tracking_method("fixed_subtr", 1.0, 1.5) == "fixed"
    assert resolve_effective_tracking_method("recursive", 1.0, 1.5) == "recursive"
