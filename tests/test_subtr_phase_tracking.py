"""The recursive sub-TR estimator must carry the seed's phase.

Synthetic ground truth: one smooth reference signal, voxels are copies shifted by a KNOWN
continuous lag drawn from a distribution that is densest at zero (as real lag maps are). Because
each integer bin is denser on its zero-facing side, the voxels found at |p| = 1 have a non-zero
mean sub-TR offset, the next seed inherits that phase, and without tracking every voxel assigned
at |p| = 2 is pushed away from zero by that amount. The product estimator is run twice on the same
data; the only difference is `subtr_phase_tracking`.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bold_lag_mapper import lag_estimators_matlab as est  # noqa: E402

TR, N = 2.5, 240


class _Mapper:
    def __init__(self, n_vox, **kw):
        self.xp = np; self.use_gpu = False; self.verbose = False
        self.num_voxels = n_vox; self.tr = TR; self.max_lag_seconds = 5.0
        self.min_corr_threshold = 0.2; self.seed_time_series = {}
        for k, v in kw.items():
            setattr(self, k, v)


def _synthetic(seed=0, n_vox=3000, noise=0.25):
    rng = np.random.default_rng(seed)
    dt = 0.05
    t_fine = np.arange(-30.0, N * TR + 30.0, dt)
    base = gaussian_filter1d(rng.standard_normal(t_fine.size), sigma=4.0 / dt)      # ~0.04 Hz content
    base -= gaussian_filter1d(base, sigma=60.0 / dt)                                  # drop very low freq
    base /= base.std()
    # true lags (TR): triangular density peaking at 0, support +/-2.6 TR -> bins p = -2..+2 populated
    lag_tr = rng.triangular(-2.6, 0.0, 2.6, n_vox)
    t_k = np.arange(N) * TR
    # positive lag = voxel LEADS the seed: x(t) = base(t + lag)
    X = np.stack([np.interp(t_k + L * TR, t_fine, base) for L in lag_tr], axis=1)
    X += noise * rng.standard_normal(X.shape)
    X = (X - X.mean(0)) / X.std(0)
    roi_seed = X[:, np.abs(lag_tr) < 0.5].mean(1)          # the "deep WM ROI" seed
    return X, roi_seed, lag_tr


def _run(X, seed, **kw):
    m = _Mapper(X.shape[1], **kw)
    lag_s, corr, _ = est.compute_lag_maps_recursive_subtr(m, X.copy(), seed.copy())
    return lag_s / TR, corr, m


def test_phase_tracking_removes_the_step_bias_at_p2():
    X, seed, truth = _synthetic()
    est_off, _, _ = _run(X, seed, subtr_phase_tracking=False)
    est_on, _, _ = _run(X, seed, subtr_phase_tracking=True)

    for name, e in (("off", est_off), ("on", est_on)):
        assert np.isfinite(e).mean() > 0.8, f"{name}: too few voxels assigned"

    sel = np.isfinite(est_off) & np.isfinite(est_on) & (np.abs(np.round(truth)) == 2)
    assert sel.sum() > 100
    err_off = est_off[sel] - truth[sel]
    err_on = est_on[sel] - truth[sel]
    # the untracked estimator is biased AWAY from zero at |p|=2: sign(err) == sign(lag)
    signed_off = np.mean(err_off * np.sign(truth[sel]))
    signed_on = np.mean(err_on * np.sign(truth[sel]))
    assert signed_off > 0.04, f"the untracked bias the fix targets is not reproduced (got {signed_off:+.3f} TR)"
    assert abs(signed_on) < 0.5 * abs(signed_off), f"tracking did not remove the bias: off {signed_off:+.3f}, on {signed_on:+.3f} TR"
    assert abs(signed_on) < 0.06, f"residual |p|=2 bias with tracking too large: {signed_on:+.3f} TR"


def test_phase_tracking_records_the_seed_phase_and_leaves_p1_essentially_unchanged():
    X, seed, truth = _synthetic(seed=1)
    est_off, _, m_off = _run(X, seed, subtr_phase_tracking=False)
    est_on, _, m_on = _run(X, seed, subtr_phase_tracking=True)
    assert set(m_on.seed_phase_trs) >= {0, -1, 1}
    assert m_on.seed_phase_trs[-1] != 0.0 and m_on.seed_phase_trs[1] != 0.0
    assert all(v == 0.0 for v in m_off.seed_phase_trs.values())
    # for voxels ASSIGNED at |p| = 1 (select on the untracked estimate's integer step, not on the
    # truth: a voxel with true lag 1.45 can be assigned at p = 2) the only difference is the
    # (tiny) phase of the lag-0 seed itself, identical for every such voxel
    sel = np.isfinite(est_off) & np.isfinite(est_on) & (np.abs(np.round(est_off)) == 1)
    diff = est_on[sel] - est_off[sel]
    assert sel.sum() > 100
    assert np.max(np.abs(diff)) < 0.05
    assert np.ptp(diff) < 1e-6, "the p=1 shift must be one constant (the lag-0 seed phase)"
    assert np.isclose(diff[0], m_on.seed_phase_trs[0], atol=1e-6)


def test_seed_update_matlab_includes_already_assigned_voxels():
    X, seed, _ = _synthetic(seed=2)
    _, _, m_new = _run(X, seed, seed_update="newly_found")
    _, _, m_mat = _run(X, seed, seed_update="matlab")
    for p in (-1, 1):
        assert m_new.seed_voxel_counts[p] == m_new.newly_found_counts[p]
        assert m_mat.seed_voxel_counts[p] >= m_mat.newly_found_counts[p]
    # with a broad, smooth correlation peak some lag-0 voxels still peak at the centre after a
    # one-TR shift, so the MATLAB rule must enlarge at least one seed
    assert any(m_mat.seed_voxel_counts[p] > m_mat.newly_found_counts[p] for p in (-1, 1))


def test_seed_update_rejects_unknown_mode():
    X, seed, _ = _synthetic(seed=3, n_vox=300)
    with pytest.raises(ValueError):
        _run(X, seed, seed_update="bogus")
