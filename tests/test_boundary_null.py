"""Boundary-bin nulling is the original's behaviour (drLag4Drev7.m drErode_Lag, release rev8hcp L354/L536:
Y(abs(Y)>=MaxLag)=NaN; issue #3 of the original repository) and therefore the default. Candidates are counted
even when nulling is off, so boundary saturation can be reported for any configuration."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from bold_lag_mapper import lag_postprocess  # noqa: E402
from bold_lag_mapper.cli import build_parser  # noqa: E402


def _map():
    lag = np.full((4, 4, 4), np.nan)
    mask = np.zeros((4, 4, 4), bool)
    mask[1:3, 1:3, 1:3] = True
    # TR 2.5, max_lag_trs 2 -> threshold (2 - 0.5) * 2.5 = 3.75 s
    vals = np.array([0.0, 2.5, -2.5, 5.0, -5.0, 4.9, -4.9, 6.2])
    lag[mask] = vals
    return lag, mask


def test_candidates_counted_but_kept_when_disabled():
    lag, mask = _map()
    out, n_cand, n_null, thr = lag_postprocess.null_search_boundary(
        lag, mask, max_lag_trs=2, tr_track=2.5, enabled=False)
    assert thr == 3.75
    assert n_cand == 5 and n_null == 0
    assert np.array_equal(np.isnan(out), np.isnan(lag))


def test_outer_bin_nulled_when_enabled():
    lag, mask = _map()
    out, n_cand, n_null, thr = lag_postprocess.null_search_boundary(
        lag, mask, max_lag_trs=2, tr_track=2.5, enabled=True)
    assert n_cand == n_null == 5
    kept = out[mask]
    assert np.isnan(kept[[3, 4, 5, 6, 7]]).all()
    assert np.isfinite(kept[[0, 1, 2]]).all()
    # the input array is not modified in place
    assert np.isfinite(lag[mask][3])


def test_out_of_mask_never_counted():
    lag, mask = _map()
    lag[0, 0, 0] = 9.0
    _, n_cand, _, _ = lag_postprocess.null_search_boundary(lag, mask, 2, 2.5, True)
    assert n_cand == 5


def test_threshold_uses_tracking_step():
    lag, mask = _map()
    # 1 s tracking step, 7 steps: threshold 6.5 s -> only the 6.2 s voxel is below, none at/above
    _, n_cand, _, thr = lag_postprocess.null_search_boundary(lag, mask, 7, 1.0, True)
    assert thr == 6.5 and n_cand == 0


def test_cli_default_is_on_and_negatable():
    base = ["--bold-files", "x.nii.gz", "--mask-file", "m.nii.gz"]
    assert build_parser().parse_args(base).boundary_null is True
    assert build_parser().parse_args(base + ["--no-boundary-null"]).boundary_null is False
