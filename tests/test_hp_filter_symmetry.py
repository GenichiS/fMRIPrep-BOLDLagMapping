"""High-pass filter window symmetry - tested against the PRODUCT, not a copy of it.

A test that reimplements the logic it is meant to check verifies nothing, so this one imports the real filter
and infers the window from behaviour: a high-pass filter whose local fit window (2r+1 samples centred on t) is
symmetric about t must commute with time reversal in the interior.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bold_lag_mapper import lag_preprocess  # noqa: E402


class _Mapper:
    """Minimal stand-in for the LagMapper attributes temporal_filter_fsl_replicated reads."""

    def __init__(self, n_vox=8):
        self.xp = np
        self.use_gpu = False
        self.gpu_batch_size = max(n_vox, 1)
        self.verbose = False
        self.lp_filter_mode = "reflect"


def _hp_only(data, hp_sigma):
    return lag_preprocess.temporal_filter_fsl_replicated(_Mapper(data.shape[1]), data, hp_sigma, 0.0)


def test_hp_filter_is_time_reversal_symmetric():
    """A window symmetric about t commutes with time reversal; an asymmetric one does not.

    This is the assertion the old test only pretended to make: with the shipped off-by-one it fails.
    """
    rng = np.random.default_rng(0)
    n, v = 120, 4
    data = np.cumsum(rng.standard_normal((n, v)), axis=0)  # smooth, so the local fit is well posed

    forward = _hp_only(data.copy(), hp_sigma=5.0)
    reversed_out = _hp_only(data[::-1].copy(), hp_sigma=5.0)[::-1]

    interior = slice(30, n - 30)  # the truncated end windows are a genuinely different problem
    assert np.allclose(forward[interior], reversed_out[interior], atol=1e-6), (
        "high-pass output is not invariant to time reversal in the interior, which means the local "
        "fit window is not symmetric about t"
    )


def test_hp_filter_removes_a_constant():
    data = np.ones((60, 3)) * 7.0
    out = _hp_only(data.copy(), hp_sigma=4.0)
    assert out.shape == data.shape
    assert np.allclose(out, 0.0, atol=1e-6), "high-pass did not remove a DC offset"


@pytest.mark.parametrize("hp_sigma", [2.0, 5.0, 12.0])
def test_hp_filter_is_finite_for_several_kernel_widths(hp_sigma):
    rng = np.random.default_rng(1)
    data = np.cumsum(rng.standard_normal((80, 2)), axis=0)
    out = _hp_only(data.copy(), hp_sigma=hp_sigma)
    assert np.isfinite(out).all(), "high-pass produced non-finite values"
