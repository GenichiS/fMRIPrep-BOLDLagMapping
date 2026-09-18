"""The per-run Tukey taper covers 5 % of the run at each end (alpha 0.1), the width of drMerge4D's linear
5 % ramp (Einsteining_v07.m L258/L271). 'maxlag' reproduces the width of earlier versions (max_lag_trs
samples per end, capped at 5 %)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from bold_lag_mapper.lag_preprocess import create_temporal_window  # noqa: E402


def test_default_is_five_percent_and_independent_of_max_lag():
    w1 = create_temporal_window(240, 5.0, 2.5)
    w2 = create_temporal_window(240, 7.5, 2.5)
    assert np.allclose(w1, w2), "the default taper must not depend on the search range"
    assert w1[0] == 0.0 and w1[-1] == 0.0
    # scipy tukey(N, 0.1): alpha*(N-1)/2 = 11.95 samples ramp per end
    assert (w1[:11] < 1.0).all()
    assert np.allclose(w1[13:227], 1.0)


def test_legacy_width_is_max_lag_trs_per_end():
    w = create_temporal_window(240, 5.0, 2.5, taper_width="maxlag")   # max_lag_trs 2 -> alpha 4/240
    assert np.isclose(w[0], 0.0)
    assert w[1] < 1.0
    assert np.allclose(w[3:237], 1.0), "legacy rule tapers only ~2 samples per end"


def test_short_run_still_five_percent():
    w = create_temporal_window(177, 7.5, 2.5)                          # a short run
    assert (w[:8] < 1.0).all()
    assert np.allclose(w[10:167], 1.0)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        create_temporal_window(240, 5.0, 2.5, taper_width="hann")
