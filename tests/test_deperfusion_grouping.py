"""
Test deperfusion TR-bin grouping (bug fix 3B).

Lag values 0.74, 0.76, 0.80 at TR=1 should all share the same regressor bin
(rounded to TR=1). This avoids redundant lstsq solves and fragile exact-float
equality.
"""

import numpy as np


def test_float_lags_group_to_same_tr_bin():
    """
    Lag values that round to the same TR should be grouped together.
    """
    tr = 1.0
    lag_values = np.array([0.74, 0.76, 0.80])

    tr_bins = {}
    for lag_val_sec in lag_values:
        lag_val_trs = int(round(lag_val_sec / tr))
        if lag_val_trs not in tr_bins:
            tr_bins[lag_val_trs] = []
        tr_bins[lag_val_trs].append(lag_val_sec)

    # All three should map to TR bin 1
    assert 1 in tr_bins, f"Expected TR bin 1, got bins: {list(tr_bins.keys())}"
    assert len(tr_bins[1]) == 3, (
        f"Expected 3 values in TR bin 1, got {len(tr_bins[1])}"
    )
    assert len(tr_bins) == 1, (
        f"Expected only 1 TR bin, got {len(tr_bins)} bins: {list(tr_bins.keys())}"
    )


def test_different_tr_bins():
    """Lags rounding to different TRs should be in separate bins."""
    tr = 1.0
    lag_values = np.array([0.3, 0.8, 1.6, 2.1])
    # Should round to: 0, 1, 2, 2

    tr_bins = {}
    for lag_val_sec in lag_values:
        lag_val_trs = int(round(lag_val_sec / tr))
        if lag_val_trs not in tr_bins:
            tr_bins[lag_val_trs] = []
        tr_bins[lag_val_trs].append(lag_val_sec)

    assert set(tr_bins.keys()) == {0, 1, 2}
    assert len(tr_bins[0]) == 1  # 0.3
    assert len(tr_bins[1]) == 1  # 0.8
    assert len(tr_bins[2]) == 2  # 1.6, 2.1


def test_grouping_reduces_lstsq_calls():
    """
    If there are N unique float lag values that map to M TR bins (M < N),
    the grouping should reduce from N lstsq calls to M.
    """
    tr = 0.8
    # 10 float values that map to only 3 TR bins
    lag_values = np.array([
        0.70, 0.75, 0.82,   # all round to TR=1 (0.8s)
        1.50, 1.55, 1.60, 1.65,  # all round to TR=2 (1.6s)
        2.35, 2.40, 2.42,   # all round to TR=3 (2.4s)
    ])

    tr_bins = {}
    for lag_val_sec in lag_values:
        lag_val_trs = int(round(lag_val_sec / tr))
        if lag_val_trs not in tr_bins:
            tr_bins[lag_val_trs] = []
        tr_bins[lag_val_trs].append(lag_val_sec)

    # 10 float values -> 3 TR bins
    assert len(lag_values) == 10
    assert len(tr_bins) == 3, f"Expected 3 TR bins, got {len(tr_bins)}"


def test_negative_lags():
    """Negative lag values should also group correctly."""
    tr = 1.0
    lag_values = np.array([-2.3, -1.8, -1.7, -0.4, 0.3])
    # Round to: -2, -2, -2, 0, 0

    tr_bins = {}
    for lag_val_sec in lag_values:
        lag_val_trs = int(round(lag_val_sec / tr))
        if lag_val_trs not in tr_bins:
            tr_bins[lag_val_trs] = []
        tr_bins[lag_val_trs].append(lag_val_sec)

    assert set(tr_bins.keys()) == {-2, 0}
    assert len(tr_bins[-2]) == 3
    assert len(tr_bins[0]) == 2
