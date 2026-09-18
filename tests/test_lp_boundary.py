"""
Test LP filter boundary mode for multi-run data (fix 2C).

Two synthetic runs with distinct endpoints: wrap vs nearest should differ
at the concatenation boundary.
"""

import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal.windows import gaussian


def _make_two_run_signal(n_per_run=50):
    """Create a concatenated signal from two runs with distinct endpoints."""
    run1 = np.ones(n_per_run) * 10.0  # constant high
    run2 = np.ones(n_per_run) * -10.0  # constant low
    return np.concatenate([run1, run2])


def _apply_lp_filter(signal, sigma, mode):
    """Apply a Gaussian low-pass filter with the specified boundary mode."""
    kernel_radius = int(np.ceil(3 * sigma))
    kernel_size = 2 * kernel_radius + 1
    kernel = gaussian(kernel_size, std=sigma)
    kernel = kernel / np.sum(kernel)
    return convolve1d(signal, kernel, axis=0, mode=mode)


def test_wrap_vs_nearest_differ_at_boundary():
    """
    With 'wrap', the end of run2 bleeds into the start of run1.
    With 'nearest', this doesn't happen.
    They should produce different results at the boundaries.
    """
    signal = _make_two_run_signal()
    sigma = 3.0

    filtered_wrap = _apply_lp_filter(signal, sigma, mode='wrap')
    filtered_nearest = _apply_lp_filter(signal, sigma, mode='nearest')

    # At the start of the concatenated signal (first few points):
    # wrap mode: bleeds from end of run2 (negative values) -> shifts result negative
    # nearest mode: pads with run1's first value (positive) -> stays positive
    start_diff = abs(filtered_wrap[0] - filtered_nearest[0])
    assert start_diff > 0.1, (
        f"Expected wrap and nearest to differ at start, "
        f"but diff is only {start_diff:.6f}"
    )


def test_wrap_introduces_cross_run_contamination():
    """
    'wrap' mode at position 0 should be influenced by the end of the signal
    (which is from a different run).
    """
    signal = _make_two_run_signal()
    sigma = 3.0

    filtered_wrap = _apply_lp_filter(signal, sigma, mode='wrap')

    # Original signal at position 0 is 10.0 (run1)
    # With wrap, the filter at position 0 sees values from the end (run2 = -10.0)
    # So the filtered value should be pulled below the original
    assert filtered_wrap[0] < signal[0], (
        f"Wrap mode should contaminate start of signal: "
        f"filtered={filtered_wrap[0]:.2f}, original={signal[0]:.2f}"
    )


def test_nearest_preserves_boundary():
    """
    'nearest' mode should pad with edge values, keeping boundaries clean.
    """
    signal = _make_two_run_signal()
    sigma = 3.0

    filtered_nearest = _apply_lp_filter(signal, sigma, mode='nearest')

    # With nearest mode, position 0 should stay close to 10.0
    # (padded with 10.0 on the left side)
    assert abs(filtered_nearest[0] - signal[0]) < 1.0, (
        f"Nearest mode should preserve boundary: "
        f"filtered={filtered_nearest[0]:.2f}, original={signal[0]:.2f}"
    )
