"""Aso's 'lag structure' view - the seed of every tracking step shifted to the
voxels it represents, one line per lag coloured by lag (boldlag viewer.lag_structure_plot) - and the
boundary-seed similarity r(seed0 shifted +max, seed0 shifted -max), the quantity behind issue #3 of the
original repository."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from bold_lag_mapper import viz  # noqa: E402


def _seeds(freq_hz, tr, n=240, max_step=3):
    t = np.arange(n) * tr
    s0 = np.sin(2 * np.pi * freq_hz * t)
    return {k: np.roll(s0, -k) for k in range(-max_step, max_step + 1)}


def test_rainbow_plot_written_and_boundary_r_in_range():
    seeds = _seeds(0.05, 2.5)                                # period 20 s, +/-3 steps = +/-7.5 s
    with tempfile.TemporaryDirectory() as d:
        r = viz.save_slfo_rainbow_plot(seeds, 2.5, os.path.join(d, "x"), [240], lim_seconds=7.5)
        assert os.path.exists(os.path.join(d, "x_slfo_rainbow.png"))
        assert -1.0 <= r <= 1.0
        # shift difference 15 s = 0.75 period -> the two boundary references are ~orthogonal
        assert abs(r) < 0.2


def test_boundary_r_is_high_when_the_span_is_a_full_period():
    # 0.0667 Hz (period 15 s): +/-7.5 s references differ by exactly one period -> near-identical
    seeds = _seeds(1.0 / 15.0, 2.5)
    with tempfile.TemporaryDirectory() as d:
        r = viz.save_slfo_rainbow_plot(seeds, 2.5, os.path.join(d, "y"), [240], lim_seconds=7.5)
        assert r > 0.9


def test_run_boundaries_and_missing_steps_are_tolerated():
    seeds = _seeds(0.05, 0.8, n=400, max_step=6)
    del seeds[4]                                              # a step with no voxels found
    with tempfile.TemporaryDirectory() as d:
        r = viz.save_slfo_rainbow_plot(seeds, 0.8, os.path.join(d, "z"), [200, 200], lim_seconds=5.0)
        assert np.isfinite(r)


def test_colourbar_label_fits_and_names_the_sign(monkeypatch):
    """The colour-bar label must fit the short figure: 'lag (s)' on the bar and 'earlier' / 'later' at its
    ends (a positive lag leads the seed), as in the lag-map figures; no interpretive wording."""
    import matplotlib.figure
    saved = []
    savefig = matplotlib.figure.Figure.savefig

    def keep(fig, *args, **kwargs):
        saved.append(fig)
        return savefig(fig, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", keep)
    with tempfile.TemporaryDirectory() as d:
        viz.save_slfo_rainbow_plot(_seeds(0.05, 2.5), 2.5, os.path.join(d, "c"), [240], lim_seconds=7.5)
    assert len(saved) == 1
    cbar = saved[0].axes[-1]
    assert cbar.get_ylabel() == "lag (s)"
    ends = {t.get_text(): t.get_position()[1] for t in cbar.texts}
    assert set(ends) == {"earlier", "later"}
    assert ends["earlier"] > ends["later"]                     # earlier (positive) at the top


def test_empty_seeds_return_nan_without_writing():
    with tempfile.TemporaryDirectory() as d:
        r = viz.save_slfo_rainbow_plot({}, 2.5, os.path.join(d, "e"), [240], lim_seconds=7.5)
        assert np.isnan(r)
        assert not os.path.exists(os.path.join(d, "e_slfo_rainbow.png"))
