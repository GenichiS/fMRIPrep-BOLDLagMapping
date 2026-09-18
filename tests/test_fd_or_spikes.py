"""Spike rule = DVARS robust-z OR FD > threshold. Product functions only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bold_lag_mapper import lag_preprocess as lp, io as lio  # noqa: E402


def _brain(T=120, n=400, spike_at=None, seed=0):
    rng = np.random.default_rng(seed)
    x = 1000.0 + rng.standard_normal((T, n)) * 5.0
    if spike_at is not None:
        x[spike_at] += 40.0
    return x


def _names(reg, names):
    return set() if reg is None else set(names)


def test_fd_alone_flags_a_frame_that_dvars_misses():
    T = 120; x = _brain(T)
    fd = np.zeros(T); fd[30] = 0.9                      # motion with no DVARS signature in the data
    reg, names = lp.identify_and_create_spike_regressors(x, T, 3.0, True, fd=fd, fd_threshold=0.5)
    assert _names(reg, names) >= {"spike_TR030", "spike_TR029"}
    reg0, names0 = lp.identify_and_create_spike_regressors(x, T, 3.0, True, fd=fd, fd_threshold=0.0)
    assert "spike_TR030" not in _names(reg0, names0)     # threshold 0 disables the FD rule
    reg1, names1 = lp.identify_and_create_spike_regressors(x, T, 3.0, True)          # no FD given
    assert "spike_TR030" not in _names(reg1, names1)


def test_union_of_dvars_and_fd_frames():
    T = 120; x = _brain(T, spike_at=50)
    fd = np.zeros(T); fd[30] = 0.9
    reg, names = lp.identify_and_create_spike_regressors(x, T, 3.0, True, fd=fd, fd_threshold=0.5)
    assert _names(reg, names) >= {"spike_TR049", "spike_TR050", "spike_TR029", "spike_TR030"}
    reg_np, names_np = lp.identify_and_create_spike_regressors(x, T, 3.0, False, fd=fd, fd_threshold=0.5)
    assert {"spike_TR030", "spike_TR050"} <= _names(reg_np, names_np)
    assert "spike_TR029" not in _names(reg_np, names_np)


def test_fd_rule_applies_even_when_dvars_is_degenerate():
    T = 60; x = np.full((T, 50), 1000.0)                 # constant data: DVARS robust SD = 0
    fd = np.zeros(T); fd[10] = 1.2
    reg, names = lp.identify_and_create_spike_regressors(x, T, 3.0, False, fd=fd, fd_threshold=0.5)
    assert _names(reg, names) == {"spike_TR010"}


def test_mispaired_fd_is_refused():
    T = 60; x = _brain(T)
    with pytest.raises(ValueError):
        lp.identify_and_create_spike_regressors(x, T, 3.0, True, fd=np.zeros(T + 1), fd_threshold=0.5)


def test_afyouni_nichols_also_takes_the_fd_rule():
    T = 120; x = _brain(T)
    fd = np.zeros(T); fd[70] = 0.7
    reg, names = lp.identify_and_create_spike_regressors_afyouni_nichols(x, T, fd=fd, fd_threshold=0.5, include_preceding_spike=False)
    assert "spike_TR070" in _names(reg, names)


def test_load_framewise_displacement_trims_like_the_bold(tmp_path):
    T = 20
    df = pd.DataFrame({"rot_x": np.zeros(T), "framewise_displacement": np.r_[np.nan, np.arange(1, T) / 10.0]})
    p = tmp_path / "c.tsv"; df.to_csv(p, sep="\t", index=False)
    fd = lio.load_framewise_displacement(str(p), T)
    assert fd.shape == (T,) and fd[0] == 0.0 and np.isclose(fd[5], 0.5)
    fd2 = lio.load_framewise_displacement(str(p), T - 3, skip_leading=3)
    assert fd2.shape == (T - 3,) and np.isclose(fd2[0], 0.3)
    with pytest.raises(ValueError):
        lio.load_framewise_displacement(str(p), T, skip_leading=3)
    df.drop(columns=["framewise_displacement"]).to_csv(p, sep="\t", index=False)
    assert lio.load_framewise_displacement(str(p), T) is None
    assert lio.load_framewise_displacement(str(tmp_path / "m.txt"), T) is None
