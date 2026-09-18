"""Dr Aso's DVARS definition (boldlag einsteining.dvars_spikes) as `--dvars-definition boldlag` (the default), and his
1.5 x median rule as `--spike-method aso-median`. The cross-check against boldlag itself runs when boldlag is
installed (https://github.com/RIKEN-BCIL/HCPstyle-BOLDLagMappingAndCleaning)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bold_lag_mapper import lag_preprocess as lp  # noqa: E402


def _synthetic_4d(T=60, shape=(6, 5, 4), seed=1, spike_at=None):
    """HCP-like 4D block: brain voxels ~1000 (never zero), a corner of air voxels that are zero."""
    rng = np.random.default_rng(seed)
    Y = 1000.0 + 8.0 * rng.standard_normal(shape + (T,))
    Y[:2, :2, :2, :] = 0.0                       # "outside the brain": exactly zero, like HCP volumes
    if spike_at is not None:
        Y[..., spike_at] += 60.0
    return Y.astype(np.float32)


def test_boldlag_dvars_matches_the_boldlag_function():
    pytest.importorskip("boldlag", reason="install boldlag to run the cross-check against Dr Aso's port")
    from boldlag.einsteining import dvars_spikes
    Y = _synthetic_4d(spike_at=30)
    ref_dvars, ref_spikes = dvars_spikes(Y, 1.5)                    # canonical, on the 4D array
    T = Y.shape[-1]
    Y2 = Y.reshape(-1, T).T                                         # (T, V) as the mapper holds it
    ours = lp.compute_dvars_boldlag(Y2)
    assert ours.shape == (T,) and ours[0] == 0.0
    assert np.allclose(ours, ref_dvars, rtol=0, atol=1e-9)         # same numbers, only the summation order differs
    reg, names = lp.identify_and_create_spike_regressors_aso_median(Y2, T, True, fd=None, fd_threshold=0.0)
    ours_frames = sorted(int(n.split("TR")[1]) for n in names)
    assert ours_frames == sorted(int(i) for i in ref_spikes)        # same frames incl. the preceding one
    assert lp.LAST_SPIKE_COUNTS["aso_1p5median"] == int(np.sum(ref_dvars > 1.5 * np.median(ref_dvars)))


def test_near_zero_mean_voxels_own_the_psc_series_but_not_the_boldlag_series():
    """Two voxels with a tiny mean make per-voxel PSC explode and hide a genuine global spike; the
    brain-mean-normalised definition still sees it."""
    rng = np.random.default_rng(3)
    T, n = 120, 400
    x = 1000.0 + 5.0 * rng.standard_normal((T, n))
    x[:, :2] = 0.5 + 0.5 * rng.standard_normal((T, 2))              # mean ~0.5 (0.05 % of the brain mean)
    x[:, :2] = np.abs(x[:, :2]) + 0.05                               # keep them never-zero and positive
    x[50, 2:] += 40.0                                                 # a real spike (4 % of the signal)
    dv_psc, dv_bl = lp.compute_dvars_psc(x), lp.compute_dvars_boldlag(x)
    # PSC: the two tiny-mean voxels dominate every frame's DVARS
    psc_share = (np.diff(100 * (x[:, :2] / x[:, :2].mean(0) - 1), axis=0) ** 2).sum() / \
                (np.diff(100 * (x / x.mean(0) - 1), axis=0) ** 2).sum()
    assert psc_share > 0.9
    # boldlag: the spike frame is the maximum of the series, and robust-z flags it
    assert int(np.argmax(dv_bl)) == 50
    reg_bl, names_bl = lp.identify_and_create_spike_regressors(x, T, 3.0, False, dvars_definition="boldlag")
    assert "spike_TR050" in names_bl
    reg_psc, names_psc = lp.identify_and_create_spike_regressors(x, T, 3.0, False, dvars_definition="psc")
    assert "spike_TR050" not in names_psc                            # the weakness of the PSC definition


def test_sidecar_count_is_faithful_whatever_the_detector_looks_at():
    x = _synthetic_4d(spike_at=20).reshape(-1, 60).T
    x = x[:, (x != 0).all(0)]
    ref = lp.aso_median_rule_count(lp.compute_dvars_boldlag(x))
    for definition in ("psc", "boldlag"):
        lp.identify_and_create_spike_regressors(x, 60, 3.0, True, dvars_definition=definition)
        assert lp.LAST_SPIKE_COUNTS["aso_1p5median"] == ref
    with pytest.raises(ValueError):
        lp.identify_and_create_spike_regressors(x, 60, 3.0, True, dvars_definition="nonsense")


def test_cli_default_spike_rule():
    """Default spike rule: DVARS definition = boldlag (Aso), robust-z k=3 OR FD > 0.5 mm, preceding frame on."""
    from bold_lag_mapper.cli import build_parser
    p = build_parser()
    acts = p._option_string_actions
    assert "aso-median" in acts["--spike-method"].choices and acts["--spike-method"].default == "robustz"
    assert acts["--dvars-definition"].choices == ["psc", "boldlag"] and acts["--dvars-definition"].default == "boldlag"
    assert acts["--dvars-threshold"].default == 3.0 and acts["--fd-spike-threshold"].default == 0.5
