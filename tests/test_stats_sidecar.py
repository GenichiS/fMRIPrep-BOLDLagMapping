"""Numeric facts of a run are written to a JSON sidecar and the seeds to an NPZ, so reports can read them
instead of retyping; the Aso 1.5 x median spike count is logged for comparison."""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from bold_lag_mapper import io as lio  # noqa: E402
from bold_lag_mapper import lag_preprocess as lp  # noqa: E402


def test_dvars_psc_matches_previous_inline_formula():
    rng = np.random.default_rng(0)
    Y = 1000 + 10 * rng.standard_normal((50, 200))
    d = lp.compute_dvars_psc(Y)
    psc = 100.0 * (Y / Y.mean(0) - 1.0)
    ref = np.insert(np.sqrt(np.mean(np.diff(psc, axis=0) ** 2, axis=1)), 0, 0.0)
    assert d.shape == (50,) and d[0] == 0.0
    assert np.allclose(d, ref)


def test_aso_median_rule_count():
    d = np.array([0.0, 1.0, 1.0, 1.0, 1.6, 1.4, 3.0])
    assert lp.aso_median_rule_count(d, 1.5) == 2          # 1.6 and 3.0 exceed 1.5 * median (1.0)
    assert lp.aso_median_rule_count(np.zeros(5), 1.5) == 0


def test_robustz_detector_still_returns_two_values_and_uses_shared_dvars(caplog):
    import logging
    rng = np.random.default_rng(1)
    x = 1000 + 5 * rng.standard_normal((60, 300))
    x[30] += 40                                            # one obvious spike
    with caplog.at_level(logging.INFO):
        reg, names = lp.identify_and_create_spike_regressors(x, 60, 3.0, True)
    assert reg is not None and "spike_TR030" in names
    assert any("1.5 x median" in m for m in caplog.messages), caplog.messages


def test_sidecar_roundtrip_and_numpy_types():
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "x_desc-stats.json")
        lio.write_stats_sidecar(p, {"n": np.int64(3), "tr": np.float32(2.5),
                                    "runs": [{"spikes_final": np.int64(1)}], "arr": np.arange(3)})
        s = json.load(open(p))
        assert s["n"] == 3 and abs(s["tr"] - 2.5) < 1e-6
        assert s["runs"][0]["spikes_final"] == 1 and s["arr"] == [0, 1, 2]


def test_seeds_npz_keys():
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "x_desc-seeds.npz")
        lio.write_seeds_npz(p, {0: np.ones(5), -2: np.zeros(5), 1: np.arange(5.0)}, 2.5, [5])
        z = np.load(p)
        assert set(z.files) == {"lag_0", "lag_m2", "lag_1", "tr_track", "run_lengths_track"}
        assert float(z["tr_track"]) == 2.5
        assert list(z["run_lengths_track"]) == [5]
        assert z["lag_1"].dtype == np.float32
