"""Non-steady-state trimming: the leading volumes
fMRIPrep flagged in the confounds are dropped from BOLD and confounds alike. Tests import the
product functions and exercise them on temporary confounds files."""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bold_lag_mapper import io as lio  # noqa: E402

MOTION = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]


def _write_tsv(path, T=20, nss_rows=(), extra_cols=True):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((T, 6)) * 1e-3, columns=MOTION)
    df["framewise_displacement"] = np.r_[np.nan, rng.random(T - 1)]
    for k, row in enumerate(nss_rows):
        col = np.zeros(T); col[row] = 1.0
        df[f"non_steady_state_outlier_{k:02d}"] = col
    if extra_cols:
        df["global_signal"] = rng.standard_normal(T)
    df.to_csv(path, sep="\t", index=False)
    return path


def test_leading_block_is_counted(tmp_path):
    p = _write_tsv(tmp_path / "c.tsv", T=20, nss_rows=(0, 1, 2))
    assert lio.count_leading_non_steady_state(str(p), 20) == 3


def test_no_columns_means_zero(tmp_path):
    p = _write_tsv(tmp_path / "c.tsv", T=20, nss_rows=())
    assert lio.count_leading_non_steady_state(str(p), 20) == 0


def test_txt_confounds_are_never_trimmed(tmp_path):
    p = tmp_path / "motion.txt"
    np.savetxt(p, np.zeros((20, 6)))
    assert lio.count_leading_non_steady_state(str(p), 20) == 0


def test_stray_flag_outside_leading_block_is_kept_and_reported(tmp_path, caplog):
    p = _write_tsv(tmp_path / "c.tsv", T=20, nss_rows=(0, 1, 7))
    with caplog.at_level(logging.WARNING, logger="bold_lag_mapper.io"):
        n = lio.count_leading_non_steady_state(str(p), 20)
    assert n == 2
    assert any("outside the leading block" in m for m in caplog.messages)


def test_mispaired_file_refuses_to_count(tmp_path):
    p = _write_tsv(tmp_path / "c.tsv", T=20, nss_rows=(0,))
    with pytest.raises(ValueError):
        lio.count_leading_non_steady_state(str(p), 21)


def test_load_motion_confounds_drops_the_same_rows(tmp_path):
    p = _write_tsv(tmp_path / "c.tsv", T=20, nss_rows=(0, 1))
    full, names = lio.load_motion_confounds(str(p), 20, True, None)
    trimmed, names2 = lio.load_motion_confounds(str(p), 18, True, None, skip_leading=2)
    assert names == names2 == MOTION
    assert trimmed.shape == (18, 6)
    assert np.allclose(trimmed, full[2:])          # exactly the rows below the leading block
    # the row count is validated AFTER trimming: passing the untrimmed length must fail
    with pytest.raises(ValueError):
        lio.load_motion_confounds(str(p), 20, True, None, skip_leading=2)


def test_load_motion_confounds_refuses_to_drop_everything(tmp_path):
    p = _write_tsv(tmp_path / "c.tsv", T=5, nss_rows=(0,))
    with pytest.raises(ValueError):
        lio.load_motion_confounds(str(p), 0, True, None, skip_leading=5)
