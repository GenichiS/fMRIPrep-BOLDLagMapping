"""Input guards and packaging: the .txt motion schema, the second-pass hole fill, run/confounds pairing in the
API, packaging metadata, and the per-run taper."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bold_lag_mapper import io, lag_preprocess as lp, lag_postprocess as post, core  # noqa: E402

ROOT = Path(core.__file__).resolve().parents[1]


# --- .txt motion confounds: the path must work, and the declared schema must be honoured ---

def _write_txt(tmp_path, arr):
    p = tmp_path / "motion.txt"
    np.savetxt(p, arr)
    return str(p)


def test_txt_confounds_spm_schema_gives_canonical_names(tmp_path):
    path = _write_txt(tmp_path, np.zeros((8, 6)))
    params, names = io.load_motion_confounds(path, 8, True, None)
    assert names == ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    augmented, aug_names = lp.calculate_motion_derivatives_and_fd(params, names)
    assert augmented.shape[0] == 8 and any(n.startswith("fd") or "framewise" in n for n in aug_names)


def test_txt_confounds_fsl_schema_swaps_the_column_roles(tmp_path):
    # rotations first (FSL .par). Put a 1 in column 0; under 'fsl' that is rot_x, under 'spm' trans_x.
    arr = np.zeros((8, 6)); arr[:, 0] = 1.0
    path = _write_txt(tmp_path, arr)
    _, names_fsl = io.load_motion_confounds(path, 8, True, None, txt_motion_schema="fsl")
    _, names_spm = io.load_motion_confounds(path, 8, True, None, txt_motion_schema="spm")
    assert names_fsl[0] == "rot_x" and names_spm[0] == "trans_x"


def test_txt_confounds_degrees_are_converted_to_radians(tmp_path):
    arr = np.zeros((4, 6)); arr[:, 3] = 180.0          # HCP: columns 3..5 are rotations, in degrees
    path = _write_txt(tmp_path, arr)
    params, names = io.load_motion_confounds(path, 4, True, None, txt_motion_schema="hcp")
    assert names[3] == "rot_x"
    np.testing.assert_allclose(params[:, 3], np.pi, atol=1e-9)


def test_txt_confounds_unknown_schema_is_refused(tmp_path):
    path = _write_txt(tmp_path, np.zeros((8, 6)))
    with pytest.raises(ValueError, match="Unknown .txt motion schema"):
        io.load_motion_confounds(path, 8, True, None, txt_motion_schema="nonsense")


def test_txt_confounds_hcp_12_column_layout_is_accepted(tmp_path):
    """HCP Movement_Regressors*.txt carries 6 rigid-body columns + their 6 derivatives."""
    arr = np.zeros((10, 12)); arr[:, 3] = 90.0        # rotation column, degrees
    path = _write_txt(tmp_path, arr)
    params, names = io.load_motion_confounds(path, 10, True, None, txt_motion_schema="hcp")
    assert params.shape == (10, 6) and names[3] == "rot_x"
    np.testing.assert_allclose(params[:, 3], np.pi / 2, atol=1e-9)


def test_txt_confounds_wrong_column_count_is_refused(tmp_path):
    path = _write_txt(tmp_path, np.zeros((8, 3)))   # 3 columns: neither layout
    with pytest.raises(ValueError, match=r"6 \(rigid-body only\) or 12"):
        io.load_motion_confounds(path, 8, True, None)


# --- the second-pass hole fill will not run without a mask ---

def test_second_pass_fill_requires_the_mask():
    lag = np.zeros((3, 3, 3)); lag[1, 1, 1] = np.nan
    with pytest.raises(ValueError, match="requires the analysis mask"):
        post.fill_isolated_holes_single_pass(lag, None)


def test_second_pass_fill_uses_only_in_mask_neighbours():
    # 5x5x5, a 3x3x3 mask, one corner hole whose in-mask neighbours are all +5.0 s.
    mask = np.zeros((5, 5, 5), dtype=bool); mask[1:4, 1:4, 1:4] = True
    lag = np.zeros((5, 5, 5)); lag[mask] = 5.0; lag[1, 1, 1] = np.nan
    out = post.fill_isolated_holes_single_pass(lag, mask)
    assert out[1, 1, 1] == pytest.approx(5.0), "out-of-mask zeros must not dilute the fill"


# --- run/confounds pairing is checked in the API, not only in the CLI ---

def test_process_runs_rejects_a_short_confounds_list():
    m = core.BOLDLagMapper(use_gpu=False, generate_mask=False)
    with pytest.raises(ValueError, match="motion-confounds"):
        m.process_runs(["a.nii.gz", "b.nii.gz"], ["only_one.tsv"], "mask.nii.gz", "/tmp")


# --- packaging metadata, checkable without tomllib ---

def test_pyproject_declares_an_importable_backend_and_supported_python():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'build-backend = "setuptools.build_meta"' in text
    assert 'requires-python = ">=3.9"' in text, "argparse.BooleanOptionalAction needs 3.9"
    assert '"data/*.nii.gz"' in text, "the bundled seed must be shipped as package data"


def test_tukey_taper_is_per_run_not_over_the_concatenated_series():
    """Per-run cleaning and taper, THEN concatenate (MATLAB-equivalent).

    The taper must be built from ONE run's length. Building it from the concatenated length leaves the internal
    run boundaries unprotected: with the 'maxlag' rule at 4x420 frames the alpha collapses to 0.0071 (6 TR at
    each outer end only) instead of 0.0286 per run. With the default width (5 % of the RUN at each end, alpha 0.1)
    the per-run taper protects 5 % of every run while a concatenated-length taper would leave the three internal
    junctions with no taper at all - the check is on the junction weights directly.
    """
    from bold_lag_mapper.lag_preprocess import create_temporal_window
    tr, max_lag_s, n_run, n_runs = 0.8, 5.0, 420, 4
    # legacy rule: the fraction argument
    w_run = create_temporal_window(n_run, max_lag_s, tr, taper_width="maxlag")
    w_cat = create_temporal_window(n_run * n_runs, max_lag_s, tr, taper_width="maxlag")
    assert len(w_run) == n_run
    assert w_run[0] < 0.05 and w_run[-1] < 0.05
    frac_run = float((w_run < 0.99).sum()) / n_run
    frac_cat = float((w_cat < 0.99).sum()) / (n_run * n_runs)
    assert frac_run > 3 * frac_cat, (
        f"per-run taper covers {frac_run:.4f} of the series, concatenated-length taper {frac_cat:.4f}; "
        "the taper must be built per run")
    # default rule (5 % per end): a per-run taper is ~0 at every run junction; a taper built from the
    # concatenated length is 1.0 there
    w_run5 = create_temporal_window(n_run, max_lag_s, tr)
    w_cat5 = create_temporal_window(n_run * n_runs, max_lag_s, tr)
    per_run = np.tile(w_run5, n_runs)
    for j in range(1, n_runs):
        assert per_run[j * n_run - 1] < 0.05 and per_run[j * n_run] < 0.05
        assert w_cat5[j * n_run] > 0.99
