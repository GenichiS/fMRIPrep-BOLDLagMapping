"""--fmriprep-dir discovery: every run of the requested space, each with its own confounds, the first run's
mask, short runs dropped, one TR; anything ambiguous or missing stops instead of being guessed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bold_lag_mapper.fmriprep import discover_fmriprep_inputs  # noqa: E402
from fmriprep_tree import write_run  # noqa: E402


def _func(root, sub="01", ses=None):
    d = root / f"sub-{sub}"
    return (d / f"ses-{ses}" / "func") if ses else (d / "func")


def test_single_run_is_paired_with_its_mask_and_confounds(tmp_path):
    b = write_run(_func(tmp_path), "sub-01_task-rest_run-01")
    got = discover_fmriprep_inputs(tmp_path, "01")
    assert got.bold_files == [str(b)]
    assert got.confounds_files == [str(b.parent / "sub-01_task-rest_run-01_desc-confounds_timeseries.tsv")]
    assert got.mask_file == str(b.parent / "sub-01_task-rest_run-01_space-T1w_desc-brain_mask.nii.gz")
    assert got.tr == 2.5 and got.dropped == []


def test_runs_are_sorted_and_the_first_runs_mask_is_used(tmp_path):
    b2 = write_run(_func(tmp_path), "sub-01_task-rest_run-02")
    b1 = write_run(_func(tmp_path), "sub-01_task-rest_run-01")
    got = discover_fmriprep_inputs(tmp_path, "sub-01")
    assert got.bold_files == [str(b1), str(b2)]
    assert [Path(c).name for c in got.confounds_files] == [
        "sub-01_task-rest_run-01_desc-confounds_timeseries.tsv", "sub-01_task-rest_run-02_desc-confounds_timeseries.tsv"]
    assert got.mask_file.endswith("sub-01_task-rest_run-01_space-T1w_desc-brain_mask.nii.gz")


def test_only_the_requested_space_is_used(tmp_path):
    write_run(_func(tmp_path), "sub-01_task-rest_run-01", space="T1w")
    m = write_run(_func(tmp_path), "sub-01_task-rest_run-01", space="MNI152NLin2009cAsym", res="2")
    got = discover_fmriprep_inputs(tmp_path, "01", space="MNI152NLin2009cAsym")
    assert got.bold_files == [str(m)]


def test_several_resolutions_need_res(tmp_path):
    write_run(_func(tmp_path), "sub-01_task-rest_run-01", space="MNI152NLin2009cAsym", res="1")
    r2 = write_run(_func(tmp_path), "sub-01_task-rest_run-01", space="MNI152NLin2009cAsym", res="2")
    with pytest.raises(ValueError, match="--res"):
        discover_fmriprep_inputs(tmp_path, "01", space="MNI152NLin2009cAsym")
    assert discover_fmriprep_inputs(tmp_path, "01", space="MNI152NLin2009cAsym", res="2").bold_files == [str(r2)]


def test_short_runs_are_dropped_and_reported(tmp_path):
    write_run(_func(tmp_path), "sub-01_task-rest_run-01", n=100)
    b2 = write_run(_func(tmp_path), "sub-01_task-rest_run-02", n=130)
    got = discover_fmriprep_inputs(tmp_path, "01")
    assert got.bold_files == [str(b2)]
    assert got.dropped == ["sub-01_task-rest_run-01_space-T1w_desc-preproc_bold.nii.gz"]
    assert got.mask_file.endswith("run-02_space-T1w_desc-brain_mask.nii.gz")


def test_all_runs_short_is_an_error(tmp_path):
    write_run(_func(tmp_path), "sub-01_task-rest_run-01", n=100)
    with pytest.raises(ValueError, match="shorter than 120"):
        discover_fmriprep_inputs(tmp_path, "01")


def test_missing_confounds_is_an_error(tmp_path):
    write_run(_func(tmp_path), "sub-01_task-rest_run-01", conf=False)
    with pytest.raises(FileNotFoundError, match="desc-confounds_timeseries.tsv"):
        discover_fmriprep_inputs(tmp_path, "01")


def test_missing_mask_is_an_error(tmp_path):
    write_run(_func(tmp_path), "sub-01_task-rest_run-01", mask=False)
    with pytest.raises(FileNotFoundError, match="desc-brain_mask"):
        discover_fmriprep_inputs(tmp_path, "01")


def test_several_sessions_need_session(tmp_path):
    write_run(_func(tmp_path, ses="1"), "sub-01_ses-1_task-rest_run-01")
    b = write_run(_func(tmp_path, ses="2"), "sub-01_ses-2_task-rest_run-01")
    with pytest.raises(ValueError, match="--session"):
        discover_fmriprep_inputs(tmp_path, "01")
    assert discover_fmriprep_inputs(tmp_path, "01", session="2").bold_files == [str(b)]


def test_several_tasks_need_task(tmp_path):
    write_run(_func(tmp_path), "sub-01_task-rest_run-01")
    write_run(_func(tmp_path), "sub-01_task-nback_run-01")
    with pytest.raises(ValueError, match="--task"):
        discover_fmriprep_inputs(tmp_path, "01")


def test_run_fragments_select_runs(tmp_path):
    write_run(_func(tmp_path), "sub-01_task-rest_dir-AP_run-01")
    b = write_run(_func(tmp_path), "sub-01_task-rest_dir-PA_run-01")
    assert discover_fmriprep_inputs(tmp_path, "01", runs=["dir-PA_run-01"]).bold_files == [str(b)]


def test_json_tr_must_match_the_header(tmp_path):
    write_run(_func(tmp_path), "sub-01_task-rest_run-01", tr=2.5, json_tr=2.0)
    with pytest.raises(ValueError, match="RepetitionTime"):
        discover_fmriprep_inputs(tmp_path, "01")


def test_runs_with_different_trs_are_an_error(tmp_path):
    write_run(_func(tmp_path), "sub-01_task-rest_run-01", tr=2.5)
    write_run(_func(tmp_path), "sub-01_task-rest_run-02", tr=2.0)
    with pytest.raises(ValueError, match="different TRs"):
        discover_fmriprep_inputs(tmp_path, "01")


def test_unknown_subject_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="sub-99"):
        discover_fmriprep_inputs(tmp_path, "99")
