"""Public CLI: the recommended settings are the defaults, only the MATLAB-family methods exist, and --fmriprep-dir
fills the inputs and switches native-space processing on for T1w input."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bold_lag_mapper import cli  # noqa: E402
from fmriprep_tree import write_run  # noqa: E402


def _parse(*argv):
    return cli.build_parser().parse_args(list(argv))


def test_recommended_settings_are_the_defaults():
    a = _parse()
    assert (a.spatial_fwhm, a.bandpass_high, a.max_lag_seconds) == (6.0, 0.09, 7.0)
    assert a.tracking_step_seconds == "auto" and a.tracking_method == "recursive" and a.seed_roi_file == "builtin"
    assert (a.min_corr_threshold, a.amplitude_threshold, a.dilate_mask_mm) == (0.2, 4.0, 4.0)
    assert (a.spike_method, a.dvars_definition, a.dvars_threshold, a.fd_spike_threshold) == ("robustz", "boldlag", 3.0, 0.5)
    assert a.boundary_null and a.taper_width == "fixed5pct" and a.seed_update == "matlab" and a.trim_non_steady_state
    assert a.min_volumes == 120 and a.space == "T1w"


def test_linked_selects_the_range_linked_low_pass():
    assert _parse("--bandpass-high", "linked").bandpass_high is None
    assert _parse("--bandpass-high", "0.06").bandpass_high == 0.06


def test_tracking_step_values():
    assert _parse("--tracking-step-seconds", "none").tracking_step_seconds == "none"
    assert _parse("--tracking-step-seconds", "1").tracking_step_seconds == 1.0
    with pytest.raises(SystemExit):
        _parse("--tracking-step-seconds", "fast")


def test_only_the_matlab_family_is_offered():
    action = {a.dest: a for a in cli.build_parser()._actions}["tracking_method"]
    assert list(action.choices) == ["recursive", "recursive_subtr", "fixed", "fixed_subtr"]


@pytest.mark.parametrize("opt", ["--peak-selection-method", "--interpolation-method", "--prior-mask", "--prior-strength",
                                 "--spatial-regularization-fwhm", "--max-iterations", "--convergence-threshold", "--n-seeds"])
def test_experimental_options_are_gone(opt):
    with pytest.raises(SystemExit):
        _parse(opt, "1")


def _run_main(argv):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(process_runs=lambda *a, **k: captured.update(process_args=a))

    with patch.object(sys, "argv", ["bold-lag-mapper", *argv]), patch.object(cli, "BOLDLagMapper", side_effect=capture):
        cli.main()
    return captured


def test_fmriprep_mode_fills_inputs_and_switches_native_space_on(tmp_path):
    root = tmp_path / "fmriprep"
    b1 = write_run(root / "sub-01" / "func", "sub-01_task-rest_run-01")
    b2 = write_run(root / "sub-01" / "func", "sub-01_task-rest_run-02")
    got = _run_main(["--fmriprep-dir", str(root), "--participant-label", "01", "--output-dir", str(tmp_path / "out")])
    assert got["bold_files"] == [str(b1), str(b2)]
    assert [Path(c).name for c in got["motion_confounds_files"]] == [
        "sub-01_task-rest_run-01_desc-confounds_timeseries.tsv", "sub-01_task-rest_run-02_desc-confounds_timeseries.tsv"]
    assert got["mask_file"].endswith("sub-01_task-rest_run-01_space-T1w_desc-brain_mask.nii.gz")
    assert got["native_space"] is True
    assert got["process_args"][0] == [str(b1), str(b2)]


def test_fmriprep_mode_in_mni_space_keeps_native_space_off(tmp_path):
    root = tmp_path / "fmriprep"
    write_run(root / "sub-01" / "func", "sub-01_task-rest_run-01", space="MNI152NLin2009cAsym", res="2")
    got = _run_main(["--fmriprep-dir", str(root), "--participant-label", "01", "--space", "MNI152NLin2009cAsym",
                     "--output-dir", str(tmp_path / "out")])
    assert got["native_space"] is False


def test_fmriprep_mode_refuses_explicit_files(tmp_path):
    root = tmp_path / "fmriprep"
    b = write_run(root / "sub-01" / "func", "sub-01_task-rest_run-01")
    with pytest.raises(SystemExit):
        _run_main(["--fmriprep-dir", str(root), "--participant-label", "01", "--bold-files", str(b),
                   "--output-dir", str(tmp_path / "out")])


@pytest.mark.parametrize("opt,val", [("--participant-label", "01"), ("--session", "1"), ("--task", "rest"),
                                     ("--res", "2"), ("--run", "run-01")])
def test_fmriprep_options_need_fmriprep_dir(tmp_path, opt, val):
    root = tmp_path / "fmriprep"
    b = write_run(root / "sub-01" / "func", "sub-01_task-rest_run-01")
    with pytest.raises(SystemExit):
        _run_main(["--bold-files", str(b), "--mask-file", str(b), opt, val, "--output-dir", str(tmp_path / "out")])


def test_an_input_mode_is_required(tmp_path):
    with pytest.raises(SystemExit):
        _run_main(["--output-dir", str(tmp_path / "out")])
