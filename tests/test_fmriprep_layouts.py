"""Native-space support files are found for the common fMRIPrep output layouts, and a transform to another
template is never substituted for the MNI152NLin2009cAsym one."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bold_lag_mapper import io  # noqa: E402

BOLD = "sub-01_task-rest_space-T1w_desc-preproc_bold.nii.gz"
BOLD_SES = "sub-01_ses-1_task-rest_space-T1w_desc-preproc_bold.nii.gz"


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


def _xfms(anat, prefix, template="MNI152NLin2009cAsym"):
    _touch(anat / f"{prefix}_from-T1w_to-{template}_mode-image_xfm.h5")
    _touch(anat / f"{prefix}_from-{template}_to-T1w_mode-image_xfm.h5")


def test_root_is_the_folder_that_contains_the_subject(tmp_path):
    for root in (tmp_path / "fmriprep_out", tmp_path / "bids" / "derivatives", tmp_path / "derivatives" / "fmriprep"):
        bold = root / "sub-01" / "func" / BOLD
        assert io.find_derivatives_root(str(bold)) == root
        bold_ses = root / "sub-01" / "ses-1" / "func" / BOLD_SES
        assert io.find_derivatives_root(str(bold_ses)) == root


def test_hcp_root_is_unchanged():
    bold = "/data/100307/MNINonLinear/Results/rfMRI_REST1_PA/rfMRI_REST1_PA.nii.gz"
    assert io.find_derivatives_root(bold) == Path("/data/100307")


def test_transforms_in_an_output_folder_not_called_derivatives(tmp_path):
    root = tmp_path / "fmriprep_out"
    _xfms(root / "sub-01" / "anat", "sub-01")
    t = io.find_transform_files(str(root / "sub-01" / "func" / BOLD))
    assert t["to_mni"].name.startswith("sub-01_from-T1w_to-MNI152NLin2009cAsym")
    assert t["from_mni"].name.startswith("sub-01_from-MNI152NLin2009cAsym_to-T1w")


def test_multi_session_subject_uses_the_subject_level_anat(tmp_path):
    root = tmp_path / "derivatives"
    _xfms(root / "sub-01" / "anat", "sub-01")
    t = io.find_transform_files(str(root / "sub-01" / "ses-1" / "func" / BOLD_SES))
    assert t["to_mni"].parent == root / "sub-01" / "anat"
    t1 = _touch(root / "sub-01" / "anat" / "sub-01_desc-preproc_T1w.nii.gz")
    assert io.find_anatomical_files(str(root / "sub-01" / "ses-1" / "func" / BOLD_SES), space="T1w") == str(t1)


def test_session_level_anat_is_preferred_when_present(tmp_path):
    root = tmp_path / "derivatives" / "fmriprep"
    _xfms(root / "sub-01" / "ses-1" / "anat", "sub-01_ses-1")
    _xfms(root / "sub-01" / "anat", "sub-01")
    t = io.find_transform_files(str(root / "sub-01" / "ses-1" / "func" / BOLD_SES))
    assert t["to_mni"].parent == root / "sub-01" / "ses-1" / "anat"


def test_a_transform_to_another_template_is_never_substituted(tmp_path):
    # a layout whose root every version resolves correctly, so the test exercises the transform search itself
    root = tmp_path / "derivatives" / "fmriprep"
    anat = root / "sub-01" / "anat"
    _xfms(anat, "sub-01", template="MNI152NLin6Asym")
    _touch(anat / "sub-01_from-fsnative_to-T1w_mode-image_xfm.txt")
    t = io.find_transform_files(str(root / "sub-01" / "func" / BOLD))
    assert "to_mni" not in t and "from_mni" not in t
