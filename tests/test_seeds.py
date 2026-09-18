"""Seed resolution: the bundled deep white-matter seed is used only where its space is known to match."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bold_lag_mapper import seeds  # noqa: E402

MNI = "sub-01_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
T1W = "sub-01_task-rest_space-T1w_desc-preproc_bold.nii.gz"
OTHER = "sub-01_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"


def test_bundled_seed_is_installed_on_the_fmriprep_res2_grid():
    img = nib.load(str(seeds.BUILTIN_SEED_PATH))
    assert img.shape == (97, 115, 97)
    assert int(img.header["sform_code"]) > 0
    assert np.allclose(img.affine, [[2, 0, 0, -96.5], [0, 2, 0, -132.5], [0, 0, 2, -78.5], [0, 0, 0, 1]])
    data = np.asarray(img.dataobj)
    assert set(np.unique(data).tolist()) <= {0, 1}
    assert int(data.sum()) == 52253


def test_builtin_is_used_for_mni2009c_input():
    assert seeds.resolve_seed_roi("builtin", MNI, native_space=False) == str(seeds.BUILTIN_SEED_PATH)


def test_builtin_is_used_for_t1w_input_with_native_space():
    assert seeds.resolve_seed_roi("builtin", T1W, native_space=True) == str(seeds.BUILTIN_SEED_PATH)


@pytest.mark.parametrize("bold", [T1W, OTHER, "rest_bold.nii.gz"])
def test_builtin_is_refused_when_the_space_does_not_match(bold):
    with pytest.raises(ValueError, match="--seed-roi-file"):
        seeds.resolve_seed_roi("builtin", bold, native_space=False)


HCP = "/data/100307/MNINonLinear/Results/rfMRI_REST1_PA/rfMRI_REST1_PA.nii.gz"


@pytest.mark.parametrize("bold", [HCP, MNI, OTHER, "rest_bold.nii.gz"])
def test_builtin_with_native_space_needs_fmriprep_t1w_input(bold):
    """--native-space moves the seed with fMRIPrep's MNI152NLin2009cAsym->T1w transform, so it is valid only for
    fMRIPrep T1w-space runs; HCP data (other template), MNI-space runs or unknown spaces must stop."""
    with pytest.raises(ValueError, match="--seed-roi-file"):
        seeds.resolve_seed_roi("builtin", bold, native_space=True)


@pytest.mark.parametrize("value", [None, "global", "GLOBAL"])
def test_global_means_no_seed_file(value):
    assert seeds.resolve_seed_roi(value, OTHER, native_space=False) is None


def test_an_explicit_path_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        seeds.resolve_seed_roi(str(tmp_path / "missing.nii.gz"), MNI, native_space=False)
    p = tmp_path / "seed.nii.gz"
    p.write_bytes(b"x")
    assert seeds.resolve_seed_roi(str(p), OTHER, native_space=False) == str(p)


def test_freesurfer_seed_takes_precedence():
    assert seeds.resolve_seed_roi("builtin", OTHER, native_space=False, auto_seed_from_freesurfer=True) is None
