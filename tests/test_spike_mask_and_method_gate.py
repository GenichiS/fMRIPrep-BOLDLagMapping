"""DVARS spike detection must see BRAIN voxels only, and sub-TR refinement is gated on the tracking step
(recursive_subtr only for a step >= 1.5 s).

Tests import the product functions; nothing here re-implements the logic it checks.
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nilearn.maskers import NiftiMasker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bold_lag_mapper import lag_preprocess  # noqa: E402
from bold_lag_mapper.core import resolve_effective_tracking_method  # noqa: E402


# ---------------------------------------------------------------- TR gate (pure function) ---
@pytest.mark.parametrize("method, tr, min_tr, expected", [
    ("recursive_subtr", 2.5, 1.5, "recursive_subtr"),   # long TR
    ("recursive_subtr", 2.0, 1.5, "recursive_subtr"),   # 2 s TR
    ("recursive_subtr", 0.8, 1.5, "recursive"),         # short TR: integer-step MATLAB replica
    ("recursive_subtr", 1.5, 1.5, "recursive_subtr"),   # boundary is inclusive
    ("recursive_subtr", 0.8, 0.0, "recursive_subtr"),   # gate disabled
    ("recursive_subtr", 0.8, None, "recursive_subtr"),
    ("recursive", 0.8, 1.5, "recursive"),               # other methods untouched
    ("fixed", 0.8, 1.5, "fixed"),
])
def test_subtr_gate(method, tr, min_tr, expected):
    assert resolve_effective_tracking_method(method, tr, min_tr) == expected


def test_subtr_gate_warns_when_it_fires(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="bold_lag_mapper.core"):
        resolve_effective_tracking_method("recursive_subtr", 0.8, 1.5)
    assert any("sub-TR refinement is NOT applied" in m for m in caplog.messages)


# ---------------------------------------------------------------- undilated mask -> masker index ---
def _sphere(shape, radius):
    c = np.array(shape) / 2.0
    g = np.indices(shape).reshape(3, -1).T
    return (np.sqrt(((g - c) ** 2).sum(1)) <= radius).reshape(shape)


def test_mask_to_masker_index_marks_exactly_the_undilated_voxels():
    shape = (20, 20, 20)
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    brain = _sphere(shape, 5.0)
    brain_img = nib.Nifti1Image(brain.astype(np.uint8), aff)
    dilated_img = lag_preprocess.dilate_mask_mm_accurate(brain_img, 4.0)
    dilated = dilated_img.get_fdata() > 0
    assert dilated.sum() > brain.sum(), "dilation must add a ring"

    masker = NiftiMasker(mask_img=dilated_img).fit()
    idx = lag_preprocess.mask_to_masker_index(brain_img, masker)

    assert idx.dtype == bool and idx.shape == (int(dilated.sum()),)
    assert idx.sum() == brain.sum()
    # the True entries are exactly the brain voxels, in the masker's own voxel order
    back = np.zeros(shape, bool)
    back[masker.mask_img_.get_fdata().astype(bool)] = idx
    assert np.array_equal(back, brain)


def test_mask_to_masker_index_identity_when_not_dilated():
    shape = (12, 12, 12)
    brain_img = nib.Nifti1Image(_sphere(shape, 4.0).astype(np.uint8), np.eye(4))
    masker = NiftiMasker(mask_img=brain_img).fit()
    idx = lag_preprocess.mask_to_masker_index(brain_img, masker)
    assert idx.all()


def test_mask_to_masker_index_refuses_empty_overlap():
    shape = (12, 12, 12)
    a = np.zeros(shape, bool); a[:4] = True
    b = np.zeros(shape, bool); b[8:] = True
    masker = NiftiMasker(mask_img=nib.Nifti1Image(a.astype(np.uint8), np.eye(4))).fit()
    with pytest.raises(ValueError):
        lag_preprocess.mask_to_masker_index(nib.Nifti1Image(b.astype(np.uint8), np.eye(4)), masker)


# ---------------------------------------------------------------- the failure the fix addresses ---
def test_ring_noise_hides_a_real_motion_spike_from_the_detector():
    """Brain voxels carry one genuine spike at t=50; a small 'ring' of near-zero-mean voxels
    carries only noise. On brain-only input the detector flags t=50 (and its predecessor); on
    ring-included input the ring's enormous percent-signal-change swamps DVARS and the genuine
    spike is no longer the frame that stands out."""
    rng = np.random.default_rng(3)
    T, n_brain, n_ring = 120, 400, 40
    brain = 1000.0 + rng.standard_normal((T, n_brain)) * 5.0
    brain[50] += 40.0                                   # a real global spike (8 sigma)
    ring = 2.0 + rng.standard_normal((T, n_ring)) * 2.0  # mean ~2, noise ~2 -> PSC ~100%

    reg_b, names_b = lag_preprocess.identify_and_create_spike_regressors(brain, T, 3.0, True)
    assert reg_b is not None and set(names_b) >= {"spike_TR050", "spike_TR049"}, names_b

    reg_f, names_f = lag_preprocess.identify_and_create_spike_regressors(np.hstack([brain, ring]), T, 3.0, True)
    flagged_f = set(names_f) if reg_f is not None else set()
    assert "spike_TR050" not in flagged_f or len(flagged_f) > len(names_b), (
        "with the ring included the detector should no longer isolate the genuine spike")
