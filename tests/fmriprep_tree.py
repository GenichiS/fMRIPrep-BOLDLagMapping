"""Tiny fake fMRIPrep derivatives for tests: 2x2x2 volumes with the real file-naming scheme."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def write_run(func_dir, stem, space="T1w", n=130, tr=2.5, json_tr=None, mask=True, conf=True, res=None):
    func_dir = Path(func_dir)
    func_dir.mkdir(parents=True, exist_ok=True)
    sp = f"space-{space}" + (f"_res-{res}" if res else "")
    img = nib.Nifti1Image(np.zeros((2, 2, 2, n), np.float32), np.eye(4))
    img.header.set_zooms((2.0, 2.0, 2.0, tr))
    img.header.set_xyzt_units("mm", "sec")
    bold = func_dir / f"{stem}_{sp}_desc-preproc_bold.nii.gz"
    nib.save(img, str(bold))
    (func_dir / f"{stem}_{sp}_desc-preproc_bold.json").write_text(
        json.dumps({"RepetitionTime": tr if json_tr is None else json_tr}))
    if mask:
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2), np.uint8), np.eye(4)),
                 str(func_dir / f"{stem}_{sp}_desc-brain_mask.nii.gz"))
    if conf:
        (func_dir / f"{stem}_desc-confounds_timeseries.tsv").write_text("trans_x\n" + "0\n" * n)
    return bold
