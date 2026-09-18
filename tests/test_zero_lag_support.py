"""A lag of exactly 0 s is a MEASURED value (the whole lag-0 band of an integer-step map is exactly 0), so consumers
must never use `lag != 0` as the in-brain proxy. The mapper therefore writes the analysis mask alongside every map."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_save_cleaned_bold import _mapper_from_cli, _synthetic_run  # noqa: E402


def test_analysis_mask_written_by_default(tmp_path):
    bold, mask, conf = _synthetic_run(tmp_path)
    m = _mapper_from_cli(tmp_path, bold, mask, conf, [])
    out = tmp_path / 'out'
    m.process_runs([bold], [conf], mask, str(out))
    hits = list(out.glob('*_desc-analysis_mask.nii.gz'))
    assert len(hits) == 1, sorted(p.name for p in out.iterdir())
    a = nib.load(str(hits[0])).get_fdata()
    assert a.dtype.kind == 'f' and set(np.unique(a)) == {0.0, 1.0}
    assert np.array_equal(a > 0.5, nib.load(mask).get_fdata() > 0.5), "with --dilate-mask-mm 0 the analysis mask is the input mask"
