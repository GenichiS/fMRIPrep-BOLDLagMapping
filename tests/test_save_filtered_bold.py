"""--save-filtered-bold writes the concatenated band-pass-filtered
percent-signal series exactly as the estimator receives it (before gate / resample / std-norm)."""
import sys
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiMasker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bold_lag_mapper import core, lag_estimators_matlab  # noqa: E402
from test_save_cleaned_bold import _mapper_from_cli, _synthetic_run, TR  # noqa: E402


def test_filtered_bold_equals_estimator_input(tmp_path):
    bold, mask, conf = _synthetic_run(tmp_path)
    # amplitude gate OFF so the estimator input is exactly filtered / std
    m = _mapper_from_cli(tmp_path, bold, mask, conf, ['--save-filtered-bold', '--amplitude-threshold', '0'])
    out = tmp_path / 'out'
    seen = {}
    orig = lag_estimators_matlab.compute_lag_maps_recursive

    def spy(mapper, ts, seed):
        seen['ts'] = np.asarray(ts).copy()
        return orig(mapper, ts, seed)

    with patch.object(core.lag_estimators_matlab, 'compute_lag_maps_recursive', side_effect=spy):
        m.process_runs([bold], [conf], mask, str(out))
    hits = list(out.glob('*_desc-filtered_bold.nii.gz'))
    assert len(hits) == 1, sorted(p.name for p in out.iterdir())
    fi = nib.load(str(hits[0]))
    assert fi.shape[-1] == 39, "one leading non-steady-state volume must have been trimmed"
    assert abs(fi.header.get_zooms()[3] - TR) < 1e-6, "the TR must survive in the header"
    assert fi.get_data_dtype() == np.float64, "the series is saved in float64 (what the estimator receives)"
    d = np.asarray(fi.dataobj)
    assert d.dtype == np.float64
    assert np.all(d[0, 0, 0, :] == 0), "outside the analysis mask the series is 0"
    m3 = np.asarray(nib.load(mask).dataobj) > 0.5
    F = np.ascontiguousarray(d[m3].T)                                          # (T, V) in-mask, C order, float64
    sd = F.std(axis=0); sd[sd == 0] = 1
    assert seen['ts'].shape == F.shape
    assert np.array_equal(F / sd, np.asarray(seen['ts'], np.float64)), "in-mask matrix / std must be EXACTLY what the estimator received"


def test_filtered_bold_not_written_by_default(tmp_path):
    bold, mask, conf = _synthetic_run(tmp_path, seed=1)
    m = _mapper_from_cli(tmp_path, bold, mask, conf, [])
    out = tmp_path / 'out'
    m.process_runs([bold], [conf], mask, str(out))
    assert not list(out.glob('*_desc-filtered_bold.nii.gz'))
