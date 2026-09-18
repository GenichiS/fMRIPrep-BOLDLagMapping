"""--save-cleaned-bold writes each run's nuisance-regressed, non-steady-state-trimmed series as
<run>_desc-cleaned_bold.nii.gz (input for external validators such as boldlag lag4d).
End-to-end on a small synthetic run; the same harness checks that the stats / seeds sidecars are written."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bold_lag_mapper import cli, core  # noqa: E402

TR = 2.5


def _synthetic_run(tmp_path, n=40, shape=(6, 6, 6), seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * TR
    slfo = np.sin(2 * np.pi * 0.03 * t)
    data = np.full((*shape, n), 800.0, dtype=np.float32)
    # 3x3x3 block of brain voxels carrying the sLFO at 1% amplitude + noise; the lag map itself is not under test
    data[1:4, 1:4, 1:4, :] = 1000.0 + 10.0 * slfo[None, None, None, :] + rng.normal(0, 1.0, (3, 3, 3, n)).astype(np.float32)
    img = nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))
    img.header.set_zooms((2.0, 2.0, 2.0, TR)); img.header.set_xyzt_units('mm', 'sec')
    bold = tmp_path / 'sub-01_task-rest_run-01_bold.nii.gz'
    nib.save(img, str(bold))
    mask = np.zeros(shape, np.uint8); mask[1:4, 1:4, 1:4] = 1
    mask_p = tmp_path / 'sub-01_mask.nii.gz'
    nib.save(nib.Nifti1Image(mask, img.affine), str(mask_p))
    # confounds: 6 motion columns (tiny noise), FD, and ONE leading non-steady-state volume
    cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    conf = tmp_path / 'sub-01_task-rest_run-01_desc-confounds_timeseries.tsv'
    with open(conf, 'w') as f:
        f.write('\t'.join(cols + ['framewise_displacement', 'non_steady_state_outlier_00']) + '\n')
        for i in range(n):
            vals = rng.normal(0, 1e-3, 6).tolist() + [0.05, 1 if i == 0 else 0]
            f.write('\t'.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in vals) + '\n')
    return str(bold), str(mask_p), str(conf)


def _mapper_from_cli(tmp_path, bold, mask, conf, extra):
    captured = {}
    def capture(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(process_runs=lambda *a, **k: None)
    # The settings of the earlier defaults (global seed, tracking at the TR, range-linked low-pass) are pinned so that
    # the end-to-end tests built on this helper keep checking the behaviour they were written for; `extra` comes
    # last and can override any of them. The current defaults are tested in test_cli_defaults.py.
    argv = ['bold-lag-mapper', '--bold-files', bold, '--mask-file', mask, '--motion-confounds-files', conf,
            '--output-dir', str(tmp_path / 'out'), '--no-validate-bids', '--no-local-staging',
            '--spatial-fwhm', '0', '--dilate-mask-mm', '0', '--tracking-method', 'recursive',
            '--max-lag-seconds', '5.0', '--verbose',
            '--seed-roi-file', 'global', '--tracking-step-seconds', 'none', '--bandpass-high', 'linked', *extra]
    with patch.object(sys, 'argv', argv), patch.object(cli, 'BOLDLagMapper', side_effect=capture):
        cli.main()
    captured['use_gpu'] = False
    return core.BOLDLagMapper(**captured)


def test_cleaned_bold_and_sidecars_are_written(tmp_path):
    bold, mask, conf = _synthetic_run(tmp_path)
    m = _mapper_from_cli(tmp_path, bold, mask, conf, ['--save-cleaned-bold'])
    out = tmp_path / 'out'
    m.process_runs([bold], [conf], mask, str(out))
    cleaned = out / 'sub-01_task-rest_run-01_bold_desc-cleaned_bold.nii.gz'
    assert cleaned.exists(), sorted(p.name for p in out.iterdir())
    ci = nib.load(str(cleaned))
    assert ci.shape[-1] == 39, "one leading non-steady-state volume must have been trimmed"
    assert abs(ci.header.get_zooms()[3] - TR) < 1e-6, "the TR must survive in the header"
    d = ci.get_fdata()
    assert np.all(d[0, 0, 0, :] == 0), "outside the analysis mask the cleaned series is 0"
    assert np.isfinite(d[2, 2, 2, :]).all() and d[2, 2, 2, :].std() > 0
    # stats / seeds sidecars from the same run
    stats = list(out.glob('*_desc-stats.json')); seeds = list(out.glob('*_desc-seeds.npz'))
    raw = list(out.glob('*_desc-raw_lagmap.nii.gz'))
    assert len(stats) == 1 and len(seeds) == 1 and len(raw) == 1, sorted(p.name for p in out.iterdir())
    s = json.load(open(stats[0]))
    assert s['n_runs'] == 1 and s['runs'][0]['n_nss_trimmed'] == 1 and s['runs'][0]['n_frames'] == 39
    assert s['boundary_null'] is True and s['lp_source'] == 'range-linked' and s['taper_width'] == 'fixed5pct'
    assert s['tracking_method_effective'] == 'recursive' and abs(s['tr_track'] - TR) < 1e-9


def test_cleaned_bold_not_written_by_default(tmp_path):
    bold, mask, conf = _synthetic_run(tmp_path, seed=1)
    m = _mapper_from_cli(tmp_path, bold, mask, conf, [])
    out = tmp_path / 'out'
    m.process_runs([bold], [conf], mask, str(out))
    assert not list(out.glob('*_desc-cleaned_bold.nii.gz'))
