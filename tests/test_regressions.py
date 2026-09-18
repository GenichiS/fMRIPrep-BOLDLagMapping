"""Regression tests from an independent code review: assertions describe the intended, corrected behaviour.
All generated data is synthetic. The staging test creates a disposable directory containing only its own marker.

`make_mapper` pins the settings of the earlier defaults (global seed, tracking at the TR, range-linked low-pass,
5 s search) so that these tests keep checking the behaviour they were written for; the current defaults are
tested in test_cli_defaults.py.
"""
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bold_lag_mapper import cli, core, io, lag_preprocess as lp, lag_postprocess as post, utils  # noqa: E402

ROOT = Path(core.__file__).resolve().parents[1]


def image_file(path, data, affine=None, tr=2.0, unit='sec'):
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4) if affine is None else affine)
    if img.ndim == 4:
        img.header.set_zooms((*img.header.get_zooms()[:3], tr))
    img.header.set_xyzt_units('mm', unit)
    nib.save(img, str(path))
    return str(path)


def small_inputs(tmp_path, runs=1, mask_affine=None):
    shape = (4, 4, 4)
    t = np.arange(80)
    data = np.broadcast_to(1000 + np.arange(4)[:, None, None, None] * 100
                           + np.sin(t / 5)[None, None, None, :], (*shape, 80)).copy()
    paths = [image_file(tmp_path / f'sub-01_run-{r+1:02d}_bold.nii.gz', data) for r in range(runs)]
    mask = np.zeros(shape)
    mask[1, 1, 1] = 1
    mask_path = image_file(tmp_path / 'mask.nii.gz', mask, mask_affine)
    return paths, mask_path


def make_mapper(tmp_path, bolds, mask, extra=()):
    captured = {}
    def capture(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(process_runs=lambda *a, **k: None)
    argv = ['bold-lag-mapper', '--bold-files', *bolds, '--mask-file', mask,
            '--output-dir', str(tmp_path / 'out'), '--no-validate-bids', '--no-local-staging',
            '--no-apply-motion-correction', '--no-apply-spike-regression',
            '--spatial-fwhm', '0', '--dilate-mask-mm', '0',
            '--seed-roi-file', 'global', '--tracking-step-seconds', 'none', '--bandpass-high', 'linked',
            '--max-lag-seconds', '5.0', *extra]
    with patch.object(sys, 'argv', argv), patch.object(cli, 'BOLDLagMapper', side_effect=capture):
        cli.main()
    return core.BOLDLagMapper(**captured)


def test_R01_staging_preserves_unrelated_files_on_input_failure(tmp_path):
    staging = tmp_path / 'existing_staging'
    staging.mkdir()
    marker = staging / 'unrelated.txt'
    marker.write_text('user content', encoding='utf-8')
    # Verify the exact deletion target is task-created disposable test data.
    assert staging.resolve().is_relative_to(tmp_path.resolve())
    assert list(staging.iterdir()) == [marker]
    m = core.BOLDLagMapper(use_gpu=False, generate_mask=False, local_staging_dir=str(staging))
    with pytest.raises(FileNotFoundError):
        m.process_runs([str(tmp_path / 'absent.nii.gz')], [None], str(tmp_path / 'missing_mask.nii.gz'))
    assert marker.exists(), 'Unrelated file was deleted even though input validation failed before staging'


def test_R02_mask_affine_is_applied_when_shapes_match(tmp_path):
    affine = np.eye(4)
    affine[0, 3] = 1
    bolds, mask = small_inputs(tmp_path, mask_affine=affine)
    m = make_mapper(tmp_path, bolds, mask)
    captured = {}
    class StopAfterRead(Exception):
        pass
    def capture_data(data, confounds):
        captured['data'] = data.copy()
        raise StopAfterRead()
    with patch.object(lp, 'perform_nuisance_regression', side_effect=capture_data):
        with pytest.raises(StopAfterRead):
            m.process_runs(bolds, [None], mask, str(tmp_path / 'out'))
    # Mask index x=1 is world x=2; the BOLD affine is identity.
    np.testing.assert_allclose(captured['data'][0, 0], 1200.0)


def test_R03_confounds_count_checked_without_motion_regression(tmp_path):
    bolds, mask = small_inputs(tmp_path, runs=2)
    conf = tmp_path / 'run1.tsv'
    conf.write_text('framewise_displacement\n' + '0\n' * 80)
    # Confounds still supply NSS and FD when motion regression is disabled.
    try:
        m = make_mapper(tmp_path, bolds, mask, ['--motion-confounds-files', str(conf)])
    except SystemExit as exc:
        assert exc.code != 0
        return
    try:
        m.process_runs(bolds, [str(conf)], mask, str(tmp_path / 'out'))
    except ValueError as exc:
        assert 'confound' in str(exc).lower() or 'run' in str(exc).lower()
        return
    pytest.fail(f'Invalid pairing accepted: {len(bolds)} BOLD runs of 80 frames each, '
                f'but only {m.num_timepoints} frames processed')


def test_R04_txt_motion_standard_path_is_usable(tmp_path):
    path = tmp_path / 'motion.txt'
    np.savetxt(path, np.zeros((8, 6)))
    params, names = io.load_motion_confounds(str(path), 8, True, None)
    augmented, _ = lp.calculate_motion_derivatives_and_fd(params, names)
    assert augmented.shape[0] == 8


def test_R05_an_fd_rule_survives_degenerate_dvars():
    fd = np.zeros(60)
    fd[10] = 1.2
    _, names = lp.identify_and_create_spike_regressors_afyouni_nichols(
        np.full((60, 50), 1000.0), 60, include_preceding_spike=False, fd=fd, fd_threshold=0.5)
    assert 'spike_TR010' in names


def test_R06_nifti_milliseconds_are_converted_to_seconds(tmp_path):
    data = np.ones((2, 2, 2, 10))
    sec = image_file(tmp_path / 'sec.nii.gz', data, tr=2, unit='sec')
    msec = image_file(tmp_path / 'msec.nii.gz', data, tr=2000, unit='msec')
    assert utils.validate_inputs([sec], None, True) == utils.validate_inputs([msec], None, True)


def test_R07_unmeasured_island_not_filled_from_outside_zero():
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, 1, 1] = True
    lag = np.zeros(mask.shape)
    lag[1, 1, 1] = np.nan
    filled = post.fill_holes_1by1(lag, mask)
    assert np.isnan(filled[1, 1, 1])
    final = post.fill_isolated_holes_single_pass(filled, mask)
    assert np.isnan(final[1, 1, 1]), 'The only in-mask voxel has no measured neighbour'


def test_R08_wm_lookup_must_not_return_csf(tmp_path):
    root = tmp_path / 'derivatives' / 'fmriprep'
    anat = root / 'sub-01' / 'anat'
    func = root / 'sub-01' / 'func'
    anat.mkdir(parents=True)
    func.mkdir()
    image_file(anat / 'sub-01_label-CSF_probseg.nii.gz', np.zeros((4, 4, 4)))
    image_file(anat / 'sub-01_desc-preproc_T1w.nii.gz', np.ones((4, 4, 4)))
    bold = image_file(func / 'sub-01_task-rest_space-T1w_desc-preproc_bold.nii.gz',
                      np.ones((4, 4, 4, 10)))
    seed = image_file(tmp_path / 'seed.nii.gz', np.ones((4, 4, 4)))
    m = SimpleNamespace(seed_roi_file=seed, refine_with_wm_mask=True)
    # Identity registration and a local template isolate WM file selection.
    # No real ANTs execution or template download is performed.
    fake_ants = SimpleNamespace(image_read=nib.load, image_write=nib.save,
                                apply_transforms=lambda **kwargs: kwargs['moving'])
    with patch.object(io, '_ANTSPY_AVAILABLE', True), patch.object(io, 'ants', fake_ants), \
         patch.object(io, 'find_transform_files', return_value={'from_mni': 'identity', 'to_mni': 'identity'}), \
         patch.object(io.datasets, 'load_mni152_template', return_value=nib.load(seed)):
        try:
            io.setup_native_space_processing(m, bold, None, str(tmp_path))
        except FileNotFoundError:
            return  # Explicitly rejecting a missing WM map is also a valid fix.
    assert nib.load(m.seed_roi_file).get_fdata().sum() == 64, 'CSF was used as WM and erased the seed'


def test_R14_declared_build_backend_can_be_imported():
    tomllib = pytest.importorskip("tomllib", reason="tomllib needs Python 3.11+")
    data = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    module, _, obj = data['build-system']['build-backend'].partition(':')
    backend = importlib.import_module(module)
    if obj:
        getattr(backend, obj)
