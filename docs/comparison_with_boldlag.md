# Differences from boldlag

[boldlag](https://github.com/RIKEN-BCIL/HCPstyle-BOLDLagMappingAndCleaning) is Dr Toshihiko Aso's official Python
port of his MATLAB scripts (`drLag4Drev7`, `drMerge4D`, `drDeperf`, `Einsteining`). It reproduces the MATLAB lag
values exactly (see its `tests/validate_matlab.py`) and is organised around the HCP directory layout. This package
implements the same tracking algorithm, but is organised around fMRIPrep derivatives and differs in several
processing steps. The table compares boldlag 0.2.0 (commit `09207f4`; later commits on its `main` branch change
only the README) with this package 2.0.0. Function and file names on both sides point to the code.

Because of the differences below, the two packages do not produce identical maps from the same data, even with
matching options.

| Aspect | boldlag 0.2.0 | this package 2.0.0 |
|---|---|---|
| **Input layout** | HCP: `<subject>/MNINonLinear/Results/<run>/<run>.nii.gz` with `Movement_Regressors.txt` and `*SBRef.nii.gz` next to each run (`einsteining.find_runs`, `einsteining.scrub_run`, `einsteining.einsteining`). `lag4d` alone accepts any 4D file. | fMRIPrep: `--fmriprep-dir` + `--participant-label` (`fmriprep.discover_fmriprep_inputs`), or explicit `--bold-files` / `--motion-confounds-files` / `--mask-file` from any pipeline. |
| **Motion regressors** | 6 parameters + backward/forward differences + FD from `Movement_Regressors.txt` (`einsteining.framewise_displacement`, `scrub_run`). | The same set, read by column name from fMRIPrep's confounds TSV (`io.load_motion_confounds`, `lag_preprocess.calculate_motion_derivatives_and_fd`); headerless `.txt` files need a declared `--txt-motion-schema`. |
| **Non-steady-state volumes** | Not handled. | Leading `non_steady_state_outlier_XX` volumes removed from BOLD and confounds of each run (`io.count_leading_non_steady_state`). |
| **Spike detection** | DVARS of the raw intensity over never-zero brain voxels, in % of the mean brain signal; a spike when DVARS > 1.5 x median, plus the preceding volume (`einsteining.dvars_spikes`). | The same DVARS definition (`lag_preprocess.compute_dvars_boldlag`), computed on the undilated brain mask because fMRIPrep's BOLD is not zero outside the brain; default rule robust-z (k = 3) OR framewise displacement > 0.5 mm from fMRIPrep's `framewise_displacement`, plus the preceding volume. `--spike-method aso-median --fd-spike-threshold 0` applies boldlag's rule. |
| **Resolution** | 2x down-sampling (`subsamp2offc`) by default in the pipeline. | No down-sampling: the input grid is used. |
| **Brain mask** | `fslmaths -thrp 10` of the temporal mean (`filters.thrp` in `lag4d.prepare`). | The pipeline's brain mask (fMRIPrep `desc-brain_mask`), dilated by `--dilate-mask-mm` (4 mm). |
| **Order across runs** | Per run: nuisance regression -> high-pass -> 5 % linear end taper; concatenate and add the mean of the run means (`merge4d.merge4d`); then percent signal change of the joined series -> smoothing -> band-pass (`lag4d.prepare`). | Per run: nuisance regression -> percent signal change (run mean) -> smoothing -> 5 % Tukey taper -> band-pass; then concatenate (`core.BOLDLagMapper.process_runs`). The lag is estimated once on the joined series in both. |
| **Spatial smoothing** | SPM's integrated-Gaussian kernel with an implicit NaN mask (`spm.smooth`), FWHM 8 mm in the pipeline. | Gaussian filter, zero outside the image (`lag_preprocess.smooth_spm_compat`), FWHM 6 mm by default. |
| **Temporal filter** | FSL `-bptf` re-implementation: high-pass by local linear fit (half-width int(3 sigma)) with the mean removed; low-pass Gaussian (half-width int(20 sigma)+2) renormalised at the edges (`filters.bptf`). | High-pass by local linear fit (half-width ceil(3 sigma)); low-pass Gaussian (half-width ceil(3 sigma)), mirror boundary (`lag_preprocess.temporal_filter_fsl_replicated`). |
| **Low-pass cut-off** | 0.9 / (2 x PosiMax) Hz, linked to the search range (`lag4d.prepare`). | 0.09 Hz by default; `--bandpass-high linked` gives the range-linked value. |
| **Seed** | `hcp`: a bundled cerebral mask for HCP data on the 2 mm down-sampled grid, trilinear weights > 0.1 (`lag4d.HCP_SEED_MASK`). | A bundled deep white-matter mask on fMRIPrep's MNI152NLin2009cAsym 2 mm grid, nearest-neighbour, >= 0.1 (`seeds.py`); moved to T1w space with fMRIPrep's transform when processing natively. |
| **Tracking method** | Pipeline default `FIXED=1` (fixed seed); threshold 0.2 in the pipeline, 0.3 in `lag4d` alone. | Default `recursive` (`FIXED=0`); threshold 0.2. `--tracking-method fixed` gives the fixed seed. |
| **Long TR** | `--reso 1` resamples the filtered series to a 1 s step (`lag4d.lag4d`). | `--tracking-step-seconds auto` does the same when TR >= 1.5 s (`core.resolve_tracking_step`, `lag_preprocess.resample_for_tracking`). |
| **Sub-step lags** | Integer steps only. | Optional `recursive_subtr` / `fixed_subtr`: 5-point quadratic refinement with seed-phase tracking, applied only when the tracking step is >= 1.5 s (`lag_estimators_matlab`). |
| **Correlation map** | `MaxR`: maximum correlation over all steps (`lag4d.track`). | `corrmap`: correlation at the assigned step. |
| **Search-boundary nulling** | Voxels with abs(lag) >= the largest lag present in the map (`nanmax`) are discarded and re-filled (`lag4d.erode_lag`). | Voxels with abs(lag) >= (steps - 0.5) x step of the known search range are discarded and re-filled (`lag_postprocess.null_search_boundary`); identical for integer maps that reach the boundary. |
| **Hole filling** | Every NaN in the field of view is filled iteratively from its 6 neighbours (circular), then the brain mask is applied (`lag4d.erode_lag`). | Filled from in-mask neighbours only; a voxel without a measured in-mask neighbour stays NaN (`lag_postprocess.fill_holes_1by1`, `fill_isolated_holes_single_pass`). |
| **Processing space** | The input space; the lag map is resliced onto the SBRef grid for deperfusion (`einsteining.reslice_lagmap`). | The input space, or the subject's T1w space with maps warped to MNI152NLin2009cAsym by fMRIPrep's `.h5` transforms (`--native-space`, `io.setup_native_space_processing`, `io.warp_to_mni_and_save`; ANTsPy). |
| **Outputs** | `LagOrig.nii`, `LagMap.nii`, `e1LagOrig.nii`, `MaxR.nii`, `Seeds.mat/.npy`, `RegionMean`, `params.json` in `Lag_{fix,rec}_...` folders. | BIDS-style names: raw, pre-fill and filled lag maps, correlation map, validity and analysis masks, `_desc-stats.json`, `_desc-seeds.npz` (see the README). |
| **Deperfusion** | Part of the pipeline, per run (`deperf.deperf`). | Optional (`--save-deperfusioned-bold`); seeds are shifted on the tracking grid and then resampled to the TR, in the same order as `deperf.deperf`. |
| **Front ends** | Command line, desktop GUI (`boldlag-gui`), browser app (`boldlag-web`). | Command line and Python API. |
| **Dependencies** | numpy, scipy, nibabel. | numpy, scipy, nibabel, nilearn, pandas, matplotlib, tqdm; optional antspyx (native space) and cupy (GPU). |
| **Numerical precision** | float64 throughout. | float64 except the resampled tracking series and the per-shift correlations (float32); near-tie voxels can therefore select a different step than a float64 implementation. |
