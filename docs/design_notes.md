# Design notes

Implementation choices that are not obvious from the code alone. References to the MATLAB scripts and to
boldlag point to Dr Toshihiko Aso's public repositories.

## Processing order

Per run: removal of leading non-steady-state volumes -> spike detection -> nuisance regression (motion,
derivatives, framewise displacement, spike regressors; OLS with the mean restored) -> percent signal change
(run mean) -> spatial smoothing -> Tukey taper (5 % of the run at each end) -> band-pass. Then the runs are
concatenated, voxels whose peak percent signal change exceeds `--amplitude-threshold` are excluded, the series is
resampled to the tracking step if one is set, each voxel is divided by its standard deviation (no mean
subtraction, as in the MATLAB code), and the seed is the mean of the seed voxels. The lag is estimated once on
the joined series, as `Einsteining_v07.m` does (per-run `fsl_regfilt`, `drMerge4D`, one `drLag4Drev7` call). Lag
estimation is limited by the amount of data, so estimating per run and averaging is not done.

Never pass an already-merged multi-run file as one run: the taper would be computed from the total length and
the run junctions would be unprotected.

## Tracking

- `recursive` replicates `drLag4Drev7` with `FIXED=0`: the data are shifted one step at a time in each
  direction; a voxel is assigned when its correlation peaks at the centre of a ±1-step window with R >= the
  threshold; the seed of the next step is the mean of the centre-peaking voxels. With `--seed-update matlab`
  (default) the seed is formed before already-assigned voxels are excluded, exactly as the MATLAB code orders the
  two statements.
- A direction ends at the first step without seed voxels (the MATLAB/boldlag seed becomes NaN there), so every
  assigned lag bin has a seed and hence a deperfusion regressor.
- `fixed` replicates `FIXED=1`: the lag-0 seed is used at every step.
- `recursive_subtr` / `fixed_subtr` add a 5-point quadratic fit around each accepted peak (with a 3-point
  fallback). The fit must be concave; a convex fit means the sampled correlations do not describe a peak, and
  the voxel keeps the integer step. `recursive_subtr` also carries the seed's mean sub-step offset from step to
  step (seed-phase tracking); without it, voxels found at |p| >= 2 are biased away from zero. Sub-step refinement
  is applied only when the tracking step is >= `--subtr-min-tr` (1.5 s); below it the integer twin runs and the
  output names say so. The integer (3-point, ±1 step) and sub-step (5-point, ±2 steps) acceptance windows are
  different rules, which matters when data with different tracking steps are pooled.

## Tracking step

The original long-TR scripts (`drLag4Drev7_longTR`, boldlag `--reso 1`) resample the filtered, amplitude-gated
series to 1 s before normalisation and tracking. `--tracking-step-seconds auto` does this for TR >= 1.5 s. The
same step is used by every consumer of the seeds: the seed plot, the rainbow plot and deperfusion, where each
seed is shifted on the tracking grid and only then resampled to the TR (`drDeperf_longTR.m`, boldlag
`deperf.py`).

## Search boundary

`drErode_Lag` in `drLag4Drev7.m` discards the outermost lag bin (`abs(Y) >= MaxLag`) and re-fills it. As discussed
in issue #3 of the original repository, with the range-linked low-pass the two boundary bins carry nearly
identical waveforms, so a peak at the end of the correlogram is unreliable. This package identifies the boundary
from the known search range, (steps - 0.5) x step, and does the same by default (`--boundary-null`). Set
`--max-lag-seconds` one step wider than the range of interest. On sub-step maps the threshold is in seconds, so
part of the outer bin can survive.

## Hole filling and masks

Unassigned voxels are filled from the mean of their measured neighbours, only inside the analysis mask:
out-of-mask voxels are 0 in the image, not NaN, and would otherwise drag boundary holes toward 0 s. A voxel with
no measured in-mask neighbour stays NaN. The pre-fill map (NaN = not measured, like `LagOrig.nii`) and a validity
mask are saved with the filled map. NaN does not survive the ANTs warp to MNI, so the validity mask (warped with
nearest neighbour) is the way to tell measured from filled voxels in MNI space. The analysis mask is saved as well;
a lag of exactly 0 s is a measured value, so "in brain" must never be defined as `lag != 0`.

## Spike detection

DVARS as defined in boldlag (`einsteining.dvars_spikes`): the RMS of raw frame-to-frame differences over voxels that
are never zero, in percent of the mean brain signal. A per-voxel percent-signal-change DVARS (`--dvars-definition
psc`) divides by each voxel's own mean, so voxels with a near-zero mean (signal dropout, the edge of the field of
view) can dominate it. DVARS is computed on the undilated brain mask: fMRIPrep's BOLD is not zero outside the
brain, and the dilation ring would otherwise dominate. A frame is also a spike when framewise displacement exceeds
`--fd-spike-threshold`, because a DVARS rule alone can miss moderate motion. The preceding frame is always added.

## Amplitude exclusion

Voxels whose band-passed percent signal change exceeds `--amplitude-threshold` anywhere in the concatenated series
are excluded (large vessels, CSF), as in the MATLAB code (`Y = Y.*(MAX<=4)`). The maximum of a longer series is
larger, so the excluded fraction depends on the number of frames; the count and the frame number are written to
the stats sidecar.

## Numerical precision

The resampled tracking series and the per-shift correlations are float32; everything else is float64. Voxels whose
correlations at two steps are nearly tied can therefore choose a different step than a float64 implementation.
`--save-filtered-bold` writes float64, the dtype the estimator receives, so that a saved series re-tracked
elsewhere reproduces the map.

## Input guards

- The run and confounds lists must have the same length (checked in the CLI and in `process_runs`).
- The mask grid is compared with the BOLD grid (shape and affine) before the BOLD is indexed with it.
- A NIfTI TR stored in milliseconds is converted to seconds.
- Motion columns are identified by name; a confounds file without the requested columns stops the run.
- A failed nuisance regression stops the run instead of continuing with uncorrected data.
- `--local-staging-dir` is treated as a parent: only a per-process child directory is created and removed.
