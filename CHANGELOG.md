# Changelog

## [2.0.0] - 2026-09-18

First public release.

- Derived from the authors' internal version 1.7.0 of the lag mapper.
- Only the MATLAB-family tracking methods are included (`recursive`, `recursive_subtr`, `fixed`,
  `fixed_subtr`); experimental FFT-based and iterative estimators of earlier internal versions were removed.
- New: `--fmriprep-dir` / `--participant-label` input discovery (runs, confounds, mask, TR checks).
- New: bundled deep white-matter seed on fMRIPrep's MNI152NLin2009cAsym 2 mm grid (`--seed-roi-file builtin`,
  the default; `global` for the whole-brain signal).
- New: `--tracking-step-seconds auto` (1 s grid when TR >= 1.5 s) and `--bandpass-high linked`.
- Changed defaults: `--spatial-fwhm 6`, `--bandpass-high 0.09`, `--max-lag-seconds 7`,
  `--tracking-step-seconds auto`, `--seed-roi-file builtin`.
- Native-space processing of fMRIPrep output: the output root is the folder that contains `sub-<label>/` (any
  name), anatomical files are searched in `sub-<label>/ses-<label>/anat` and then `sub-<label>/anat`, and only the
  MNI152NLin2009cAsym <-> T1w transforms are accepted (no substitution of another template's transform).
- sLFO rainbow QC figure: the colour bar reads "lag (s)" with "earlier" / "later" at its ends (the longer label
  was clipped).
- README: example figures from OpenNeuro ds000258, a processing diagram, and an HCP-pipeline example.
