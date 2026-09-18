"""
Command-Line Interface (CLI) Module
-----------------------------------

This module provides the command-line interface for the BOLD lag mapping pipeline.
It uses Python's standard `argparse` library to define and parse all user-provided
arguments, such as input files, output directories, and algorithmic parameters.

The `main` function defined here is the primary entry point for running the
pipeline from the command line. It handles argument validation, sets up logging,
instantiates the main `BOLDLagMapper` class, and initiates the processing.
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

# Import the main class from the core module and version info
from .core import BOLDLagMapper
from .io import BIDSPathParser
from .fmriprep import discover_fmriprep_inputs
from . import __version__

logger = logging.getLogger(__name__)


def validate_bids_inputs(args):
    """
    Validate BIDS compliance and detect pipeline types.
    """
    logger.info("Validating BIDS compliance and detecting pipeline types...")

    pipeline_types = set()
    subjects = set()

    for bold_file in args.bold_files:
        if not os.path.exists(bold_file):
            raise FileNotFoundError(f"BOLD file not found: {bold_file}")

        try:
            parser = BIDSPathParser(bold_file)
            pipeline_types.add(parser.pipeline_type)

            subject_id = parser.get_subject_id()
            if subject_id:
                subjects.add(subject_id)
            else:
                logger.warning(f"Could not extract subject ID from: {bold_file}")

            logger.info(f"File: {Path(bold_file).name}")
            logger.info(f"  Pipeline: {parser.pipeline_type}")
            logger.info(f"  Subject: {subject_id}")
            logger.info(f"  Entities: {parser.entities}")

        except Exception as e:
            logger.warning(f"BIDS parsing failed for {bold_file}: {e}")

    # Validate consistency
    if len(pipeline_types) > 1:
        logger.warning(f"Mixed pipeline types detected: {pipeline_types}")
        logger.warning("This may cause issues with native space processing.")

    if len(subjects) > 1:
        logger.warning(f"Multiple subjects detected: {subjects}")
        logger.warning("Consider processing subjects separately for optimal results.")

    primary_pipeline = list(pipeline_types)[0] if pipeline_types else 'unknown'
    logger.info(f"Detected pipeline: {primary_pipeline}")
    if args.native_space:
        if primary_pipeline == 'hcp':
            logger.info("  Native space processing: transforms are searched in MNINonLinear/xfms/")
        elif primary_pipeline == 'fmriprep':
            logger.info("  Native space processing: .h5 transforms are searched in the subject's anat/")

    return primary_pipeline, subjects


def _lp_cutoff(value):
    """--bandpass-high: a frequency in Hz, or 'linked' for the range-linked cut-off 0.9 / (2 * max lag)."""
    if str(value).strip().lower() == "linked":
        return None
    f = float(value)
    if f <= 0:
        raise argparse.ArgumentTypeError("--bandpass-high must be a positive frequency in Hz, or 'linked'")
    return f


def _tracking_step(value):
    """--tracking-step-seconds: 'auto', 'none', or a positive number of seconds."""
    v = str(value).strip().lower()
    if v in ("auto", "none"):
        return v
    try:
        f = float(v)
    except ValueError:
        raise argparse.ArgumentTypeError("--tracking-step-seconds must be 'auto', 'none' or a number of seconds")
    if f <= 0:
        raise argparse.ArgumentTypeError("--tracking-step-seconds must be positive")
    return f


def build_parser():
    """Build the argument parser. Importable (tests assert defaults on it); main() parses sys.argv with it."""
    examples = """
    --- Examples ---

    1. fMRIPrep subject, default settings (T1w-space processing, MNI outputs):
       bold-lag-mapper --fmriprep-dir /data/derivatives/fmriprep --participant-label 01 \\
           --output-dir /data/derivatives/lagmap/sub-01

    2. fMRIPrep MNI-space runs (no transforms needed):
       bold-lag-mapper --fmriprep-dir /data/derivatives/fmriprep --participant-label 01 \\
           --space MNI152NLin2009cAsym --output-dir /data/derivatives/lagmap/sub-01

    3. Explicit files, global-signal seed (any pipeline):
       bold-lag-mapper --bold-files run1.nii.gz run2.nii.gz --motion-confounds-files run1.tsv run2.tsv \\
           --mask-file brain_mask.nii.gz --seed-roi-file global --output-dir out/

    4. Settings of the original long-TR scripts (range-linked low-pass, fixed seed):
       bold-lag-mapper ... --bandpass-high linked --tracking-method fixed
    """
    parser = argparse.ArgumentParser(
        prog="bold-lag-mapper",
        description="BOLD lag mapping (recursive lag tracking after Aso et al.) for fMRIPrep derivatives, "
                    "HCP outputs or explicitly listed files.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- Version Argument ---
    parser.add_argument(
        '-v', '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    # --- Input/Output Arguments ---
    parser.add_argument("--bold-files", nargs="+", type=str, default=None,
                        help="Path(s) to BOLD NIfTI files, one per run (fMRIPrep, HCP or any 4D NIfTI). "
                             "Not needed with --fmriprep-dir.")
    parser.add_argument("--txt-motion-schema", type=str, default="spm",
                        choices=["spm", "fsl", "hcp", "afni"],
                        help="Column convention for HEADERLESS .txt motion files. A .txt file carries no column "
                             "names, so the order and the rotation units must be declared: spm = translations(mm) "
                             "then rotations(rad) [rp_*.txt]; fsl = rotations(rad) then translations(mm) [MCFLIRT "
                             ".par]; hcp = translations(mm) then rotations(deg) [Movement_Regressors.txt]; afni = "
                             "rotations(deg) then translations(mm) [3dvolreg .1D]. Degrees are converted to "
                             "radians. Ignored for fMRIPrep .tsv input, which has names.")
    parser.add_argument("--motion-confounds-files", nargs="+", type=str, default=[],
                        help="Path(s) to motion confounds files (.tsv or .txt), one per run in the order of "
                             "--bold-files.")
    parser.add_argument("--mask-file", type=str, default=None,
                        help="Path to a brain mask. Required unless --generate-mask or --fmriprep-dir is used.")
    parser.add_argument("--output-dir", type=str, default="./lag_mapping_output",
                        help="Directory for all output files.")

    # --- fMRIPrep input ---
    fp = parser.add_argument_group(
        "fMRIPrep input", "Alternative to --bold-files / --mask-file / --motion-confounds-files: find one subject's "
        "runs, confounds and mask in an fMRIPrep derivatives directory.")
    fp.add_argument("--fmriprep-dir", default=None,
                    help="fMRIPrep derivatives directory (the one that contains sub-<label>/).")
    fp.add_argument("--participant-label", default=None, help="Subject label, with or without the 'sub-' prefix.")
    fp.add_argument("--session", default=None, help="Session label; required when the subject has several sessions.")
    fp.add_argument("--task", default=None, help="Task label; required when the subject has several tasks.")
    fp.add_argument("--space", default="T1w", choices=["T1w", "MNI152NLin2009cAsym"],
                    help="Space of the BOLD runs to use. T1w (default) switches --native-space on: processing in "
                         "the subject's space, outputs warped to MNI152NLin2009cAsym with fMRIPrep's transforms.")
    fp.add_argument("--res", default=None, help="Resolution label of MNI-space runs (e.g. 2) when several exist.")
    fp.add_argument("--run", dest="runs", nargs="+", default=None,
                    help="Keep only runs whose file name contains _<fragment>_, e.g. run-01 or dir-AP_run-02.")
    fp.add_argument("--min-volumes", type=int, default=120,
                    help="Drop runs with fewer volumes than this (default 120).")

    # --- BIDS and Pipeline Options ---
    parser.add_argument("--native-space", action="store_true",
                        help="Process in the subject's space and write MNI-space outputs (requires ANTsPy and the "
                             "transform files of fMRIPrep or HCP). Set automatically by --fmriprep-dir --space T1w.")
    parser.add_argument("--no-validate-bids", dest="validate_bids", action="store_false", default=True,
                        help="Skip the BIDS parsing and pipeline-detection log (enabled by default).")

    # --- Core Algorithm Parameters ---
    parser.add_argument("--tracking-method", type=str, default="recursive",
                        choices=["recursive", "recursive_subtr", "fixed", "fixed_subtr"],
                        help="Lag tracking method, after drLag4Drev7: recursive (FIXED=0, the seed is re-formed "
                             "at every step; default), fixed (FIXED=1, one seed), recursive_subtr (recursive + "
                             "5-point sub-step refinement + seed-phase tracking), fixed_subtr (fixed seed + "
                             "5-point sub-step refinement). Below --subtr-min-tr the *_subtr methods run as their "
                             "integer twins and the outputs are named after the method actually run.")
    parser.add_argument("--max-lag-seconds", type=float, default=7.0,
                        help="Search range in seconds (default 7.0). The search is set in whole tracking steps, "
                             "round(max_lag_seconds / step), and the effective range is logged. With "
                             "--boundary-null (default) the outermost step is discarded and re-filled, so the "
                             "kept range is one step narrower; set this one step wider than the range of "
                             "interest. Sub-step refinement can add up to half a step to saved lags.")
    parser.add_argument("--min-corr-threshold", type=float, default=0.2, help="Minimum correlation (R-value) for a voxel to be assigned a lag.")
    parser.add_argument("--spatial-fwhm", type=float, default=6.0, help="FWHM in mm for spatial smoothing of BOLD data (default 6.0).")
    parser.add_argument("--seed-roi-file", type=str, default="builtin",
                        help="Seed for the initial reference signal: 'builtin' (default) = the bundled deep "
                             "white-matter mask in MNI152NLin2009cAsym, usable for input in that space or with "
                             "--native-space; a path to a mask (in MNI152NLin2009cAsym with --native-space, else "
                             "in the input's space); or 'global' for the mean of all voxels in the mask. "
                             "Overridden by --auto-seed-from-freesurfer.")
    parser.add_argument('--auto-seed-from-freesurfer', action='store_true', help="Automatically find FreeSurfer outputs based on BIDS layout and create a cerebral GM+WM seed mask (optional, not the default).")
    parser.add_argument("--amplitude-threshold", type=float, default=4.0, help="Peak percent-signal-change amplitude (after band-pass) above which a voxel is excluded before lag calculation, to drop high-amplitude voxels such as large vessels / CSF (e.g. 4 == 4%% PSC). Applied before std-normalisation. Note that this is not independent of run length: max|x| grows with the number of samples, and it is applied to the CONCATENATED series. The exclusion fraction and the frame count are logged and written to the stats sidecar. Set 0 to disable.")
    parser.add_argument("--subtr-min-tr", type=float, default=1.5, help="Sub-step refinement ('recursive_subtr' / 'fixed_subtr') is applied only when the TRACKING STEP (the TR, or --tracking-step-seconds) is >= this many seconds (default 1.5). Below it the integer-step twin ('recursive' / 'fixed', MATLAB FIXED=0/1) is run instead and the outputs are named after it. Also the TR limit for --tracking-step-seconds auto. Set to 0 to always apply sub-step refinement.")
    parser.add_argument("--tracking-step-seconds", type=_tracking_step, default="auto", help="Step of the lag grid. 'auto' (default): resample the filtered series to 1 s when the TR is >= --subtr-min-tr (the approach of the original's long-TR scripts, drLag4Drev7_longTR / boldlag --reso 1; scipy resample_poly, Kaiser window) and track at the TR otherwise. 'none': always track at the TR. A number: resample to that step. Output names carry _step<value> when a step is set.")
    parser.add_argument("--subtr-phase-tracking", action=argparse.BooleanOptionalAction, default=True, help="recursive_subtr only. Carry the recursive seed's sub-step phase into the assigned lag (default ON). Without it every voxel found at |p|>=2 is biased away from zero, because the seed formed at step p-1 inherits the mean sub-step offset of its voxels while the integer shift assumes it sits at exactly -(p-1) steps.")
    parser.add_argument("--seed-update", type=str, choices=["newly_found", "matlab"], default="matlab", help="recursive / recursive_subtr: which voxels form the next seed. 'matlab' (default) replicates drLag4Drev7.m literally (SeedD = mean(Downward(:,I==2)) is taken BEFORE already-assigned voxels are masked out), so every voxel whose local peak sits at the centre with R >= threshold enters the seed, including voxels assigned earlier. 'newly_found' averages only the voxels assigned at the current step.")
    parser.add_argument("--boundary-null", action=argparse.BooleanOptionalAction, default=True, help="Default ON. Null and re-interpolate voxels whose lag saturated at the search boundary (|lag| >= (max_lag_steps-0.5)*step, a SECONDS threshold applied to the sub-step value; integer maps lose exactly the outer bin) before hole-filling. This is the original's behaviour: drLag4Drev7.m drErode_Lag nulls abs(Y)>=MaxLag deliberately, because with the range-linked low-pass the two boundary bins have near-identical waveforms (issue #3 of the original repository). Set --max-lag-seconds one step wider than the range of interest.")

    # --- Filtering Parameters ---
    parser.add_argument("--bandpass-low", type=float, default=None, help="High-pass cut-off in Hz (default: 0.008 Hz).")
    parser.add_argument("--bandpass-high", type=_lp_cutoff, default=0.09, help="Low-pass cut-off in Hz (default 0.09). 'linked' uses the original's range-linked cut-off 0.9 / (2 * --max-lag-seconds). The value used and its source are written to the stats sidecar.")
    parser.add_argument("--trim-non-steady-state", action=argparse.BooleanOptionalAction, default=True, help="Drop the LEADING volumes that fMRIPrep flagged as non-steady-state (confounds columns non_steady_state_outlier_XX) from both the BOLD and the confounds of each run, before any processing (default ON). Only the leading contiguous block is dropped; flags elsewhere are reported and kept. No effect on .txt confounds or when the columns are absent. The number trimmed is logged per run.")
    parser.add_argument("--skip-tukey-window", action="store_true", help="Skip the per-run taper. Note that the original pipeline tapers each run (drMerge4D, linear 5%% ramp per end); skipping only reproduces drLag4Drev7 run on a single un-tapered file.")
    parser.add_argument("--taper-width", type=str, choices=["fixed5pct", "maxlag"], default="fixed5pct", help="Width of the per-run Tukey taper. fixed5pct (default): 5%% of the run at each end (alpha 0.1), the width of drMerge4D's linear ramp. maxlag: max_lag_trs samples per end capped at 5%%; kept to reproduce outputs of earlier versions.")
    parser.add_argument("--skip-bandpass-filter", action="store_true", help="Skip bandpass filtering entirely. Use only when input data is already filtered (e.g., MATLAB preprocessed data).")
    parser.add_argument("--lp-filter-mode", type=str, default="reflect", choices=["wrap", "nearest", "constant", "reflect"], help="Boundary mode for LP filter convolution. 'reflect'=mirror (default), 'nearest'=edge padding, 'constant'=zero padding, 'wrap'=circular. NOTE: 'wrap' is non-physical for BOLD time series (the run end is convolved onto its start) and is not how FSL's -bptf behaves; retained only for backward comparison.")

    # --- Masking Parameters ---
    parser.add_argument("--generate-mask", action="store_true", help="Generate a brain mask from the mean of the first BOLD run.")
    parser.add_argument("--mask-threshold", type=float, default=10.0, help="Intensity threshold for --generate-mask.")
    parser.add_argument("--dilate-mask-mm", type=float, default=4.0, help="Distance in mm to dilate the brain mask.")
    parser.add_argument("--refine-with-wm-mask", action="store_true", help="Refine a warped seed with the subject's white matter probability segmentation (for native space processing).")

    # --- Nuisance Correction Parameters ---
    parser.add_argument('--apply-motion-correction', action=argparse.BooleanOptionalAction, default=True, help="Regress out motion parameters (default: enabled). Use --no-apply-motion-correction to disable.")
    parser.add_argument("--custom-motion-regressors", nargs="+", type=str, default=None, help="Specific column names from a .tsv for motion regression.")
    parser.add_argument("--apply-spike-regression", dest="apply_spike_regression", action=argparse.BooleanOptionalAction, default=True, help="Enable spike regression (scrubbing). Enabled by default; use --no-apply-spike-regression to disable.")
    parser.add_argument("--dvars-threshold", type=float, default=3.0, help="Robust spike-detection cutoff: a frame is flagged when its DVARS exceeds median + (this many) robust SDs (1.4826*MAD), i.e. a modified z-score. Default 3.0; the classic modified z-score outlier cutoff is 3.5. Lower = more frames flagged, higher = fewer.")
    parser.add_argument("--fd-spike-threshold", type=float, default=0.5, help="A frame is also a spike when its framewise displacement exceeds this many mm (default 0.5; Power et al. 2012). FD is fMRIPrep's framewise_displacement column (trimmed like the BOLD), else the FD computed from the motion parameters. Set 0 to use the DVARS rule alone.")
    parser.add_argument("--include-preceding-spike", action=argparse.BooleanOptionalAction, default=True, help="Include the volume preceding a spike as part of scrubbing (default: enabled). Use --no-include-preceding-spike to disable.")
    parser.add_argument("--spike-method", type=str, choices=["robustz", "afyouni-nichols", "aso-median"], default="robustz", help="Spike-detection rule for --apply-spike-regression. 'robustz' (default): flag frames with DVARS above median + (--dvars-threshold) robust SDs. 'afyouni-nichols': statistical DVARS test (Afyouni & Nichols 2018), flagging frames that are BOTH statistically significant (p < --an-alpha/(T-1), Bonferroni across frames) AND practically significant (Delta-percent-D-var > --an-dpd-threshold). 'aso-median': the rule of the original Einsteining scripts, DVARS > 1.5 x median (boldlag dvars_spikes); with --dvars-definition boldlag and --fd-spike-threshold 0 it is the original rule. All methods use the same preceding+spike one-hot window.")
    parser.add_argument("--dvars-definition", type=str, choices=["psc", "boldlag"], default="boldlag", help="Which DVARS series the spike rule looks at. 'boldlag' (default): Dr Aso's definition, the RMS of RAW intensity differences over never-zero brain voxels in percent of the mean brain signal. 'psc': RMS of per-voxel percent-signal-change differences, which voxels with a near-zero mean (signal dropout, field-of-view edge) can dominate. The sidecar's spikes_aso_1p5median is always computed on the 'boldlag' series.")
    parser.add_argument("--an-alpha", type=float, default=0.05, help="Afyouni-Nichols spike test: family-wise significance level, Bonferroni-corrected across frames (default 0.05). Only used when --spike-method afyouni-nichols.")
    parser.add_argument("--an-dpd-threshold", type=float, default=5.0, help="Afyouni-Nichols spike test: practical-significance floor on Delta-percent-D-var (default 5.0, the value recommended by Afyouni & Nichols 2018). Only used when --spike-method afyouni-nichols.")

    # --- Performance and Saving Parameters ---
    parser.add_argument("--use-gpu", action="store_true", help="Use a CUDA-enabled GPU (if CuPy is installed) for accelerated computations.")
    parser.add_argument("--gpu-batch-size", type=int, default=10000, help="Voxel batch size for GPU processing to manage memory.")
    parser.add_argument("--save-debugging-files", action="store_true", help="Save intermediate files for debugging (e.g., masks, initial seeds).")
    parser.add_argument("--save-screenshot", action="store_true", help="Save a screenshot of the final (MNI-space) lag map overlaid on a T1w image.")
    parser.add_argument("--save-carpet-map", action="store_true", help="Save carpet maps of nuisance regressors and BOLD data, the seed signal plot and the sLFO rainbow plot for quality control.")
    parser.add_argument("--save-cleaned-bold", action="store_true", help="Write each run's nuisance-regressed (motion + spikes), non-steady-state-trimmed series as <run>_desc-cleaned_bold.nii.gz in the output directory (zeros outside the analysis mask, TR kept in the header), e.g. as input for boldlag lag4d.")
    parser.add_argument("--save-filtered-bold", action="store_true", help="Write the concatenated, band-pass-filtered percent-signal series exactly as the estimator receives it (after PSC, smoothing, taper, HP+LP and run concatenation; BEFORE the peak-PSC gate, resampling and std-normalisation) as <prefix>_desc-filtered_bold.nii.gz (float64, zeros outside the analysis mask, TR in the header).")
    parser.add_argument('--save-deperfusioned-bold', type=str,
                        choices=['none', 'raw', 'cleaned', 'both'],
                        default='none',
                        help="Which deperfusioned BOLD files to save. 'none': no files (default), "
                             "'raw': deperfusioned raw data, 'cleaned': deperfusioned cleaned data, "
                             "'both': save both raw and cleaned versions.")
    parser.add_argument("--screenshot-z-coord", type=int, default=15, help="Z-coordinate (in mm) for the screenshot.")
    parser.add_argument("--screenshot-alpha", type=float, default=0.7, help="Alpha (transparency) for the lag map overlay (0.0-1.0).")
    parser.add_argument("--screenshot-threshold", type=float, default=0.5, help="Threshold for lag map overlay, to make near-zero values transparent.")
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True, help="Show progress bars and verbose output (default: enabled).")
    parser.add_argument("--plot-decimation-factor", type=int, default=1, help="Temporal decimation factor for plotting only, to speed up carpet map generation.")

    # --- Local Staging Parameters ---
    parser.add_argument('--no-local-staging', action='store_true', default=False,
                        help="Disable local file staging for large .nii.gz files. "
                             "By default, files larger than --local-staging-threshold-mb "
                             "are copied and decompressed to a local directory for faster loading.")
    parser.add_argument('--local-staging-dir', type=str, default=None,
                        help="Parent of the per-process staging directory (default: the system temp directory; "
                             "a bold_lag_mapper_staging_<PID> child is created and removed). "
                             "Must be on a local filesystem for performance benefit.")
    parser.add_argument('--local-staging-threshold-mb', type=float, default=200.0,
                        help="Size threshold in MB for local file staging. Files smaller than this "
                             "are loaded directly from their original location. Default: 200 MB.")

    return parser


def apply_fmriprep_inputs(args, parser):
    """Fill --bold-files, --motion-confounds-files and --mask-file from --fmriprep-dir (fMRIPrep mode)."""
    if args.fmriprep_dir is None:
        if args.participant_label is not None:
            parser.error("--participant-label needs --fmriprep-dir.")
        if not args.bold_files:
            parser.error("Give --bold-files (with --mask-file), or --fmriprep-dir with --participant-label.")
        return None
    if args.bold_files or args.mask_file or args.motion_confounds_files:
        parser.error("--fmriprep-dir replaces --bold-files, --mask-file and --motion-confounds-files; do not combine them.")
    if not args.participant_label:
        parser.error("--fmriprep-dir needs --participant-label.")
    found = discover_fmriprep_inputs(args.fmriprep_dir, args.participant_label, session=args.session, task=args.task,
                                     space=args.space, res=args.res, runs=args.runs, min_volumes=args.min_volumes)
    args.bold_files, args.motion_confounds_files, args.mask_file = found.bold_files, found.confounds_files, found.mask_file
    args.native_space = args.space == "T1w"
    return found


def main():
    """
    Main function to parse arguments and run the BOLD lag mapping pipeline.
    """
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Setup Logging ---
    # Creates a log file in the output directory and also prints to the console.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(args.output_dir, f"bold_lagmapper_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path, mode="w"), logging.StreamHandler()],
    )

    # --- Inputs: explicit files or fMRIPrep discovery ---
    apply_fmriprep_inputs(args, parser)

    # --- BIDS Validation and Pipeline Detection ---
    if args.validate_bids:
        try:
            validate_bids_inputs(args)
        except Exception as e:
            logger.warning(f"BIDS validation failed: {e}")
            logger.warning("Continuing with user-provided settings...")

    # --- Enhanced Argument Validation ---
    if not args.mask_file and not args.generate_mask:
        parser.error("A mask must be provided via --mask-file, or enabled with --generate-mask.")

    # The count check must NOT depend on motion regression: confounds are also read for non-steady-state
    # trimming and the FD spike rule, and core.process_runs zips the two lists.
    if args.motion_confounds_files and len(args.bold_files) != len(args.motion_confounds_files):
        parser.error("The number of --bold-files must match the number of --motion-confounds-files.")

    # Validate native space requirements
    if args.native_space:
        logger.info("Native space processing enabled - will search for transformation files...")
        if args.auto_seed_from_freesurfer:
            logger.info("Will also search for FreeSurfer segmentation files...")

    # Check for file existence
    missing_files = [f for f in args.bold_files if not os.path.exists(f)]
    if missing_files:
        parser.error(f"BOLD files not found: {missing_files}")

    # Log all the arguments used for this run for reproducibility
    logger.info("--- Script Arguments ---")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("------------------------")

    # If no motion files are provided, create a list of Nones to match the BOLD files list
    motion_files = args.motion_confounds_files or [None] * len(args.bold_files)

    # Instantiate the mapper class with all arguments, converting the args namespace to a dictionary
    mapper = BOLDLagMapper(**vars(args))

    # Run the main processing pipeline
    mapper.process_runs(args.bold_files, motion_files, args.mask_file, args.output_dir)
