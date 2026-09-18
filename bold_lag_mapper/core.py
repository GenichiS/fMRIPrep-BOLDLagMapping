"""
Core Module
-----------

This module contains the main `BOLDLagMapper` class, which acts as the orchestrator
for the entire lag mapping and deperfusion pipeline. It initializes with all the
parameters from the command-line interface, holds the pipeline's state (e.g., TR,
masker object), and calls functions from the other modules (io, processing, viz, utils)
to perform the actual work in a logical sequence. This class-based approach allows for
a clean and organized management of the numerous parameters and intermediate data
products involved in the analysis.
"""

import gc
import logging
import os
import tempfile
import warnings
from pathlib import Path
import nibabel as nib
import numpy as np
from nilearn import image as nilearn_image
from nilearn.maskers import NiftiMasker

# Import the refactored modules
from . import (
    io, lag_preprocess, lag_postprocess, utils, viz,
    lag_estimators_matlab, seeds
)


# --- Conditional Import for GPU (CuPy) ---
# This allows the code to run on systems without a GPU or CuPy installed.
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    # If CuPy isn't found, alias `cp` to `numpy`. This allows the rest of the code
    # to use `xp` as a generic handle for array operations, regardless of the backend.
    cp = np
    _CUPY_AVAILABLE = False
    warnings.warn(
        "CuPy not found. Running computations on CPU. Install CuPy for a significant speed-up.",
        ImportWarning,
    )

logger = logging.getLogger(__name__)


def resolve_effective_tracking_method(tracking_method, tr, subtr_min_tr):
    """Sub-TR refinement only where the tracking step is coarse enough.

    `recursive_subtr` is kept for a tracking step >= subtr_min_tr (e.g. TR 2.5 s tracked at the TR). Below
    it (e.g. TR 0.8 s, or any 1 s resampled series) the integer-step `recursive` method - the MATLAB
    FIXED=0 replica, no sub-TR step - is run instead, and the caller names its outputs after the method
    actually run. Pure function so it can be unit-tested without a BOLD file.

    The gate does more than switch off sub-TR refinement - the two propagation loops use different
    local-peak windows, so the ACCEPTANCE RULE differs:
      recursive_subtr (step >= subtr_min_tr): window_trs=2, a 5-point window; the centre is accepted
          only if corr(0) beats BOTH +/-1 and +/-2 steps.
      recursive       (step <  subtr_min_tr): window_trs=1, a 3-point window; corr(0) need only beat
          +/-1 step, which is strictly the laxer test.
    window_trs=1 is the MATLAB original (Lim=1) for integer tracking, and the windows are not equivalent
    in seconds either (+/-1.6 s at TR 0.8 s vs +/-5 s at TR 2.5 s). Report which rule applied when data
    with different TRs are pooled.
    """
    # fixed_subtr follows the same gate and resolves to the integer 'fixed' (MATLAB FIXED=1). `tr` is
    # the TRACKING step: the TR, or --tracking-step-seconds when the series is resampled, so a 1 s step
    # always resolves the sub-TR methods to their integer twins.
    _integer_twin = {"recursive_subtr": "recursive", "fixed_subtr": "fixed"}
    if tracking_method in _integer_twin and subtr_min_tr is not None and tr < subtr_min_tr:
        twin = _integer_twin[tracking_method]
        logger.warning(
            f"Tracking step {tr:.3f} s is below --subtr-min-tr {subtr_min_tr:.2f} s: sub-TR refinement is NOT "
            f"applied; running the integer-step '{twin}' method (MATLAB FIXED={0 if twin == 'recursive' else 1}). "
            f"Output files are named tracking-{twin} so the artefact reflects the method that ran.")
        return twin
    return tracking_method


def resolve_tracking_step(value, tr, subtr_min_tr):
    """Tracking step in seconds, or None to track at the acquisition TR.

    'auto' tracks long-TR data (TR >= subtr_min_tr) on a 1 s grid, as the long-TR variant of the original code
    does (drLag4Drev7_longTR; boldlag --reso 1), and short-TR data at their own TR. 'none' keeps the TR.
    """
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "auto":
            return 1.0 if (subtr_min_tr is not None and tr >= subtr_min_tr) else None
        if v == "none":
            return None
        value = float(v)
    value = float(value)
    if value <= 0:
        raise ValueError(f"The tracking step must be positive, got {value}")
    return value


class BOLDLagMapper:
    """
    This class manages the state and execution of the BOLD Lag Mapping and
    Deperfusion Pipeline, providing a Python replication and extension of the
    original MATLAB implementation by Dr. Toshihiko Aso.
    """
    def __init__(self, **kwargs):
        """
        Initializes the BOLDLagMapper pipeline with specified parameters from the CLI.

        Args:
            **kwargs: A dictionary of keyword arguments, typically parsed by argparse,
                      containing all runtime parameters for the pipeline.
        """
        # Store all parameters passed from the CLI as attributes of the class instance
        # This makes parameters like `self.tr`, `self.tracking_method` easily accessible.
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Set the array backend based on GPU availability and user request
        self.use_gpu = self.use_gpu and _CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np
        if self.use_gpu:
            logger.info("CuPy found. GPU acceleration is ENABLED.")
        else:
            logger.info("GPU acceleration is DISABLED. Running on CPU.")

        # --- Initialize state variables that will be populated during the pipeline run ---
        self.tr = None
        self.num_timepoints = None
        self.num_voxels = None
        self.masker = None # Nilearn object for converting between 3D/4D images and 2D matrices
        self.lag_map_values = None
        self.max_correlations = None
        self.mask_img_data = None
        self.seed_time_series = {} # Dictionary to store prototypical seed signals for deperfusion
        self.bold_files = []
        self.output_dir = "."
        self.seed_mask_1d = None

        # --- Native space processing state ---
        self.is_native_space_run = False
        self.transform_to_native_path = None
        self.transform_to_mni_path = None
        self.mni_template_ants = None

    def process_runs(self, bold_files, motion_confounds_files, mask_file, output_dir="."):
        """
        Main method to run the entire multi-run lag mapping and deperfusion pipeline.
        
        This method executes the following steps in sequence:
        1.  Validates all inputs.
        2.  Sets up paths and native space processing if required.
        3.  Creates or loads the analysis mask.
        4.  Loops through each BOLD run to perform preprocessing in a memory-efficient manner.
        5.  Concatenates the processed data from all runs.
        6.  Computes the lag map using the selected tracking method.
        7.  Performs post-processing on the lag map (hole-filling).
        8.  Saves all output files (lag maps, correlation maps, etc.).
        9.  If enabled, performs deperfusion on each run and saves the cleaned BOLD files.
        """
        # Validate the BOLD/confounds pairing HERE, not only in the CLI. The loop below zips the two
        # lists, so a short confounds list would silently drop the trailing runs and still log
        # "Processing N runs" and finish cleanly. The Python API calls this method directly.
        if motion_confounds_files is None:
            motion_confounds_files = [None] * len(bold_files)
        if len(motion_confounds_files) != len(bold_files):
            raise ValueError(
                f"{len(bold_files)} BOLD run(s) but {len(motion_confounds_files)} motion-confounds "
                "entr(y/ies): the lists must be the same length. Pass None for a run that has no "
                "confounds; confounds are used for non-steady-state trimming and the FD spike rule "
                "even when motion regression is disabled.")

        # --- Setup local staging directory ---
        staging_dir = None
        if not getattr(self, 'no_local_staging', False):
            # --local-staging-dir is a PARENT: only a per-process child of it is created and later removed,
            # so a pre-existing directory with the user's own files is never deleted and concurrent runs
            # cannot delete each other's staged files.
            _staging_parent = getattr(self, "local_staging_dir", None) or tempfile.gettempdir()
            staging_dir = os.path.join(_staging_parent, f"bold_lag_mapper_staging_{os.getpid()}")
            logger.info(f"Local staging enabled (threshold: {getattr(self, 'local_staging_threshold_mb', 200.0):.0f} MB, dir: {staging_dir})")
        staging_threshold_mb = getattr(self, 'local_staging_threshold_mb', 200.0)

        try:
            os.makedirs(output_dir, exist_ok=True)   # the Python API calls process_runs directly,
            # so the output directory is created here as well as in cli.main().
            self.bold_files = bold_files
            self.output_dir = output_dir
            # --- 1. Initialization and Validation ---
            self.tr = utils.validate_inputs(bold_files, mask_file, self.generate_mask)
            first_img_nib = nib.load(bold_files[0])
            # The TRACKING step is the TR unless --tracking-step-seconds asks for the filtered series to be
            # resampled (drLag4Drev7_longTR / boldlag --reso); 'auto' does so for long-TR data only.
            self.tracking_step_seconds = resolve_tracking_step(
                getattr(self, 'tracking_step_seconds', None), float(self.tr), getattr(self, 'subtr_min_tr', 1.5))
            self.tr_track = self.tracking_step_seconds if self.tracking_step_seconds else float(self.tr)
            # Sub-TR refinement is applied only when the tracking step is >= --subtr-min-tr (default 1.5 s).
            # Resolved before the output prefix is built so that file names carry the method actually run.
            self.tracking_method_requested = self.tracking_method
            self.tracking_method = resolve_effective_tracking_method(
                self.tracking_method, self.tr_track, getattr(self, 'subtr_min_tr', 1.5))
            logger.info(f"Effective tracking method: {self.tracking_method} "
                        f"(TR {self.tr:.3f} s, tracking step {self.tr_track:.3f} s)")
            _steps = int(round(self.max_lag_seconds / self.tr_track))
            logger.info(f"Lag search: +/-{_steps} step(s) of {self.tr_track:g} s = +/-{_steps * self.tr_track:g} s "
                        f"(--max-lag-seconds {self.max_lag_seconds:g}); with --boundary-null the outermost step is "
                        "discarded and re-filled.")

            # The seed: the bundled deep white-matter mask, a user-supplied mask, or the global mean signal.
            self.seed_roi_file = seeds.resolve_seed_roi(
                getattr(self, 'seed_roi_file', 'builtin'), bold_files[0], bool(self.native_space),
                bool(getattr(self, 'auto_seed_from_freesurfer', False)))

            # --- 2. Setup Native Space Processing (if enabled) ---
            if self.native_space:
                io.setup_native_space_processing(self, bold_files[0], mask_file, output_dir)

            # Define a consistent output file prefix based on input filenames and key parameters
            # A character-level common prefix of the run names would give a truncated, non-BIDS stem for
            # multi-run data (dir-AP/dir-PA x run-01..04 -> "sub-X_ses-01_task-rest_dir-"). Build the stem
            # from the first run instead and drop only the entities that vary across runs (run-, dir-),
            # keeping every other entity intact.
            if len(bold_files) == 1:
                base_name = os.path.basename(bold_files[0]).split(".nii")[0]
            else:
                _stem = os.path.basename(bold_files[0]).split(".nii")[0]
                _kept = [p for p in _stem.split("_")
                         if not (p.startswith("run-") or p.startswith("dir-"))]
                base_name = "_".join(_kept)
                logger.info(f"{len(bold_files)} runs: output stem '{base_name}' "
                            f"(run-/dir- entities dropped from '{_stem}').")
            _step_tag = f"_step{str(self.tr_track).replace('.', 'p')}" if self.tracking_step_seconds else ""
            output_prefix = os.path.join(output_dir, f"{base_name}_tracking-{self.tracking_method}_lag{str(self.max_lag_seconds).replace('.', 'p')}{_step_tag}_sm{int(self.spatial_fwhm)}_thr{int(self.min_corr_threshold*100):02d}")
            logger.info(f"--- BOLD Lag Mapping Pipeline ---")
            logger.info(f"Processing {len(bold_files)} runs. Output prefix: {output_prefix}")

            # --- 3. Seed and Mask Preparation ---
            # Automatically generate a robust seed from FreeSurfer outputs if requested
            final_seed_roi_file = self.seed_roi_file
            if self.auto_seed_from_freesurfer:
                logger.info("--auto-seed-from-freesurfer is enabled. Creating robust seed mask.")
                try:
                    # This will override any manually provided seed_roi_file
                    final_seed_roi_file = io.create_cerebrum_mask_from_freesurfer(
                        bold_ref_file=bold_files[0],
                        output_dir=output_dir
                    )
                except FileNotFoundError as e:
                    logger.error(f"Failed to auto-create cerebrum mask. Ensure your directory structure is correct.", exc_info=True)
                    raise
            self.seed_roi_file = final_seed_roi_file

            if not self.seed_roi_file:
                 logger.warning("No seed ROI specified or generated. Defaulting to using the entire brain mask as the seed.")

            # Generate or load the brain mask for the analysis
            analysis_mask_img = nib.load(mask_file) if not self.generate_mask else nilearn_image.new_img_like(first_img_nib, (nilearn_image.mean_img(first_img_nib).get_fdata() > self.mask_threshold).astype(np.uint8))
            undilated_mask_img = analysis_mask_img   # kept for spike detection (see below)

            # Dilate the mask if requested, using the accurate, refactored function
            analysis_mask_img = lag_preprocess.dilate_mask_mm_accurate(
                analysis_mask_img, self.dilate_mask_mm
            )
            if self.dilate_mask_mm > 0 and self.save_debugging_files:
                nib.save(analysis_mask_img, f"{output_prefix}_dilated_mask.nii.gz")

            # Resample mask to match BOLD resolution to avoid costly 4D resampling.
            # Without this, NiftiMasker.transform() resamples every BOLD volume to
            # the mask grid (e.g., HCP 0.7mm mask → 260^3 per volume × 375 timepoints).
            bold_ref_img = nilearn_image.index_img(first_img_nib, 0)
            # Compare the AFFINE as well as the shape. The BOLD is extracted with a raw boolean index
            # (bold_data_4d[self.mask_img_data]) below, which bypasses the affine check NiftiMasker.transform
            # would do, so a same-shape mask with a different origin/orientation/voxel size would silently
            # select the wrong anatomy.
            _grid_differs = (analysis_mask_img.shape[:3] != bold_ref_img.shape[:3]
                             or not np.allclose(analysis_mask_img.affine, bold_ref_img.affine, atol=1e-4))
            if _grid_differs:
                logger.info(f"Resampling mask {analysis_mask_img.shape[:3]} to match the BOLD grid "
                            f"{bold_ref_img.shape[:3]} (shape and/or affine differ)...")
                analysis_mask_img = nilearn_image.resample_to_img(
                    analysis_mask_img, bold_ref_img,
                    interpolation='nearest', force_resample=True, copy_header=True
                )
                # Re-binarize after resampling
                analysis_mask_img = nib.Nifti1Image(
                    (analysis_mask_img.get_fdata() > 0.5).astype(np.uint8),
                    analysis_mask_img.affine, analysis_mask_img.header
                )
                logger.info(f"Mask resampled to {analysis_mask_img.shape[:3]}.")

            if self.save_debugging_files:
                nib.save(analysis_mask_img, f"{output_prefix}_analysis_mask.nii.gz")

            # Initialize the Nilearn masker, which will handle data conversion and masking
            self.masker = NiftiMasker(mask_img=analysis_mask_img, t_r=self.tr)
            self.masker.fit()
            self.mask_img_data = self.masker.mask_img_.get_fdata().astype(bool)
            self.num_voxels = np.sum(self.mask_img_data)

            # DVARS for spike detection is computed over the UNDILATED brain mask only. With
            # --dilate-mask-mm 4 the ring outside the brain has near-zero means, so its signal change
            # would dominate DVARS and the detector would flag ring noise instead of head motion. HCP
            # volumes are zero outside the brain, so the MATLAB whole-FOV DVARS was brain-only in effect;
            # fMRIPrep's preproc BOLD is not. The lag-mapping mask itself is the dilated one.
            self.spike_mask_1d = lag_preprocess.mask_to_masker_index(undilated_mask_img, self.masker)
            logger.info(f"Spike detection (DVARS) restricted to the undilated brain mask: "
                        f"{int(self.spike_mask_1d.sum()):,} of {self.num_voxels:,} analysis voxels.")

            # --- 4. Per-Run Preprocessing Loop ---
            self.final_hp_cutoff_hz = self.bandpass_low if self.bandpass_low is not None else 0.008
            if self.bandpass_high is not None:
                self.final_lp_cutoff_hz = self.bandpass_high
            else:
                self.final_lp_cutoff_hz = 1.0 / (self.max_lag_seconds * 2.0) * 0.9
            
            logger.info(f"Final band-pass filter range: {self.final_hp_cutoff_hz:.4f} Hz - {self.final_lp_cutoff_hz:.4f} Hz")
            hp_sigma = 1 / (self.final_hp_cutoff_hz * 2.35 * self.tr)
            lp_sigma = 1 / (self.final_lp_cutoff_hz * 2.35 * self.tr) if self.final_lp_cutoff_hz > 0 else -1

            # --- MEMORY REFACTOR ---
            # Instead of storing all runs' data, process sequentially and only keep what's needed.
            all_runs_filtered_data_for_concat = []
            all_runs_data_for_deperfusion = []
            run_lengths = []
            self.nss_trimmed = {}   # run index -> number of leading non-steady-state volumes dropped
            self.run_stats = []     # per-run facts for the stats sidecar

            for i, (bold_file, confounds_file) in enumerate(zip(bold_files, motion_confounds_files)):
                logger.info(f"--- Pre-processing Run {i+1}/{len(bold_files)}: {os.path.basename(bold_file)} ---")

                # Stage large files to local disk for faster loading
                load_path = bold_file
                if staging_dir is not None:
                    load_path = io.stage_to_local(bold_file, staging_dir, staging_threshold_mb)

                img = nib.load(load_path)
                num_timepoints_run = img.shape[-1]

                # Load as float32 and apply mask manually (mask already resampled to BOLD grid).
                # Assert the grid at the seam: this raw boolean index bypasses NiftiMasker.transform's
                # own affine check.
                if img.shape[:3] != self.mask_img_data.shape or not np.allclose(
                        img.affine, self.masker.mask_img_.affine, atol=1e-4):
                    raise ValueError(
                        f"Run {i+1} grid {img.shape[:3]} / affine does not match the analysis mask "
                        f"{self.mask_img_data.shape}; refusing to index the BOLD with a mask from a "
                        "different grid.")
                bold_data_4d = img.get_fdata(dtype=np.float32)
                timeseries_2d_raw_np = bold_data_4d[self.mask_img_data].T
                del bold_data_4d
                gc.collect()

                # Drop the LEADING volumes fMRIPrep flagged as non-steady-state (confounds columns
                # non_steady_state_outlier_XX, one-hot) from BOTH the BOLD and the confounds of this run,
                # before anything else touches the data. Short-TR data can carry several such volumes per
                # run, more than the taper suppresses, and they are not steady-state signal. Only the
                # leading contiguous block is dropped (see io.count_leading_non_steady_state); the
                # deperfusion outputs, if requested, are correspondingly shorter.
                n_nss = 0
                if getattr(self, 'trim_non_steady_state', True) and confounds_file is not None:
                    n_nss = io.count_leading_non_steady_state(confounds_file, num_timepoints_run)
                if n_nss > 0:
                    timeseries_2d_raw_np = timeseries_2d_raw_np[n_nss:]
                    num_timepoints_run -= n_nss
                    logger.info(f"  Trimmed {n_nss} leading non-steady-state volume(s); {num_timepoints_run} volumes remain.")
                else:
                    logger.info("  Non-steady-state trimming: nothing to drop for this run.")
                self.nss_trimmed[i] = n_nss
                run_lengths.append(num_timepoints_run)

                # Clean up staged file immediately to free /tmp space
                if load_path != bold_file:
                    try:
                        os.remove(load_path)
                        logger.info(f"  Removed staged file: {os.path.basename(load_path)}")
                    except OSError:
                        pass

                confound_matrices, confound_names = [], []
                loaded_confounds_np, loaded_names = io.load_motion_confounds(confounds_file, num_timepoints_run, self.apply_motion_correction, self.custom_motion_regressors, skip_leading=n_nss, txt_motion_schema=getattr(self, "txt_motion_schema", "spm"))
                if loaded_confounds_np is not None:
                    # Pass loaded_names: io.load_motion_confounds requests the fMRIPrep columns as rot_*
                    # then trans_*, so FD must resolve them by name, not position.
                    motion_confounds_np, motion_names = lag_preprocess.calculate_motion_derivatives_and_fd(loaded_confounds_np, loaded_names) if not self.custom_motion_regressors else (loaded_confounds_np, loaded_names)
                    confound_matrices.append(motion_confounds_np)
                    confound_names.extend(motion_names)
                _aso_count, _spike_counts = None, {}
                if self.apply_spike_regression:
                    # brain-only voxels for DVARS; the regressors apply to every voxel
                    spike_input_np = timeseries_2d_raw_np[:, self.spike_mask_1d]
                    # Spikes = DVARS rule OR FD > --fd-spike-threshold. FD comes from fMRIPrep's
                    # framewise_displacement column (trimmed like the BOLD); if absent, from the FD computed
                    # from the motion parameters; if neither, DVARS only.
                    fd_thr = float(getattr(self, 'fd_spike_threshold', 0.5) or 0.0)
                    fd_run = None
                    if fd_thr > 0:
                        fd_run = io.load_framewise_displacement(confounds_file, num_timepoints_run, skip_leading=n_nss)
                        if fd_run is None and loaded_confounds_np is not None and 'framewise_displacement' in confound_names:
                            fd_run = motion_confounds_np[:, confound_names.index('framewise_displacement')]
                        if fd_run is None:
                            logger.warning("FD spike rule requested but no framewise displacement is available for this run; using DVARS only.")
                    _spike_method = getattr(self, 'spike_method', 'robustz')
                    _dvars_def = getattr(self, 'dvars_definition', 'boldlag')   # Aso's definition is the default
                    if _spike_method == 'afyouni-nichols':
                        spike_regressors_np, spike_names = lag_preprocess.identify_and_create_spike_regressors_afyouni_nichols(
                            spike_input_np, num_timepoints_run,
                            getattr(self, 'an_alpha', 0.05), getattr(self, 'an_dpd_threshold', 5.0),
                            self.include_preceding_spike, fd=fd_run, fd_threshold=fd_thr)
                    elif _spike_method == 'aso-median':
                        # Aso's 1.5 x median rule on the chosen DVARS series
                        spike_regressors_np, spike_names = lag_preprocess.identify_and_create_spike_regressors_aso_median(
                            spike_input_np, num_timepoints_run, self.include_preceding_spike,
                            fd=fd_run, fd_threshold=fd_thr, dvars_definition=_dvars_def)
                    else:
                        spike_regressors_np, spike_names = lag_preprocess.identify_and_create_spike_regressors(
                            spike_input_np, num_timepoints_run, self.dvars_threshold, self.include_preceding_spike,
                            fd=fd_run, fd_threshold=fd_thr, dvars_definition=_dvars_def)
                    # Aso's 1.5 x median count for the sidecar, on Aso's own DVARS definition (the detectors
                    # stash it; the Afyouni-Nichols path computes it here).
                    _spike_counts = dict(lag_preprocess.LAST_SPIKE_COUNTS)
                    _aso_count = _spike_counts.get('aso_1p5median')
                    if _aso_count is None:
                        _aso_count = lag_preprocess.aso_median_rule_count(lag_preprocess.compute_dvars_boldlag(spike_input_np))
                    del spike_input_np
                    if spike_regressors_np is not None:
                        confound_matrices.append(spike_regressors_np)
                        confound_names.extend(spike_names)
                        logger.info(f"Spike regressors (run {i+1}): {spike_names}")
                
                self.run_stats.append(dict(
                    file=os.path.basename(bold_file), n_frames_raw=int(num_timepoints_run + n_nss),
                    n_nss_trimmed=int(n_nss), n_frames=int(num_timepoints_run),
                    spikes_dvars=_spike_counts.get('dvars'), spikes_fd=_spike_counts.get('fd'),
                    spikes_union=_spike_counts.get('union'), spikes_final=_spike_counts.get('final'),
                    spikes_aso_1p5median=_aso_count))
                full_confounds_np = np.hstack(confound_matrices) if confound_matrices else None
                if self.save_carpet_map and full_confounds_np is not None:
                    viz.save_regressor_carpet_map(full_confounds_np, confound_names, f"{output_prefix}_run-{i+1}")

                timeseries_2d_cleaned_np = lag_preprocess.perform_nuisance_regression(timeseries_2d_raw_np, full_confounds_np)

                # The cleaned run as a 4D file, for external validators (e.g. boldlag lag4d) that must start
                # from exactly this series. `img` is the run's own image, so the affine and the header TR are
                # preserved; the trimmed length is what the data carry.
                if getattr(self, 'save_cleaned_bold', False):
                    _run_base = Path(bold_file).name.split(".nii")[0]
                    _cleaned_img = io.copy_nifti_affine_and_header(
                        img, self.masker.inverse_transform(timeseries_2d_cleaned_np.astype(np.float32)))
                    _cleaned_img.header.set_zooms((*_cleaned_img.header.get_zooms()[:3], float(self.tr)))
                    _cleaned_img.header.set_xyzt_units('mm', 'sec')      # boldlag lag4d reads the TR from here
                    _cleaned_path = os.path.join(output_dir, f"{_run_base}_desc-cleaned_bold.nii.gz")
                    nib.save(_cleaned_img, _cleaned_path)
                    logger.info(f"  Saved cleaned run ({timeseries_2d_cleaned_np.shape[0]} frames) to {_cleaned_path}")

                # --- MEMORY REFACTOR ---
                # Store data needed for deperfusion and carpet plots
                # The retained raw/cleaned arrays are ONLY read inside the
                # `if self.save_deperfusioned_bold != "none":` block below (the carpet figure is drawn
                # there too), so they are kept only when deperfusion is requested.
                if self.save_deperfusioned_bold != "none":
                    all_runs_data_for_deperfusion.append({
                        "raw": timeseries_2d_raw_np,
                        "cleaned": timeseries_2d_cleaned_np
                    })

                del timeseries_2d_raw_np
                gc.collect()

                # --- PREPROCESSING ---
                # Percent signal change calculation
                # Note: MATLAB uses 100*S/mean (centered at 100), but Python uses 100*(S/mean - 1)
                # (centered at 0). After bandpass filtering (which removes DC), these are equivalent.
                mean_signal = np.mean(timeseries_2d_cleaned_np, axis=0)
                mean_signal[mean_signal == 0] = 1
                timeseries_psc_np = 100 * (timeseries_2d_cleaned_np / mean_signal - 1)

                # --- MEMORY REFACTOR ---
                # Delete cleaned data immediately after PSC conversion
                del timeseries_2d_cleaned_np
                gc.collect()

                # Spatial smoothing
                timeseries_smoothed_np = self.masker.transform(
                    lag_preprocess.smooth_spm_compat(
                        self.masker.inverse_transform(timeseries_psc_np),
                        self.spatial_fwhm
                    )
                )

                # Apply Tukey window unless skip_tukey_window is set (for MATLAB matching)
                if getattr(self, 'skip_tukey_window', False):
                    logger.info("Skipping Tukey temporal window (MATLAB matching mode).")
                    timeseries_windowed_np = timeseries_smoothed_np
                else:
                    temporal_window = lag_preprocess.create_temporal_window(
                        num_timepoints_run, self.max_lag_seconds, self.tr, getattr(self, 'taper_width', 'fixed5pct'))
                    if self.save_carpet_map:
                        plot_title = f"Tukey Window (Run {i+1}, {num_timepoints_run} timepoints)"
                        viz.save_tukey_window_plot(temporal_window, f"{output_prefix}_run-{i+1}", plot_title)
                    timeseries_windowed_np = timeseries_smoothed_np * temporal_window[:, np.newaxis]

                # Temporal bandpass filtering
                if getattr(self, 'skip_bandpass_filter', False):
                    logger.info("Skipping bandpass filtering (preprocessed data mode).")
                    run_filtered_data = timeseries_windowed_np
                else:
                    run_filtered_data = lag_preprocess.temporal_filter_fsl_replicated(self, timeseries_windowed_np, hp_sigma, lp_sigma)

                # --- MEMORY REFACTOR ---
                # Append only the final filtered data needed for concatenation
                all_runs_filtered_data_for_concat.append(run_filtered_data)

                # Clean up intermediate arrays from the loop
                del timeseries_psc_np
                del timeseries_smoothed_np
                del timeseries_windowed_np
                gc.collect()

            # --- 5. Concatenate Data and Final Prep ---
            # Multi-run handling is MATLAB-equivalent. Nuisance regression and the taper are per RUN
            # (above); the cleaned runs are concatenated here and the lag is estimated ONCE on the joined
            # series - exactly what Einsteining_v07.m does (per-run fsl_regfilt -> drMerge4D -> a single
            # drLag4Drev7 call, whose "cat<N>" argument is only a folder-name string). Lag estimation is
            # sampling-limited, so estimating per run and averaging afterwards is not done.
            # NEVER pass an already-merged multi-run file as ONE run: the taper is then computed from the
            # total length and the internal boundaries are unprotected.
            concatenated_windowed_data = np.vstack(all_runs_filtered_data_for_concat)
            # The series exactly as the estimator receives it, before the amplitude gate / resampling /
            # std-normalisation - the counterpart of boldlag's sm*.nii (lag4d.prepare) for stage-by-stage
            # comparisons. `img` is the last run's image; every run was asserted to share the mask grid
            # above, so affine and header are the BOLD's. Saved in float64, the dtype the estimator
            # actually receives: a float32 copy can flip near-tie peak assignments when the saved series
            # is re-tracked, so it would not reproduce the mapper exactly.
            if getattr(self, 'save_filtered_bold', False):
                _filt_img = io.copy_nifti_affine_and_header(
                    img, self.masker.inverse_transform(np.asarray(concatenated_windowed_data, np.float64)))
                _filt_img.set_data_dtype(np.float64)
                _filt_img.header.set_zooms((*_filt_img.header.get_zooms()[:3], float(self.tr)))
                _filt_img.header.set_xyzt_units('mm', 'sec')
                _filt_path = f"{output_prefix}_desc-filtered_bold.nii.gz"
                nib.save(_filt_img, _filt_path)
                logger.info(f"  Saved filtered series ({concatenated_windowed_data.shape[0]} frames x "
                            f"{concatenated_windowed_data.shape[1]} voxels) to {_filt_path}")
            # --- MEMORY REFACTOR ---
            # Delete the list of arrays after concatenation
            del all_runs_filtered_data_for_concat
            gc.collect()
            
            self.num_timepoints = concatenated_windowed_data.shape[0]
            timeseries_2d_filtered_backend = self.xp.asarray(concatenated_windowed_data)
            
            # --- MEMORY REFACTOR ---
            # Delete the large concatenated CPU array after moving to GPU
            del concatenated_windowed_data
            gc.collect()
            utils.free_gpu_memory(self.use_gpu)

            # --- Seed ROI setup ---
            seed_roi_1d_mask = None
            if self.seed_roi_file:
                logger.info(f"Using provided ROI as seed: {self.seed_roi_file}")
                seed_roi_resampled = nilearn_image.resample_to_img(
                    nib.load(self.seed_roi_file), self.masker.mask_img_,
                    interpolation="nearest", force_resample=True, copy_header=True
                )
                seed_roi_1d_values = self.masker.transform(seed_roi_resampled).flatten()

                # MATLAB threshold: ROIimage(ROIimage < 0.1) = NaN
                seed_roi_1d_mask = seed_roi_1d_values >= 0.1

                if not np.any(seed_roi_1d_mask):
                    raise ValueError(f"Seed ROI file resulted in an empty mask after resampling. Check ROI alignment.")

                logger.info(f"  ROI voxels (>= 0.1): {np.sum(seed_roi_1d_mask):,}")

            # Seed will be computed after global normalization
            global_signal_backend = None

            # MATLAB: the amplitude-based exclusion is applied to the bandpass-filtered
            # percent-signal-change data BEFORE std-normalization, to drop high-amplitude
            # voxels (large vessels / CSF):  MAX = max(abs(Y),[],4); Y = Y.*(MAX<=4);
            # The threshold is in percent-signal units (e.g. 4 == 4% peak).
            #
            # This gate is NOT independent of run length: max|x| is a peak statistic, so on noise the
            # fraction of voxels exceeding a fixed cutoff grows with the number of samples. It is applied
            # to the CONCATENATED series, so a long multi-run short-TR acquisition is screened over many
            # more samples than a short single run at the same cutoff. Excluded voxels become holes and
            # are interpolated, so the amount of interpolation can differ systematically between
            # datasets with different lengths - a confound when data are pooled. The exclusion count and
            # the number of frames it was computed over are therefore logged and written to the sidecar.
            if self.amplitude_threshold > 0:
                n_frames_screened = int(timeseries_2d_filtered_backend.shape[0])
                logger.info(f"Excluding voxels with peak PSC amplitude > {self.amplitude_threshold}% (large vessels / CSF)...")
                max_abs_signal = self.xp.max(self.xp.abs(timeseries_2d_filtered_backend), axis=0)
                exclusion_mask = max_abs_signal > self.amplitude_threshold
                timeseries_2d_filtered_backend[:, exclusion_mask] = 0
                utils.free_gpu_memory(self.use_gpu) # Free memory after this large operation
                n_excl = int(self.xp.sum(exclusion_mask))
                n_vox = int(exclusion_mask.shape[0])
                self.amplitude_exclusion_stats = dict(n_excluded=n_excl, n_voxels=n_vox,
                                                      frac=n_excl / max(n_vox, 1),
                                                      n_frames=n_frames_screened,
                                                      threshold=float(self.amplitude_threshold))
                logger.info(f"Excluded {n_excl:,} of {n_vox:,} voxels ({100.0*n_excl/max(n_vox,1):.2f}%) "
                            f"by the {self.amplitude_threshold}% peak-PSC threshold, computed over "
                            f"{n_frames_screened} concatenated frames. NOTE this fraction is "
                            "length-dependent (max|x| is a peak statistic); compare it across "
                            "datasets only alongside the frame count.")

            # Optional resampling to the tracking step, exactly where drLag4Drev7_longTR / boldlag lag4d do
            # it - after the band-pass and the amplitude gate, before std-normalisation and the seed (both
            # are computed on the resampled series).
            self.run_lengths_track = [int(n) for n in run_lengths]
            if self.tracking_step_seconds and abs(self.tr_track - self.tr) > 1e-9:
                logger.info(f"Resampling the filtered series from TR {self.tr:.3f} s to the tracking step "
                            f"{self.tr_track:.3f} s (resample_poly, Kaiser window; drLag4Drev7_longTR).")
                _ts_np = utils.to_numpy(timeseries_2d_filtered_backend, self.use_gpu)
                _ts_np = lag_preprocess.resample_for_tracking(_ts_np, self.tr, self.tr_track)
                timeseries_2d_filtered_backend = self.xp.asarray(_ts_np)
                self.run_lengths_track = [int(round(n * self.tr / self.tr_track)) for n in run_lengths]
                self.num_timepoints = int(_ts_np.shape[0])
                del _ts_np
                utils.free_gpu_memory(self.use_gpu)
                logger.info(f"  {sum(run_lengths)} frames at TR -> {self.num_timepoints} samples at the tracking step "
                            f"(per run: {self.run_lengths_track}).")

            # MATLAB-MATCHING: Divide by std only (no mean subtraction)
            # MATLAB: Y = Y./repmat( Ysd, [ size( Y,1) 1]);
            ts_std = self.xp.std(timeseries_2d_filtered_backend, axis=0)
            ts_std[ts_std == 0] = 1
            timeseries_2d_filtered_backend = timeseries_2d_filtered_backend / ts_std

            # Compute seed if not already computed via MATLAB-style normalization
            if global_signal_backend is None:
                # MATLAB-MATCHING: Compute global signal as mean of ALREADY std-normalized voxels
                # MATLAB: Seed = nanmean( sY,2); where sY is std-normalized
                # This is different from z-scoring then averaging - each voxel contributes equally
                if seed_roi_1d_mask is not None:
                    global_signal_backend = self.xp.mean(timeseries_2d_filtered_backend[:, self.xp.asarray(seed_roi_1d_mask)], axis=1)
                else:
                    logger.info("Using global signal (mean of all in-mask voxels) as seed.")
                    global_signal_backend = self.xp.mean(timeseries_2d_filtered_backend, axis=1)

            # --- 6. Compute Lag Map ---
            lag0_mask = None
            if self.tracking_method == "recursive":
                self.lag_map_values, self.max_correlations, lag0_mask = lag_estimators_matlab.compute_lag_maps_recursive(self, timeseries_2d_filtered_backend, global_signal_backend)
            elif self.tracking_method == "recursive_subtr":
                self.lag_map_values, self.max_correlations, lag0_mask = lag_estimators_matlab.compute_lag_maps_recursive_subtr(self, timeseries_2d_filtered_backend, global_signal_backend)
            elif self.tracking_method == "fixed":
                self.lag_map_values, self.max_correlations, lag0_mask = lag_estimators_matlab.compute_lag_maps_fixed_seed(self, timeseries_2d_filtered_backend, global_signal_backend)
            elif self.tracking_method == "fixed_subtr":
                self.lag_map_values, self.max_correlations, lag0_mask = lag_estimators_matlab.compute_lag_maps_fixed_subtr(self, timeseries_2d_filtered_backend, global_signal_backend)
            else:
                raise ValueError(f"Unknown tracking method '{self.tracking_method}'.")

            # --- MEMORY REFACTOR ---
            # Main computation is done, free GPU memory.
            del timeseries_2d_filtered_backend, global_signal_backend
            utils.free_gpu_memory(self.use_gpu)
            gc.collect()

            if self.save_debugging_files and lag0_mask is not None:
                lag0_mask_np = utils.to_numpy(lag0_mask, self.use_gpu)
                lag0_seed_mask_img = self.masker.inverse_transform(lag0_mask_np.astype(np.int8))
                nib.save(lag0_seed_mask_img, f"{output_prefix}_lag0_seed_mask.nii.gz")

            if self.save_carpet_map and 0 in self.seed_time_series:
                # the seed lives on the tracking grid, so the time axis uses the tracking step, not the TR
                viz.save_seed_signal_plot(self.seed_time_series[0], float(getattr(self, 'tr_track', self.tr)), output_prefix)

            # --- 7. Post-process and Save Lag Map ---
            lag_map_values_np = utils.to_numpy(self.lag_map_values, self.use_gpu)
            lag_map_3d_initial = self.masker.inverse_transform(lag_map_values_np).get_fdata()

            # MATLAB drErode_Lag (rev8hcp L354/L536: Y(abs(Y)>=MaxLag)=NaN) discards the OUTERMOST lag
            # bin and re-fills it like any other hole. As explained in issue #3 of the original repository:
            # with the LP cut-off 0.9/(2*PosiMax) the two boundary bins are ~0.9 period apart, so their
            # waveforms are near-identical and the peak at the end of the curve is unreliable. The boundary
            # is identified from the KNOWN search range ((max_lag_trs-0.5)*step) rather than nanmax(lag):
            # the same set for integer maps; for sub-TR maps it is a SECONDS threshold on the phase-corrected
            # value, so at TR 2.5 s / 3 steps everything with |lag| >= 6.25 s is nulled and |lag| < 6.25 s is
            # kept (part of the outer bin survives). This also improves deperfusion, since these voxels then
            # regress a reliable interior-lag seed instead of the most edge-contaminated boundary seed.
            tr_track = float(getattr(self, 'tr_track', self.tr))
            max_lag_trs = int(round(self.max_lag_seconds / tr_track))
            mask_bool = self.mask_img_data.astype(bool)
            lag_map_3d_raw = lag_map_3d_initial.copy()      # estimator output before any nulling (saved below)
            _bn_enabled = bool(getattr(self, 'boundary_null', True))
            lag_map_3d_initial, n_boundary_cand, n_boundary_nulled, boundary_lag_threshold = \
                lag_postprocess.null_search_boundary(lag_map_3d_initial, mask_bool, max_lag_trs, tr_track, enabled=_bn_enabled)
            self.boundary_stats = dict(n_candidates=n_boundary_cand, n_nulled=n_boundary_nulled,
                                       threshold_s=boundary_lag_threshold, enabled=_bn_enabled)
            logger.info(f"Search-boundary voxels: {n_boundary_cand:,} candidates (|lag| >= {boundary_lag_threshold:.2f} s), "
                        f"{n_boundary_nulled:,} nulled (boundary_null={'ON' if _bn_enabled else 'OFF'}).")

            # MATCH THE MATLAB OUTPUT CONVENTION. drLag4Drev7 writes TWO maps: LagOrig.nii, the raw map
            # with NaN wherever no lag was assigned, and LagMap.nii, the same map after drErode_Lag has
            # filled every NaN from its 6-neighbour mean. Out-of-mask voxels arrive as 0.0 from
            # masker.inverse_transform, so with the filled map alone "measured", "interpolated" and
            # "outside the brain" would be indistinguishable downstream. The pre-fill map (NaN = never
            # measured, MATLAB LagOrig) and a boolean validity mask are saved alongside the filled map.
            lag_map_3d_prefill = lag_map_3d_initial.copy()
            lag_map_3d_prefill[~mask_bool] = np.nan          # outside the brain is NaN, not 0
            validity_3d = (~np.isnan(lag_map_3d_prefill)) & mask_bool
            n_measured, n_inmask = int(validity_3d.sum()), int(mask_bool.sum())
            logger.info(f"Lag assigned in {n_measured:,} of {n_inmask:,} in-mask voxels "
                        f"({100.0 * n_measured / max(n_inmask, 1):.1f}%); the remainder is filled by "
                        "neighbour interpolation (MATLAB drErode_Lag).")

            lag_map_3d_filled = lag_postprocess.fill_holes_1by1(lag_map_3d_initial, self.mask_img_data)
            lag_map_eroded = lag_postprocess.fill_isolated_holes_single_pass(lag_map_3d_filled, self.mask_img_data)

            # Convert back to an image to use for saving and final steps
            # BUG FIX: Use float32 dtype for lag values, not uint8 from mask header
            lag_map_img_final = nib.Nifti1Image(
                lag_map_eroded.astype(np.float32),
                self.masker.mask_img_.affine
            )
            
            # Update the 1D values for deperfusion
            self.lag_map_values = self.masker.transform(lag_map_img_final).flatten()

            lag_map_img_native = io.copy_nifti_affine_and_header(first_img_nib, lag_map_img_final)
            max_correlations_np = utils.to_numpy(self.max_correlations, self.use_gpu)
            corr_map_img_native = io.copy_nifti_affine_and_header(first_img_nib, self.masker.inverse_transform(max_correlations_np))

            # MATLAB LagOrig equivalent + an explicit validity mask (see the note above).
            lag_prefill_img_native = io.copy_nifti_affine_and_header(
                first_img_nib, nib.Nifti1Image(lag_map_3d_prefill.astype(np.float32), self.masker.mask_img_.affine))
            validity_img_native = io.copy_nifti_affine_and_header(
                first_img_nib, nib.Nifti1Image(validity_3d.astype(np.uint8), self.masker.mask_img_.affine))
            # The ANALYSIS MASK is written with every map so that consumers define "in brain" from it, never
            # from `lag != 0`. A lag of exactly 0 s is a measured value: on an integer-step map the whole
            # lag-0 band is exactly 0, and a `lag != 0` test would silently drop it.
            analysis_img_native = io.copy_nifti_affine_and_header(
                first_img_nib, nib.Nifti1Image(mask_bool.astype(np.uint8), self.masker.mask_img_.affine))


            # --- 8. Save all results to disk ---
            if self.is_native_space_run:
                if self.save_debugging_files:
                    nib.save(lag_map_img_native, f"{output_prefix}_space-T1w_lagmap.nii.gz")
                    nib.save(corr_map_img_native, f"{output_prefix}_space-T1w_corrmap.nii.gz")

                mni_lag_path = f"{output_prefix}_space-MNI152NLin2009cAsym_lagmap.nii.gz"
                lag_map_img_mni = io.warp_to_mni_and_save(self, lag_map_img_native, mni_lag_path, interpolator="linear")
                if self.save_screenshot:
                    bg_img_path = io.find_anatomical_image(bold_files[0], space="MNI152NLin2009cAsym")
                    viz.save_lag_map_screenshot(
                        lag_map_img_mni, f"{output_prefix}_mni", self.screenshot_z_coord, 
                        self.tracking_method, self.spatial_fwhm, self.min_corr_threshold, 
                        self.screenshot_alpha, self.screenshot_threshold, bg_img_path
                    )
                io.warp_to_mni_and_save(self, corr_map_img_native, f"{output_prefix}_space-MNI152NLin2009cAsym_corrmap.nii.gz")
                # The VALIDITY MASK is the authoritative "was this voxel measured?" artifact in MNI
                # space. NaN does NOT survive the ANTs warp (a warped pre-fill map comes back without
                # NaNs), so a NaN-coded LagOrig cannot be carried to MNI. The mask is warped with
                # nearestNeighbor so it stays binary, and the pre-fill lag map is written in NATIVE
                # space only, where its NaNs are intact.
                io.warp_to_mni_and_save(self, validity_img_native,
                                        f"{output_prefix}_space-MNI152NLin2009cAsym_desc-validity_mask.nii.gz",
                                        interpolator="nearestNeighbor")
                io.warp_to_mni_and_save(self, analysis_img_native,
                                        f"{output_prefix}_space-MNI152NLin2009cAsym_desc-analysis_mask.nii.gz",
                                        interpolator="nearestNeighbor")
                nib.save(lag_prefill_img_native, f"{output_prefix}_space-T1w_desc-prefill_lagmap.nii.gz")
            else:
                nib.save(lag_map_img_native, f"{output_prefix}_lagmap.nii.gz")
                nib.save(corr_map_img_native, f"{output_prefix}_corrmap.nii.gz")
                nib.save(lag_prefill_img_native, f"{output_prefix}_desc-prefill_lagmap.nii.gz")
                nib.save(validity_img_native, f"{output_prefix}_desc-validity_mask.nii.gz")
                nib.save(analysis_img_native, f"{output_prefix}_desc-analysis_mask.nii.gz")
                if self.save_screenshot:
                    bg_img_path = io.find_anatomical_image(bold_files[0], space="MNI152NLin2009cAsym")
                    viz.save_lag_map_screenshot(
                        lag_map_img_native, output_prefix, self.screenshot_z_coord, 
                        self.tracking_method, self.spatial_fwhm, self.min_corr_threshold, 
                        self.screenshot_alpha, self.screenshot_threshold, bg_img_path
                    )

            # --- 8a. Sidecars: raw pre-null map, seeds, numeric facts ---
            # Downstream reports can read these instead of parsing logs or retyping numbers.
            from . import __version__ as _pkg_version
            _raw_suffix = "_space-T1w_desc-raw_lagmap.nii.gz" if self.is_native_space_run else "_desc-raw_lagmap.nii.gz"
            nib.save(io.copy_nifti_affine_and_header(first_img_nib, nib.Nifti1Image(
                np.where(mask_bool, lag_map_3d_raw, np.nan).astype(np.float32), self.masker.mask_img_.affine)),
                f"{output_prefix}{_raw_suffix}")
            _tr_track = float(getattr(self, 'tr_track', self.tr))
            _run_lengths_track = [int(n) for n in getattr(self, 'run_lengths_track', run_lengths)]
            io.write_seeds_npz(f"{output_prefix}_desc-seeds.npz", self.seed_time_series, _tr_track, _run_lengths_track)
            _amp = getattr(self, 'amplitude_exclusion_stats', None) or {}
            self.run_summary_stats = dict(
                version=_pkg_version, tr=float(self.tr), tr_track=_tr_track,
                tracking_step_seconds=getattr(self, 'tracking_step_seconds', None),
                tracking_method_requested=getattr(self, 'tracking_method_requested', self.tracking_method),
                tracking_method_effective=self.tracking_method,
                max_lag_seconds=float(self.max_lag_seconds), max_lag_trs=int(max_lag_trs),
                hp_hz=float(self.final_hp_cutoff_hz), lp_hz=float(self.final_lp_cutoff_hz),
                lp_source="explicit" if self.bandpass_high is not None else "range-linked",
                taper_width=getattr(self, 'taper_width', 'fixed5pct'),
                skip_tukey_window=bool(getattr(self, 'skip_tukey_window', False)),
                boundary_null=self.boundary_stats['enabled'], boundary_threshold_s=self.boundary_stats['threshold_s'],
                n_boundary_candidates=self.boundary_stats['n_candidates'], n_boundary_nulled=self.boundary_stats['n_nulled'],
                n_inmask=int(n_inmask), n_measured=int(n_measured),
                n_measured_raw=int(np.isfinite(lag_map_3d_raw[mask_bool]).sum()),
                amplitude_threshold=float(self.amplitude_threshold),
                n_amp_excluded=_amp.get('n_excluded'), n_frames_screened=_amp.get('n_frames'),
                spike_method=getattr(self, 'spike_method', 'robustz') if self.apply_spike_regression else None,
                dvars_definition=getattr(self, 'dvars_definition', 'boldlag') if self.apply_spike_regression else None,
                dvars_threshold=float(self.dvars_threshold), fd_spike_threshold=float(getattr(self, 'fd_spike_threshold', 0.5) or 0.0),
                n_runs=len(bold_files), runs=self.run_stats,
                seed_voxel_counts={str(k): int(v) for k, v in getattr(self, 'seed_voxel_counts', {}).items()},
                seed_phase_trs={str(k): float(v) for k, v in getattr(self, 'seed_phase_trs', {}).items()},
                boundary_seed_r=None,
            )
            # Aso's lag-structure view + the boundary-seed similarity behind issue #3 of the original
            # repository. Drawn whenever QC figures are requested; the similarity is recorded in the
            # sidecar regardless of the figure.
            try:
                _r = viz.save_slfo_rainbow_plot(self.seed_time_series, _tr_track, output_prefix, _run_lengths_track,
                                                lim_seconds=float(self.max_lag_seconds)) if self.save_carpet_map else \
                     viz.boundary_seed_similarity(self.seed_time_series)
                self.run_summary_stats["boundary_seed_r"] = None if (_r is None or not np.isfinite(_r)) else float(_r)
                logger.info(f"Boundary-seed similarity r(seed0 shifted +max, -max) = {self.run_summary_stats['boundary_seed_r']}")
            except Exception as e:   # a QC figure must never abort the run; the sidecar records the gap
                logger.warning(f"sLFO rainbow / boundary-seed similarity failed: {e}")
            io.write_stats_sidecar(f"{output_prefix}_desc-stats.json", self.run_summary_stats)

            # --- 9. Perform Deperfusion (if enabled) ---
            if self.save_deperfusioned_bold != 'none':
                # For fixed seed, the seed is the same for all lags.
                if self.tracking_method in ('fixed', 'fixed_subtr') and 0 in self.seed_time_series:
                    proto_seed_0 = self.seed_time_series[0]
                    # the same binning as the deperfusion grouping (lag_postprocess.lag_steps, float64)
                    all_lags_in_map_trs = lag_postprocess.fixed_seed_keys(self.lag_map_values, self.tr_track)
                    for lag_trs in all_lags_in_map_trs:
                        self.seed_time_series[lag_trs] = proto_seed_0
                
                if abs(self.tr_track - self.tr) > 1e-9:
                    logger.info(f"Deperfusion with a resampled tracking step ({self.tr_track:g} s vs TR {self.tr:g} s): "
                                "each seed is shifted on the tracking grid, then resampled to the TR grid "
                                "(drDeperf_longTR / boldlag deperf order).")
                # A negative lag step (late signal) is shifted forward in time, mean-padded (non-circular).
                deperfusion_regressors = lag_postprocess.build_deperfusion_regressors(
                    self.seed_time_series, self.tr, self.tr_track, int(sum(run_lengths)))
                # How many voxels are regressed (incl. those assigned to the range edge) / have no lag and
                # keep their temporal mean, in its own sidecar so the stats sidecar stays identical whether or
                # not deperfusion is requested. Raises on a bin inside the regressor range without a
                # regressor, before any output is written.
                _, _dep_summary = lag_postprocess.group_voxels_by_lag_bin(
                    self.lag_map_values, deperfusion_regressors.keys(), self.tr_track)
                io.write_stats_sidecar(f"{output_prefix}_desc-deperfusion.json", _dep_summary)

                # Determine which targets to process
                targets_to_process = []
                if self.save_deperfusioned_bold in ['raw', 'both']:
                    targets_to_process.append('raw')
                if self.save_deperfusioned_bold in ['cleaned', 'both']:
                    targets_to_process.append('cleaned')

                current_timepoint = 0
                for i, bold_file in enumerate(bold_files):
                    run_len = run_lengths[i]
                    # Get the stored data for this run
                    run_data = all_runs_data_for_deperfusion[i]
                    run_regressors = {lag: reg[current_timepoint : current_timepoint + run_len] for lag, reg in deperfusion_regressors.items()}
                    
                    run_basename = Path(bold_file).name.split(".nii")[0]
                    
                    # Process each target type
                    for target in targets_to_process:
                        target_data = run_data[target]
                        deperfusioned_ts = lag_postprocess.apply_deperfusion_regression_matlab_compat(self, target_data, run_regressors)
                        deperfusioned_img_native = io.copy_nifti_affine_and_header(first_img_nib, self.masker.inverse_transform(deperfusioned_ts))
                        
                        # Save carpet map only for the first target (to avoid duplicates)
                        if self.save_carpet_map and target == targets_to_process[0]:
                            diff_img = self.masker.inverse_transform(target_data - deperfusioned_ts)
                            viz.save_bold_comparison_carpet_map(
                                self.masker.inverse_transform(run_data["raw"]), 
                                self.masker.inverse_transform(run_data["cleaned"]), 
                                deperfusioned_img_native, diff_img, self.masker.mask_img_, 
                                f"{output_prefix}_run-{i+1}", target, self.plot_decimation_factor
                            )

                        if self.is_native_space_run:
                            if self.save_debugging_files:
                                nib.save(deperfusioned_img_native, os.path.join(output_dir, f"{run_basename}_space-T1w_deperfusioned-{target}.nii.gz"))
                            final_path = os.path.join(output_dir, f"{run_basename}_space-MNI152NLin2009cAsym_deperfusioned-{target}.nii.gz")
                            io.warp_to_mni_and_save(self, deperfusioned_img_native, final_path, interpolator="bSpline")
                        else:
                            final_path = os.path.join(output_dir, f"{run_basename}_deperfusioned-{target}.nii.gz")
                            nib.save(deperfusioned_img_native, final_path)
                    
                    current_timepoint += run_len

            logger.info("--- Pipeline Finished ---")
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            raise
        finally:
            io.cleanup_staged_files(staging_dir)

