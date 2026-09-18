"""
Input/Output Module
-------------------

This module is the interface between the lag mapping pipeline and the filesystem.
It encapsulates all logic related to reading data from disk (NIfTI images,
confound files), writing results back to disk (lag maps, deperfusioned BOLD),
and managing file paths, particularly within BIDS-like directory structures.

A key feature is its handling of native space processing, which leverages the
ANTsPy library to apply complex spatial transformations, ensuring that analyses
run in subject-specific space can be correctly interpreted and compared in a
standard template space like MNI. By isolating these file operations, the core
algorithmic code remains clean and focused on the numerical computations.
"""

import gzip
import logging
import os
import re
import shutil
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image as nilearn_image

# Conditional import for ANTsPy. This allows the core package to function
# even if ANTsPy is not installed, as long as native space processing is not requested.
# This prevents installation issues for users who only need standard-space analysis.
try:
    import ants
    _ANTSPY_AVAILABLE = True
except ImportError:
    ants = None
    _ANTSPY_AVAILABLE = False


logger = logging.getLogger(__name__)

# Column conventions for headerless .txt motion files. A .txt file has no names, so the order and the
# rotation units must be declared rather than guessed.
# Values: (canonical column names in file order, rotation units, human description).
TXT_MOTION_SCHEMAS = {
    "spm":  (("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"), "rad",
             "SPM rp_*.txt: translations (mm) then rotations (radians)"),
    "fsl":  (("rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"), "rad",
             "FSL MCFLIRT .par: rotations (radians) then translations (mm)"),
    "hcp":  (("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"), "deg",
             "HCP Movement_Regressors.txt: translations (mm) then rotations (degrees)"),
    "afni": (("rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"), "deg",
             "AFNI 3dvolreg .1D: rotations (degrees) then translations (mm)"),
}


class BIDSPathParser:
    """
    Robust BIDS path parser that handles both FMRIPREP and HCP pipeline outputs.
    """
    
    def __init__(self, bold_file_path):
        self.bold_path = Path(bold_file_path)
        self.entities = self._parse_bids_entities()
        self.pipeline_type = self._detect_pipeline_type()
        
    def _parse_bids_entities(self):
        """
        Parse BIDS entities from filename using robust regex patterns.
        """
        filename = self.bold_path.name
        entities = {}
        
        # Common BIDS entity patterns
        patterns = {
            'sub': r'sub-([a-zA-Z0-9]+)',
            'ses': r'ses-([a-zA-Z0-9]+)',
            'task': r'task-([a-zA-Z0-9]+)',
            'acq': r'acq-([a-zA-Z0-9]+)',
            'ce': r'ce-([a-zA-Z0-9]+)',
            'dir': r'dir-([a-zA-Z0-9]+)',
            'run': r'run-([0-9]+)',
            'echo': r'echo-([0-9]+)',
            'space': r'space-([a-zA-Z0-9]+)',
            'cohort': r'cohort-([a-zA-Z0-9]+)',
            'res': r'res-([a-zA-Z0-9]+)',
            'den': r'den-([a-zA-Z0-9]+)',
            'desc': r'desc-([a-zA-Z0-9]+)',
            'atlas': r'atlas-([a-zA-Z0-9]+)',
        }
        
        for entity, pattern in patterns.items():
            match = re.search(pattern, filename)
            if match:
                entities[entity] = match.group(1)
        
        # Special handling for HCP-style naming
        if 'rfMRI' in filename or 'tfMRI' in filename:
            # HCP naming convention: e.g., rfMRI_REST1_PA, tfMRI_GAMBLING_PA
            hcp_match = re.search(r'([rt]fMRI)_([A-Z0-9_]+)(?:_([A-Z]+))?', filename)
            if hcp_match:
                entities['datatype'] = hcp_match.group(1)
                task_run = hcp_match.group(2)
                if hcp_match.group(3):
                    entities['dir'] = hcp_match.group(3)
                
                # Parse task and run from HCP format
                if 'REST' in task_run:
                    entities['task'] = 'rest'
                    run_match = re.search(r'REST(\d+)', task_run)
                    if run_match:
                        entities['run'] = run_match.group(1)
                else:
                    entities['task'] = task_run.lower()
        
        return entities
    
    def _detect_pipeline_type(self):
        """
        Detect whether this is FMRIPREP or HCP pipeline output.
        """
        path_str = str(self.bold_path)
        
        # HCP indicators
        if any(indicator in path_str for indicator in ['MNINonLinear', 'rfMRI', 'tfMRI', '/Results/']):
            return 'hcp'
        
        # FMRIPREP indicators
        if any(indicator in path_str for indicator in ['derivatives/fmriprep', 'space-', 'desc-preproc']):
            return 'fmriprep'
        
        # Generic BIDS indicators
        if 'derivatives' in path_str:
            return 'derivatives'
        
        return 'unknown'
    
    def get_subject_id(self):
        """Get subject ID with fallback methods."""
        if 'sub' in self.entities:
            return self.entities['sub']
        
        # Fallback: try to extract from path
        for part in self.bold_path.parts:
            if part.startswith('sub-'):
                return part[4:]  # Remove 'sub-' prefix
            if part.startswith('HCP_'):
                return part[4:]  # Remove 'HCP_' prefix
            # For HCP numeric subjects
            if part.isdigit() and len(part) >= 6:
                return part
        
        return None
    
    def get_session_id(self):
        """Get session ID with fallback methods."""
        if 'ses' in self.entities:
            return self.entities['ses']
        
        # Fallback: try to extract from path
        for part in self.bold_path.parts:
            if part.startswith('ses-'):
                return part[4:]  # Remove 'ses-' prefix
        
        return None


def find_derivatives_root(bold_file):
    """
    Find the derivatives root directory from a BOLD file path.
    Works with both FMRIPREP and HCP pipelines.
    """
    path = Path(bold_file)
    
    # For HCP data
    if 'MNINonLinear' in path.parts:
        # Find the subject directory in HCP structure
        for i, part in enumerate(path.parts):
            if part.isdigit() and len(part) >= 6:  # HCP subject ID
                return Path(*path.parts[:i+1])
    
    # BIDS derivatives (fMRIPrep): the directory that contains the sub-<label>/ folder of this file,
    # whatever it is called (derivatives/fmriprep/, derivatives/ itself, or any output directory).
    for i in range(len(path.parts) - 2, 0, -1):
        if path.parts[i].startswith('sub-'):
            return Path(*path.parts[:i])

    # For FMRIPREP/BIDS derivatives
    if 'derivatives' in path.parts:
        deriv_idx = path.parts.index('derivatives')
        return Path(*path.parts[:deriv_idx+2])  # Include pipeline name

    # Fallback: use parent directories
    return path.parent.parent


def _fmriprep_anat_candidates(derivatives_root, subject_id, session_id):
    """(anat_dir, file_prefix) pairs to search, in order. fMRIPrep writes the anatomical outputs of a
    single-session subject under sub-<label>/ses-<label>/anat and those of a multi-session subject (one
    anatomical template for all sessions) under sub-<label>/anat, so both are tried."""
    sub = derivatives_root / f"sub-{subject_id}"
    pairs = []
    if session_id:
        pairs.append((sub / f"ses-{session_id}" / "anat", f"sub-{subject_id}_ses-{session_id}"))
    pairs.append((sub / "anat", f"sub-{subject_id}"))
    return pairs


def find_anatomical_files(bold_file, space="MNI152NLin2009cAsym", file_type="T1w", label=None):
    """
    Find corresponding anatomical files with robust BIDS/HCP support.
    """
    # `label` restricts the match to one tissue class. Without it, file_type="probseg" would match
    # label-CSF / label-GM / label-WM alike and return whichever came first, so --refine-with-wm-mask
    # could intersect the seed with a CSF map and erase it.
    parser = BIDSPathParser(bold_file)
    subject_id = parser.get_subject_id()
    session_id = parser.get_session_id()
    
    if not subject_id:
        logger.warning(f"Could not extract subject ID from {bold_file}")
        return None
    
    derivatives_root = find_derivatives_root(bold_file)
    
    # Search patterns based on pipeline type
    search_patterns = []
    
    if parser.pipeline_type == 'hcp':
        # HCP structure: subject/MNINonLinear/xfms/ or T1w/
        hcp_base = derivatives_root
        if space == "native" or space == "T1w":
            search_patterns.extend([
                hcp_base / "T1w" / f"T1w_acpc_dc_restore.nii.gz",
                hcp_base / "T1w" / f"{subject_id}_T1w.nii.gz"
            ])
        else:
            search_patterns.extend([
                hcp_base / "MNINonLinear" / f"T1w_restore.nii.gz",
                hcp_base / "MNINonLinear" / f"{subject_id}_T1w.nii.gz"
            ])
    
    elif parser.pipeline_type == 'fmriprep':
        # FMRIPREP structure: session-level anat first, then subject-level anat
        for anat_dir, file_prefix in _fmriprep_anat_candidates(derivatives_root, subject_id, session_id):
            if space == "native" or space == "T1w":
                search_patterns.extend([
                    anat_dir / f"{file_prefix}_desc-preproc_{file_type}.nii.gz",
                    anat_dir / f"{file_prefix}_{file_type}.nii.gz",
                    # fMRIPrep may add a run- entity to anatomical names
                    # (sub-X_run-01_desc-preproc_T1w). The glob is anchored to this
                    # subject's anat_dir so extra entities are tolerated.
                    anat_dir / f"{file_prefix}_*desc-preproc_{file_type}.nii.gz",
                    anat_dir / f"{file_prefix}_*_{file_type}.nii.gz",
                ])
            else:
                search_patterns.extend([
                    anat_dir / f"{file_prefix}_space-{space}_desc-preproc_{file_type}.nii.gz",
                    anat_dir / f"{file_prefix}_space-{space}_{file_type}.nii.gz",
                    # FMRIPREP may include res-* entity (e.g. res-2)
                    anat_dir / f"{file_prefix}_space-{space}_res-*_desc-preproc_{file_type}.nii.gz",
                    # a run- entity, when present, precedes space- in fMRIPrep filenames
                    anat_dir / f"{file_prefix}_*space-{space}_res-*_desc-preproc_{file_type}.nii.gz",
                    anat_dir / f"{file_prefix}_*space-{space}*_{file_type}.nii.gz",
                ])
    
    else:
        # Generic BIDS fallback
        subject_dir = derivatives_root / f"sub-{subject_id}"
        if session_id:
            subject_dir = subject_dir / f"ses-{session_id}"
        
        anat_dir = subject_dir / "anat"
        search_patterns.extend([
            anat_dir / f"*space-{space}*{file_type}.nii.gz",
            anat_dir / f"*{file_type}.nii.gz"
        ])
    
    def _label_ok(p):
        return label is None or f"label-{label}_" in os.path.basename(str(p))

    # Search for files
    for pattern in search_patterns:
        if pattern.exists() and _label_ok(pattern):
            logger.info(f"Found anatomical file: {pattern}")
            return str(pattern)
        
        # Try glob pattern if exact match fails
        if '*' in str(pattern):
            # Sorted, so the choice is deterministic across re-runs when several files match
            # (run- entities), as in find_transform_files.
            matches = sorted(pattern.parent.glob(pattern.name))
            matches = [m for m in matches if _label_ok(m)]
            if matches:
                logger.info(f"Found anatomical file via glob: {matches[0]}")
                return str(matches[0])
    
    logger.warning(f"Could not find {file_type} in space {space} for subject {subject_id}")
    return None


def find_transform_files(bold_file):
    """
    Find transformation files for native space processing.
    Supports both FMRIPREP and HCP pipelines.
    """
    parser = BIDSPathParser(bold_file)
    subject_id = parser.get_subject_id()
    session_id = parser.get_session_id()

    if not subject_id:
        raise FileNotFoundError(f"Could not extract subject ID from {bold_file}")
    
    derivatives_root = find_derivatives_root(bold_file)
    transforms = {}
    
    if parser.pipeline_type == 'hcp':
        # HCP transform structure
        xfms_dir = derivatives_root / "MNINonLinear" / "xfms"
        
        # HCP uses different transform naming
        transforms.update({
            'to_mni': xfms_dir / "acpc_dc2standard.nii.gz",
            'from_mni': xfms_dir / "standard2acpc_dc.nii.gz",
            'to_mni_warp': xfms_dir / f"{subject_id}_T1w_space-MNI152NLin2009cAsym_warp.nii.gz"
        })
        
        # Alternative HCP patterns
        alt_patterns = [
            xfms_dir / "T1w2MNI152NLin2009cAsym.nii.gz",
            xfms_dir / "MNI152NLin2009cAsym2T1w.nii.gz"
        ]
        
        for pattern in alt_patterns:
            if pattern.exists():
                if "T1w2MNI" in pattern.name:
                    transforms['to_mni'] = pattern
                elif "MNI2T1w" in pattern.name:
                    transforms['from_mni'] = pattern
    
    elif parser.pipeline_type == 'fmriprep':
        # FMRIPREP transform structure: session-level anat first, then subject-level anat. Only the
        # MNI152NLin2009cAsym <-> T1w pair is accepted: the bundled seed and the outputs are defined in
        # that space, so a transform to any other template must never be substituted.
        searched = []
        for anat_dir, file_prefix in _fmriprep_anat_candidates(derivatives_root, subject_id, session_id):
            searched.append(str(anat_dir))
            found = {}
            transform_patterns = {
                'to_mni': anat_dir / f"{file_prefix}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
                'from_mni': anat_dir / f"{file_prefix}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
            }
            # fMRIPrep may add a run- entity to transform names. The glob is anchored to THIS
            # subject's anat_dir with the exact directional name (run-tolerant), so another
            # subject's transform can never match.
            glob_patterns = {
                'to_mni': f"{file_prefix}_*from-T1w_to-MNI152NLin2009cAsym*xfm.h5",
                'from_mni': f"{file_prefix}_*from-MNI152NLin2009cAsym_to-T1w*xfm.h5",
            }
            for key, pattern in transform_patterns.items():
                if pattern.exists():
                    found[key] = pattern
                else:
                    alternatives = sorted(anat_dir.glob(glob_patterns[key]))
                    if alternatives:
                        found[key] = alternatives[0]
            if len(found) == 2:
                transforms.update(found)
                break
        if len(transforms) < 2:
            logger.error("fMRIPrep MNI152NLin2009cAsym <-> T1w transforms (*_from-T1w_to-MNI152NLin2009cAsym*_xfm.h5 "
                         f"and *_from-MNI152NLin2009cAsym_to-T1w*_xfm.h5) not found in {searched}. Run fMRIPrep with "
                         "MNI152NLin2009cAsym among --output-spaces, or process the MNI-space runs without --native-space.")
        return transforms

    # Validate that we found the necessary transforms
    required_transforms = ['to_mni', 'from_mni']
    missing = [t for t in required_transforms if t not in transforms]

    if missing:
        # Last-resort search (non-fMRIPrep layouts only; fMRIPrep returns above).
        search_dir = derivatives_root
        available_files = list(search_dir.rglob("*xfm*")) + list(search_dir.rglob("*warp*"))
        logger.warning(f"Missing transforms: {missing}")
        logger.info(f"Available transform files: {[str(f) for f in available_files[:10]]}")
        
        # Try to find any transform files as fallback
        for transform_file in available_files:
            if any(pattern in transform_file.name.lower() for pattern in ['t1w2mni', 'to-mni', 'native2mni']):
                transforms['to_mni'] = transform_file
            elif any(pattern in transform_file.name.lower() for pattern in ['mni2t1w', 'to-t1w', 'mni2native']):
                transforms['from_mni'] = transform_file
    
    return transforms


def find_freesurfer_files(bold_file):
    """
    Find FreeSurfer segmentation files with support for different directory structures.
    """
    parser = BIDSPathParser(bold_file)
    subject_id = parser.get_subject_id()
    
    if not subject_id:
        return {}
    
    derivatives_root = find_derivatives_root(bold_file)
    fs_files = {}
    
    # Search patterns for different pipeline types
    search_locations = []
    
    if parser.pipeline_type == 'hcp':
        # HCP FreeSurfer structure
        search_locations.extend([
            derivatives_root / "T1w" / f"{subject_id}",
            derivatives_root.parent / "freesurfer" / subject_id,
            derivatives_root / "freesurfer" / subject_id
        ])
    
    elif parser.pipeline_type == 'fmriprep':
        # FMRIPREP typically has FreeSurfer in parallel derivatives
        fmriprep_root = derivatives_root.parent
        search_locations.extend([
            fmriprep_root / "freesurfer" / f"sub-{subject_id}",
            fmriprep_root / "freesurfer" / subject_id,
            derivatives_root.parent.parent / "freesurfer" / f"sub-{subject_id}"
        ])
    
    # Generic search patterns
    search_locations.extend([
        derivatives_root.parent / "freesurfer" / subject_id,
        derivatives_root / "freesurfer" / subject_id
    ])
    
    # File patterns to search for
    file_patterns = {
        'aparc_aseg': ['mri/aparc+aseg.mgz', 'mri/aparc.a2009s+aseg.mgz'],
        'brain_mask': ['mri/brainmask.mgz', 'mri/brain.mgz'],
        'wm_mask': ['mri/wm.mgz'],
        'aseg': ['mri/aseg.mgz']
    }
    
    for location in search_locations:
        if not location.exists():
            continue
            
        logger.info(f"Searching FreeSurfer files in: {location}")
        
        for file_type, patterns in file_patterns.items():
            for pattern in patterns:
                file_path = location / pattern
                if file_path.exists():
                    fs_files[file_type] = file_path
                    logger.info(f"Found {file_type}: {file_path}")
                    break
            
            if file_type in fs_files:
                break
    
    return fs_files


def count_leading_non_steady_state(confounds_file, num_timepoints):
    """Number of LEADING volumes fMRIPrep flagged as non-steady-state in a confounds .tsv.

    fMRIPrep writes one one-hot column per flagged volume (non_steady_state_outlier_00, _01, ...),
    always for the first volumes of a run. Exactly that leading contiguous block is dropped from BOLD
    and confounds alike. A flag outside the leading block is reported and left alone (it is not a
    dummy scan). Returns 0 for .txt confounds, for
    files without such columns, and when trimming is disabled by the caller. Raises if the file
    length does not match the BOLD, so a mispaired confounds file cannot decide how many volumes
    to drop.
    """
    if confounds_file is None or not str(confounds_file).endswith(".tsv"):
        return 0
    df = pd.read_csv(confounds_file, sep="\t")
    cols = [c for c in df.columns if c.startswith("non_steady_state_outlier")]
    if not cols:
        return 0
    if len(df) != num_timepoints:
        raise ValueError(
            f"{confounds_file} has {len(df)} rows but the BOLD run has {num_timepoints} volumes; "
            "refusing to count non-steady-state volumes from a mispaired file.")
    flagged = (df[cols].fillna(0).to_numpy() > 0).any(axis=1)
    n = 0
    while n < len(flagged) and flagged[n]:
        n += 1
    stray = int(flagged[n:].sum())
    if stray:
        logger.warning(f"{stray} non_steady_state_outlier flag(s) lie outside the leading block in "
                       f"{os.path.basename(confounds_file)}; only the leading {n} volume(s) are trimmed.")
    logger.info(f"Non-steady-state columns {cols}: leading block = {n} volume(s).")
    return n


def load_framewise_displacement(confounds_file, num_timepoints, skip_leading=0):
    """fMRIPrep's `framewise_displacement` column (Power FD, mm; the first frame's NaN becomes 0),
    with the same leading rows dropped as the BOLD. Returns None when the file is not a .tsv or
    has no such column; raises when the length does not match the run (used by the FD spike rule)."""
    if confounds_file is None or not str(confounds_file).endswith(".tsv"):
        return None
    header = pd.read_csv(confounds_file, sep="\t", nrows=0).columns
    if "framewise_displacement" not in header:
        return None
    fd = pd.read_csv(confounds_file, sep="\t", usecols=["framewise_displacement"])["framewise_displacement"].to_numpy(dtype=float)
    fd = np.nan_to_num(fd[skip_leading:])
    if fd.shape[0] != num_timepoints:
        raise ValueError(f"framewise_displacement in {confounds_file} has {fd.shape[0]} rows after trimming "
                         f"but the run has {num_timepoints} volumes.")
    return fd


def load_motion_confounds(motion_confounds_file, num_timepoints, apply_motion_correction,
                          custom_motion_regressors, skip_leading=0, txt_motion_schema="spm"):
    """
    Loads motion confounds from a single specified file, supporting both FMRIPREP-style
    .tsv files and traditional AFNI/FSL-style .txt files.

    Args:
        motion_confounds_file (str): Path to the confounds file.
        num_timepoints (int): Number of timepoints in the corresponding BOLD run AFTER any
                              non-steady-state trimming, for validation.
        apply_motion_correction (bool): A flag to enable or disable confound loading.
        custom_motion_regressors (list[str] or None): Specific regressors to load from a .tsv file.
                                                     If None, defaults to standard 6 motion parameters.
        skip_leading (int): Rows to drop from the top, matching the leading non-steady-state
                            volumes dropped from the BOLD. Default 0.

    Returns:
        tuple[np.ndarray or None, list[str]]: A tuple containing the confounds matrix (time x regressors)
                                              and a list of their names. Returns (None, []) if disabled.
    """
    # If motion correction is disabled or no file is provided, exit early.
    if not apply_motion_correction or motion_confounds_file is None:
        return None, []

    logger.info(f"Loading motion confounds from: {motion_confounds_file}")
    
    # Handle FMRIPREP-style tab-separated value (.tsv) files
    if motion_confounds_file.endswith(".tsv"):
        confounds_df = pd.read_csv(motion_confounds_file, sep="\t")
        # Use user-specified columns or default to the 6 standard motion parameters
        motion_cols_to_use = custom_motion_regressors or ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
        # Filter to only include columns that actually exist in the file
        motion_cols = [col for col in motion_cols_to_use if col in confounds_df.columns]
        # A confounds file with renamed columns (a different fMRIPrep major version, or a
        # --custom-motion-regressors typo) must stop the run: continuing would silently drop ALL
        # motion regression and leave spike regressors only.
        if not motion_cols:
            raise ValueError(
                f"None of the requested motion regressors {motion_cols_to_use} exist in "
                f"{motion_confounds_file}. Columns starting rot_/trans_ that ARE present: "
                f"{[c for c in confounds_df.columns if c.startswith(('rot_', 'trans_'))][:12]}. "
                "Refusing to continue with no motion regression.")
        if len(motion_cols) != len(motion_cols_to_use):
            missing = [c for c in motion_cols_to_use if c not in motion_cols]
            raise ValueError(
                f"Motion regressors {missing} are missing from {motion_confounds_file}; a partial "
                "motion model would otherwise be applied silently. Refusing to continue.")
        motion_confounds_np = confounds_df[motion_cols].values
        motion_names = motion_cols

    # Handle legacy space-delimited text files (.txt)
    elif motion_confounds_file.endswith(".txt"):
        # A .txt file carries no column names, so the ORDER and the ROTATION UNITS have to be
        # declared, not guessed: the schema below maps each convention to canonical fMRIPrep-style
        # names and converts degrees to radians, which is what the FD formula (radius 50 mm) assumes.
        motion_confounds_np = pd.read_csv(motion_confounds_file, sep=r"\s+", header=None).values
        n_found = motion_confounds_np.shape[1]
        # HCP writes 12 columns: the 6 rigid-body parameters followed by their 6 temporal
        # derivatives (Movement_Regressors.txt, _dt.txt, _demean.txt). The mapper computes its
        # own derivatives in lag_preprocess.calculate_motion_derivatives_and_fd, so only the first
        # six are read - but the file must be one of the shapes understood here, not whatever
        # happens to be there.
        if n_found not in (6, 12):
            raise ValueError(
                f"{motion_confounds_file} has {n_found} column(s); a headerless motion file must have "
                "6 (rigid-body only) or 12 (rigid-body + derivatives, the HCP layout) columns.")
        if n_found == 12:
            logger.info(f"{os.path.basename(motion_confounds_file)}: 12 columns - reading the first 6 "
                        "rigid-body parameters; the 6 derivative columns are recomputed by this "
                        "pipeline and are not used.")
        motion_confounds_np = motion_confounds_np[:, :6].astype(float)
        schema = (txt_motion_schema or "spm").lower()
        if schema not in TXT_MOTION_SCHEMAS:
            raise ValueError(f"Unknown .txt motion schema '{schema}'. "
                             f"Choose one of: {', '.join(sorted(TXT_MOTION_SCHEMAS))}.")
        order, rot_units, description = TXT_MOTION_SCHEMAS[schema]
        motion_names = list(order)
        rot_cols = [i for i, n in enumerate(motion_names) if n.startswith("rot_")]
        if rot_units == "deg":
            motion_confounds_np[:, rot_cols] = np.deg2rad(motion_confounds_np[:, rot_cols])
        logger.warning(
            f".txt motion file read with the '{schema}' schema ({description}); columns named "
            f"{motion_names}, rotations in {rot_units} -> radians. A .txt file carries no column "
            "names: if this convention is wrong for your data the FD values and the motion "
            "regressors will be wrong. Set --txt-motion-schema explicitly.")
        if custom_motion_regressors:
            logger.warning("--custom_motion_regressors is ignored for .txt input; using the 6 "
                           "rigid-body columns of the declared schema.")
    else:
        raise ValueError(f"Unsupported motion file format: {motion_confounds_file}.")

    # Drop the same leading non-steady-state rows that were dropped from the BOLD
    if skip_leading:
        if skip_leading >= motion_confounds_np.shape[0]:
            raise ValueError(f"skip_leading={skip_leading} would remove every row of {motion_confounds_file}.")
        motion_confounds_np = motion_confounds_np[skip_leading:]
        logger.info(f"Dropped the first {skip_leading} confounds row(s) to match the trimmed BOLD.")

    # Replace any potential NaN values (e.g., from FMRIPREP at the beginning of derivatives) with 0
    motion_confounds_np[np.isnan(motion_confounds_np)] = 0

    # Validate that the number of timepoints in the confounds matches the BOLD data
    if motion_confounds_np.shape[0] != num_timepoints:
        raise ValueError(
            f"Mismatch between BOLD timepoints ({num_timepoints}) and motion confound timepoints ({motion_confounds_np.shape[0]})."
        )
    return motion_confounds_np, motion_names


def find_anatomical_image(bold_file, space="MNI152NLin2009cAsym"):
    """
    Finds a corresponding preprocessed T1w image, with fallbacks for non-FMRIPREP data.
    """
    return find_anatomical_files(bold_file, space=space, file_type="T1w")


def setup_native_space_processing(mapper, bold_file, mask_file, output_dir):
    """
    Prepares for native space processing by finding required transformation files
    and warping the MNI-space seed ROI into the subject's native BOLD space.

    Args:
        mapper (BOLDLagMapper): The main mapper instance to update its state.
        bold_file (str): Path to the native-space BOLD file.
        mask_file (str): Path to the native-space mask file.
        output_dir (str): Directory for saving the warped seed and other intermediate files.
    """
    logger.info("--- Setting up for Native Space Processing ---")
    if not _ANTSPY_AVAILABLE:
        raise ImportError("Native space processing requires 'antspyx'. Please install it (`pip install antspyx`).")

    parser = BIDSPathParser(bold_file)
    subject_id = parser.get_subject_id()
    
    if not subject_id:
        raise ValueError(f"Could not extract subject ID from {bold_file}")

    # Find transformation files
    transforms = find_transform_files(bold_file)
    
    if 'from_mni' not in transforms:
        raise FileNotFoundError(f"Could not find MNI-to-native transform for subject {subject_id}")
    
    if 'to_mni' not in transforms:
        raise FileNotFoundError(f"Could not find native-to-MNI transform for subject {subject_id}")

    mapper.transform_to_native_path = transforms['from_mni']
    mapper.transform_to_mni_path = transforms['to_mni']
    mapper.is_native_space_run = True

    logger.info(f"Found To-Native transform: {mapper.transform_to_native_path}")
    logger.info(f"Found To-MNI transform: {mapper.transform_to_mni_path}")

    # If a seed ROI was provided (which is in MNI space by default), warp it to native space
    if mapper.seed_roi_file:
        logger.info(f"Warping seed ROI '{mapper.seed_roi_file}' to native space...")
        native_t1w_path = find_anatomical_files(bold_file, space="native", file_type="T1w")
        
        if not native_t1w_path:
            raise FileNotFoundError("Could not find native T1w image to use as warping reference.")

        # Load images using ANTsPy for transformation
        native_ref_ants = ants.image_read(native_t1w_path)
        seed_roi_ants = ants.image_read(mapper.seed_roi_file)

        # Apply the transform from MNI to native T1w space using nearest neighbor to preserve binary mask
        warped_seed_t1w_space = ants.apply_transforms(
            fixed=native_ref_ants, moving=seed_roi_ants,
            transformlist=[str(mapper.transform_to_native_path)], interpolator="nearestNeighbor"
        )
        
        # The seed is now in native T1w space, but we need it in native BOLD space.
        # This requires resampling to match the BOLD grid (affine, dimensions, voxel size).
        temp_warped_seed_path = Path(output_dir) / "temp_warped_seed_for_resample.nii.gz"
        try:
            ants.image_write(warped_seed_t1w_space, str(temp_warped_seed_path))
            warped_seed_t1w_nib = nib.load(temp_warped_seed_path)
            bold_ref_nib = nib.load(bold_file)
            warped_seed_final = nilearn_image.resample_to_img(
                source_img=warped_seed_t1w_nib, target_img=bold_ref_nib,
                interpolation="nearest", force_resample=True, copy_header=True
            )
        finally:
            if temp_warped_seed_path.exists():
                os.remove(temp_warped_seed_path)

        final_seed_to_save = None
        # Optionally refine the warped seed using the subject's white matter mask for a more precise seed
        if mapper.refine_with_wm_mask:
            # Try to find WM probability map
            wm_files = find_anatomical_files(bold_file, space="native", file_type="probseg", label="WM")
            if wm_files:
                logger.info(f"Refining seed mask with WM probability: {wm_files}")
                wm_probseg_nib = nib.load(wm_files)
                wm_probseg_resampled = nilearn_image.resample_to_img(
                    source_img=wm_probseg_nib, target_img=bold_ref_nib,
                    interpolation="continuous", force_resample=True, copy_header=True
                )
                # Multiply the warped seed by the WM probability and threshold
                refined_prob_data = warped_seed_final.get_fdata() * wm_probseg_resampled.get_fdata()
                binary_data = (refined_prob_data > 0.5).astype(np.uint8)
                final_seed_to_save = nib.Nifti1Image(binary_data, warped_seed_final.affine, warped_seed_final.header)
            else:
                logger.warning("WM probabilistic segmentation not found. Using standard binarization.")

        if final_seed_to_save is None:
            binary_data = (warped_seed_final.get_fdata() > 0.5).astype(np.uint8)
            final_seed_to_save = nib.Nifti1Image(binary_data, warped_seed_final.affine, warped_seed_final.header)

        # Save the final, native-space seed and update the mapper to use this new file path
        warped_seed_path = Path(output_dir) / f"sub-{subject_id}_space-T1w_desc-warpedseed_mask.nii.gz"
        nib.save(final_seed_to_save, warped_seed_path)
        logger.info(f"Warped seed saved to: {warped_seed_path}")
        mapper.seed_roi_file = str(warped_seed_path)

    # Load the MNI template once to use as a reference for warping results back to MNI space
    mni_template_nib = datasets.load_mni152_template(resolution=2)
    temp_mni_path = Path(output_dir) / "temp_mni_template.nii.gz"
    try:
        nib.save(mni_template_nib, temp_mni_path)
        mapper.mni_template_ants = ants.image_read(str(temp_mni_path))
    finally:
        if temp_mni_path.exists():
            os.remove(temp_mni_path)


def warp_to_mni_and_save(mapper, native_img_nib, output_path, interpolator="linear"):
    """
    Warps a native space NIfTI image to MNI space using the pre-found transform and saves it.

    Args:
        mapper (BOLDLagMapper): The main mapper instance containing transform paths.
        native_img_nib (Nifti1Image): The native-space image to warp.
        output_path (str): The path to save the warped MNI-space image.
        interpolator (str): The interpolation method for ANTs ('linear', 'nearestNeighbor', 'bSpline').

    Returns:
        Nifti1Image: The loaded warped image from disk.
    """
    logger.info(f"Warping '{Path(output_path).name}' to MNI space using '{interpolator}' interpolation...")
    # Use a temporary file to pass the Nibabel object to ANTsPy
    temp_native_path = Path(output_path).parent / f"temp_native_{Path(output_path).name}"
    try:
        nib.save(native_img_nib, temp_native_path)
        native_img_ants = ants.image_read(str(temp_native_path))
        # ANTs needs to know if the image is a vector (4D BOLD) or scalar (3D map)
        imagetype = 3 if native_img_nib.ndim == 4 else 0
        
        # Apply the transform from native T1w to MNI space
        warped_img_ants = ants.apply_transforms(
            fixed=mapper.mni_template_ants, moving=native_img_ants,
            transformlist=[str(mapper.transform_to_mni_path)],
            interpolator=interpolator, imagetype=imagetype
        )
        ants.image_write(warped_img_ants, output_path)
    finally:
        if temp_native_path.exists():
            os.remove(temp_native_path)
            
    logger.info(f"Saved MNI-space image to: {output_path}")
    return nib.load(output_path)


def create_cerebrum_mask_from_freesurfer(bold_ref_file, output_dir):
    """
    Creates a robust cerebral GM+WM seed mask from FreeSurfer segmentation.
    Works with both FMRIPREP and HCP pipeline outputs.
    """
    logger.info("Creating cerebrum mask from FreeSurfer segmentation...")
    
    # Find FreeSurfer files
    fs_files = find_freesurfer_files(bold_ref_file)
    
    if 'aparc_aseg' not in fs_files:
        raise FileNotFoundError("Could not find FreeSurfer aparc+aseg segmentation file")
    
    parser = BIDSPathParser(bold_ref_file)
    subject_id = parser.get_subject_id()
    
    # Load the segmentation
    seg_path = fs_files['aparc_aseg']
    logger.info(f"Loading FreeSurfer segmentation: {seg_path}")
    
    try:
        if str(seg_path).endswith('.mgz'):
            # Use nibabel to load MGZ files
            seg_img = nib.load(seg_path)
        else:
            seg_img = nib.load(seg_path)
        
        seg_data = seg_img.get_fdata()
        
        # Define cerebral GM and WM labels (excluding brainstem, cerebellum)
        cerebral_labels = {
            # Cortical GM
            'cortical_gm': list(range(1000, 1036)) + list(range(2000, 2036)),
            # Subcortical GM
            'subcortical_gm': [10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58],
            # White matter
            'wm': [2, 41, 77, 251, 252, 253, 254, 255]
        }
        
        # Create combined mask
        cerebral_mask = np.zeros_like(seg_data, dtype=bool)
        
        for region, labels in cerebral_labels.items():
            for label in labels:
                cerebral_mask |= (seg_data == label)
        
        # Convert to MNI space if needed
        mni_template = datasets.load_mni152_template(resolution=2)
        
        # Resample to MNI space and BOLD resolution
        cerebral_img = nib.Nifti1Image(cerebral_mask.astype(np.uint8), seg_img.affine, seg_img.header)
        
        # Resample to MNI template space
        cerebral_mni = nilearn_image.resample_to_img(
            cerebral_img, mni_template, interpolation='nearest'
        )
        
        # Save the mask
        mask_filename = f"sub-{subject_id}_space-MNI152NLin2009cAsym_desc-cerebrum_mask.nii.gz"
        mask_path = Path(output_dir) / mask_filename
        nib.save(cerebral_mni, mask_path)
        
        logger.info(f"Created cerebrum seed mask: {mask_path}")
        return str(mask_path)
        
    except Exception as e:
        logger.error(f"Failed to create cerebrum mask from FreeSurfer: {e}")
        raise


def copy_nifti_affine_and_header(source_img, target_img):
    """
    Copies the affine transformation and qform/sform codes from a source to a target NIfTI image.
    This is crucial for ensuring that derived images (like lag maps) have the correct spatial
    orientation and position information as the original BOLD data. Without this, the maps
    would not align correctly in viewers like FSLeyes.

    Args:
        source_img (Nifti1Image): The source image with the correct header (e.g., the original BOLD).
        target_img (Nifti1Image): The target image to be modified (e.g., the generated lag map).

    Returns:
        Nifti1Image: The modified target image with the updated header.
    """
    # Extract the qform/sform codes which define how the affine is interpreted by software
    qform_code = int(source_img.header["qform_code"])
    sform_code = int(source_img.header["sform_code"])
    
    # Set the affine transformations on the target image's header, preserving the codes
    target_img.header.set_qform(source_img.get_qform(), code=qform_code)
    target_img.header.set_sform(source_img.get_sform(), code=sform_code)
    
    return target_img


def stage_to_local(nifti_path, staging_dir, size_threshold_mb=200):
    """
    Stage a large .nii.gz file to a local directory for faster loading.

    For files above the size threshold, copies the .nii.gz to a local staging
    directory and decompresses it to .nii for fast memory-mapped loading.
    Files below the threshold or already .nii are returned unchanged.

    Args:
        nifti_path (str): Path to the original NIfTI file.
        staging_dir (str): Path to the local staging directory.
        size_threshold_mb (float): Size threshold in MB. Files smaller than this
            are not staged.

    Returns:
        str: Path to the file to load (either original or staged/decompressed).
    """
    nifti_path = str(nifti_path)

    # Skip staging for uncompressed files
    if not nifti_path.endswith('.nii.gz'):
        logger.info(f"  File is not .nii.gz, skipping staging: {os.path.basename(nifti_path)}")
        return nifti_path

    # Skip staging for small files
    file_size_mb = os.path.getsize(nifti_path) / (1024 * 1024)
    if file_size_mb < size_threshold_mb:
        logger.info(f"  File size {file_size_mb:.0f} MB < {size_threshold_mb:.0f} MB threshold, skipping staging.")
        return nifti_path

    logger.info(f"  Staging large file ({file_size_mb:.0f} MB) to local disk for faster loading...")

    # Create staging directory
    os.makedirs(staging_dir, exist_ok=True)

    # Copy .nii.gz to staging directory (fast binary copy)
    staged_gz_path = os.path.join(staging_dir, os.path.basename(nifti_path))
    logger.info(f"  Copying to local staging: {staged_gz_path}")
    shutil.copy2(nifti_path, staged_gz_path)

    # Stream-decompress locally to .nii (avoids loading entire array into memory)
    staged_nii_path = staged_gz_path[:-3]  # Remove '.gz'
    logger.info(f"  Decompressing locally to: {staged_nii_path}")
    with gzip.open(staged_gz_path, 'rb') as f_in:
        with open(staged_nii_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Remove the staged .gz copy to save space
    os.remove(staged_gz_path)

    staged_size_gb = os.path.getsize(staged_nii_path) / (1024 ** 3)
    logger.info(f"  Staging complete. Decompressed size: {staged_size_gb:.2f} GB")
    return staged_nii_path


def cleanup_staged_files(staging_dir):
    """
    Remove the per-run staging directory created by this process.

    This is called from a `finally` block, so it also runs when the pipeline fails before staging
    anything. It refuses to delete a directory that this process did not create: the caller
    (core.process_runs) always appends a `bold_lag_mapper_staging_<pid>` child to whatever
    `--local-staging-dir` was given, and only such a child may be removed.

    Args:
        staging_dir (str or None): Path to the per-run staging directory. If None, does nothing.
    """
    if staging_dir is None:
        return

    base = os.path.basename(os.path.normpath(staging_dir))
    if not base.startswith("bold_lag_mapper_staging_"):
        logger.warning(f"Refusing to remove '{staging_dir}': not a directory created by this run "
                       "(expected a 'bold_lag_mapper_staging_<pid>' child). Nothing was deleted.")
        return

    if os.path.isdir(staging_dir):
        file_count = len(os.listdir(staging_dir))
        logger.info(f"Cleaning up staging directory: {staging_dir} ({file_count} files)")
        shutil.rmtree(staging_dir, ignore_errors=True)
        logger.info("Staging directory cleaned up.")


def write_stats_sidecar(path, stats):
    """Write the run's numeric facts as a JSON sidecar.

    Downstream reports can read this file instead of parsing logs or retyping numbers.
    numpy scalars/arrays are converted; anything else unknown is stringified rather than dropped.
    """
    import json

    def _default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (Path,)):
            return str(o)
        return str(o)

    with open(path, "w") as f:
        json.dump(stats, f, indent=1, default=_default)
    logger.info(f"Stats sidecar written: {path}")


def write_seeds_npz(path, seed_time_series, tr_track, run_lengths_track):
    """Seed (sLFO) time course of every tracking step, in the tracking-step sampling.

    Keys: lag_<k> for k >= 0 and lag_m<k> for k < 0 (k in tracking steps, as the estimators key
    mapper.seed_time_series), plus tr_track (s) and run_lengths_track (frames per run at that step).
    Read by the sLFO rainbow plot and usable for boundary-seed similarity checks.
    """
    arrays = {("lag_%d" % k if k >= 0 else "lag_m%d" % -k): np.asarray(v, np.float32)
              for k, v in seed_time_series.items()}
    np.savez(path, tr_track=np.float64(tr_track),
             run_lengths_track=np.asarray(run_lengths_track, np.int64), **arrays)
    logger.info(f"Seeds written: {path} ({len(arrays)} lag steps)")
