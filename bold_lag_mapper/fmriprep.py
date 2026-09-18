"""
fMRIPrep input discovery
------------------------

Collects one subject's inputs from an fMRIPrep derivatives directory: every preprocessed BOLD run of the
requested space (sorted by path), each paired with the confounds file of the same run, the first run's brain
mask, runs shorter than a minimum length dropped, and a single TR for all runs. Anything ambiguous (several
sessions, tasks or resolutions without a selection) or missing stops with a message instead of being guessed.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np

from .utils import _tr_seconds

logger = logging.getLogger(__name__)

BOLD_SUFFIX = "_desc-preproc_bold.nii.gz"
_ENTITY = re.compile(r"(?:^|_)(sub|ses|task|acq|ce|rec|dir|run|echo|space|cohort|res|den)-([a-zA-Z0-9]+)")


@dataclass
class FmriprepInputs:
    bold_files: List[str]
    confounds_files: List[str]
    mask_file: str
    tr: float
    dropped: List[str] = field(default_factory=list)


def _entities(name):
    return dict(_ENTITY.findall(name))


def discover_fmriprep_inputs(fmriprep_dir, participant_label, session=None, task=None, space="T1w",
                             res=None, runs=None, min_volumes=120):
    """Find one subject's BOLD runs, confounds and mask in an fMRIPrep derivatives directory.

    Args:
        fmriprep_dir: the directory that contains sub-<label>/.
        participant_label: subject label, with or without the 'sub-' prefix.
        session, task, res: entity values to select; required when the subject has several.
        space: 'T1w' or a template name such as 'MNI152NLin2009cAsym'.
        runs: optional list of file-name fragments (e.g. 'run-01', 'dir-AP_run-02'); a run is kept when its
            name contains '_<fragment>_'.
        min_volumes: runs with fewer volumes are dropped (and reported).

    Returns:
        FmriprepInputs with parallel bold_files / confounds_files, the first kept run's mask, the TR and the
        names of dropped runs.
    """
    label = participant_label[4:] if participant_label.startswith("sub-") else participant_label
    subject_dir = Path(fmriprep_dir) / f"sub-{label}"
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"No fMRIPrep subject directory sub-{label} in {fmriprep_dir}")

    candidates = []
    for p in sorted(subject_dir.rglob(f"*{BOLD_SUFFIX}"), key=str):
        ent = _entities(p.name)
        if ent.get("sub") != label or ent.get("space") != space:
            continue
        if session is not None and ent.get("ses") != str(session):
            continue
        if task is not None and ent.get("task") != str(task):
            continue
        if res is not None and ent.get("res") != str(res):
            continue
        if runs and not any(f"_{r}_" in p.name for r in runs):
            continue
        candidates.append((p, ent))
    if not candidates:
        raise FileNotFoundError(
            f"No *_space-{space}*{BOLD_SUFFIX} runs for sub-{label} in {subject_dir} "
            f"(session={session}, task={task}, res={res}, runs={runs}).")

    for key, flag in (("ses", "--session"), ("task", "--task"), ("res", "--res")):
        values = sorted({e[key] for _, e in candidates if key in e})
        if len(values) > 1:
            raise ValueError(f"sub-{label} has several {key} values {values} in space {space}; choose one with {flag}.")

    bolds, confounds, dropped, mask = [], [], [], None
    for p, _ in candidates:
        n_vols = nib.load(str(p)).shape[-1]
        if n_vols < min_volumes:
            dropped.append(p.name)
            logger.warning(f"Run dropped as too short: {p.name} has {n_vols} volumes (< {min_volumes}).")
            continue
        conf = p.with_name(p.name.split("_space-")[0] + "_desc-confounds_timeseries.tsv")
        m = p.with_name(p.name.replace(BOLD_SUFFIX, "_desc-brain_mask.nii.gz"))
        if not conf.exists():
            raise FileNotFoundError(f"No confounds file for {p.name}: expected {conf.name}")
        if not m.exists():
            raise FileNotFoundError(f"No brain mask for {p.name}: expected {m.name}")
        bolds.append(str(p))
        confounds.append(str(conf))
        if mask is None:
            mask = str(m)
    if not bolds:
        raise ValueError(f"Every run of sub-{label} is shorter than {min_volumes} volumes; nothing to map.")

    trs = []
    for b in bolds:
        tr_header = _tr_seconds(nib.load(b), b)
        sidecar = Path(b[: -len(".nii.gz")] + ".json")
        if sidecar.exists():
            rt = json.loads(sidecar.read_text()).get("RepetitionTime")
            if rt is not None and not np.isclose(float(rt), tr_header, atol=1e-3):
                raise ValueError(f"{Path(b).name}: RepetitionTime {rt} s in the JSON sidecar differs from the "
                                 f"NIfTI header TR {tr_header} s.")
        else:
            logger.warning(f"No JSON sidecar for {Path(b).name}; using the NIfTI header TR ({tr_header} s).")
        trs.append(tr_header)
    if not np.allclose(trs, trs[0], atol=1e-3):
        raise ValueError(f"Runs of sub-{label} have different TRs: {trs}")

    logger.info(f"fMRIPrep inputs for sub-{label} (space {space}): {len(bolds)} run(s), TR {trs[0]:g} s, "
                f"{len(dropped)} dropped.")
    for b, c in zip(bolds, confounds):
        logger.info(f"  run {Path(b).name} | confounds {Path(c).name}")
    logger.info(f"  mask {Path(mask).name}")
    return FmriprepInputs(bolds, confounds, mask, float(trs[0]), dropped)
