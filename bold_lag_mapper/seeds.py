"""
Seed ROI resolution
-------------------

Lags are measured relative to a seed time series. By default the seed is a deep white-matter mask shipped
with the package, defined in MNI152NLin2009cAsym space on fMRIPrep's 2 mm grid. It is used only when the
input is in that space, or when the input is in the subject's T1w space and processed with --native-space
(the mask is then moved to T1w space with fMRIPrep's MNI152NLin2009cAsym-to-T1w transform). For any other
input the caller must choose: a seed mask in the input's space, or the global mean signal.
"""
import logging
from pathlib import Path

from .io import BIDSPathParser

logger = logging.getLogger(__name__)

BUILTIN_SEED_SPACE = "MNI152NLin2009cAsym"
BUILTIN_SEED_PATH = (Path(__file__).resolve().parent / "data"
                     / "tpl-MNI152NLin2009cAsym_res-02_desc-deepWMseed_mask.nii.gz")


def resolve_seed_roi(value, bold_file, native_space, auto_seed_from_freesurfer=False):
    """Return the path of the seed mask to use, or None to use the global mean signal.

    value: 'builtin' (the bundled deep white-matter mask), 'global' / None (global mean), or a file path.
    """
    if auto_seed_from_freesurfer:
        return None            # the FreeSurfer-derived seed is built later and replaces any other choice
    if value is None or str(value).strip().lower() == "global":
        logger.info("Seed: global mean signal of the analysis mask.")
        return None
    if str(value).strip().lower() == "builtin":
        if not BUILTIN_SEED_PATH.exists():
            raise FileNotFoundError(f"The bundled seed is missing from this installation: {BUILTIN_SEED_PATH}")
        space = BIDSPathParser(bold_file).entities.get("space")
        if native_space or space == BUILTIN_SEED_SPACE:
            logger.info(f"Seed: bundled deep white-matter mask ({BUILTIN_SEED_SPACE}).")
            return str(BUILTIN_SEED_PATH)
        raise ValueError(
            f"The bundled seed is defined in {BUILTIN_SEED_SPACE}, but the input BOLD ({Path(bold_file).name}) "
            f"is in space '{space or 'unknown'}'. Use --native-space for fMRIPrep T1w-space input, or give "
            "--seed-roi-file PATH (a mask in the input's space) or --seed-roi-file global.")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"Seed ROI file not found: {value}")
    return str(path)
