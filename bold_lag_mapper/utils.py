"""
Utilities Module
----------------

This module contains various utility and helper functions that support the main
lag mapping pipeline. These functions are generally small, self-contained, and
perform common tasks like input validation, GPU memory management, and data
type conversions between NumPy and CuPy arrays. This modular approach helps keep
the core processing scripts clean and focused on algorithmic logic.
"""

import logging
import os
import nibabel as nib
import numpy as np

# Conditional import for GPU (CuPy). This allows the package to run in a CPU-only
# environment without CuPy installed. A global flag `_CUPY_AVAILABLE` tracks its presence.
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    # If CuPy is not found, we make `cp` an alias for `np`. This allows for a unified
    # API (`xp = cp if use_gpu else np`) throughout the code, where `xp` can be used
    # for array operations regardless of the backend.
    cp = np
    _CUPY_AVAILABLE = False


logger = logging.getLogger(__name__)


def _tr_seconds(img, path):
    """
    TR of a 4D NIfTI in SECONDS.

    `header.get_zooms()[-1]` is in the file's own time unit, which NIfTI records in xyzt_units. A TR stored
    in msec must not be used as seconds: it would mis-set the filter widths, the lag search window and the
    TR >= --subtr-min-tr gate without any error.
    """
    tr = float(img.header.get_zooms()[-1])
    unit = img.header.get_xyzt_units()[1]        # '' when unset
    factors = {"sec": 1.0, "msec": 1e-3, "usec": 1e-6}
    if unit in factors:
        tr *= factors[unit]
    elif unit in ("", "unknown", None):
        logger.warning(f"{os.path.basename(path)}: NIfTI time unit is unset; assuming the TR "
                       f"({tr}) is in SECONDS.")
    else:
        raise ValueError(f"{os.path.basename(path)}: unsupported NIfTI time unit '{unit}'. "
                         "Expected sec, msec or usec.")
    return tr


def validate_inputs(bold_files, mask_file, generate_mask):
    """
    Validates the existence and format of input files and ensures that all provided
    BOLD runs have consistent spatial properties (affine, dimensions) and temporal
    properties (TR). This is a critical first step to prevent errors during processing.

    Args:
        bold_files (list[str]): A list of paths to BOLD NIfTI files.
        mask_file (str): Path to a brain mask NIfTI file.
        generate_mask (bool): Flag indicating if the mask is being generated automatically.

    Returns:
        float: The Repetition Time (TR) extracted from the first BOLD file's header.

    Raises:
        ValueError: If inputs are invalid (e.g., no BOLD files, format mismatch, inconsistent headers).
        FileNotFoundError: If a required file does not exist.
    """
    if not bold_files:
        raise ValueError("At least one BOLD file must be provided.")

    # Check for existence and valid NIfTI extension for all BOLD files
    for bold_file in bold_files:
        if not os.path.exists(bold_file):
            raise FileNotFoundError(f"BOLD file not found: {bold_file}")
        if not bold_file.endswith((".nii", ".nii.gz")):
            raise ValueError(f"BOLD file must be NIfTI format: {bold_file}")

    # Check for mask file existence if it's not being auto-generated
    if not generate_mask and (mask_file is None or not os.path.exists(mask_file)):
        raise FileNotFoundError(f"Mask file not found or not provided: {mask_file}")

    # Load the first image to use as a reference for header consistency
    first_img = nib.load(bold_files[0])
    # Extract TR from the last element of the zooms (pixdim array in header)
    tr = _tr_seconds(first_img, bold_files[0])
    if tr <= 0:
        raise ValueError(f"Invalid TR value from BOLD header: {tr}.")

    # Check if all subsequent images have the same affine, dimensions, and TR
    # This is crucial for correctly concatenating runs.
    for bold_file in bold_files[1:]:
        img = nib.load(bold_file)
        # `np.allclose` is used for robust floating-point comparison of affine matrices
        if not np.allclose(img.affine, first_img.affine):
            raise ValueError("All BOLD files must have the same affine transformation.")
        # Check spatial dimensions (first 3 dimensions)
        if img.shape[:3] != first_img.shape[:3]:
            raise ValueError("All BOLD files must have the same spatial dimensions.")
        # Check TR
        if not np.isclose(_tr_seconds(img, bold_file), tr):
            raise ValueError("All BOLD files must have the same TR.")

    return float(tr)


def to_numpy(array_backend, use_gpu):
    """
    Converts a CuPy array back to a NumPy array if GPU is used. If not, it's a no-op.
    This provides a single function call to ensure data is back on the CPU for tasks
    like saving to disk or using CPU-only libraries (e.g., Nilearn).

    Args:
        array_backend (np.ndarray or cp.ndarray): The array to potentially convert.
        use_gpu (bool): Flag indicating if GPU is being used.

    Returns:
        np.ndarray: The array as a NumPy array.
    """
    if use_gpu and _CUPY_AVAILABLE:
        # `cp.asnumpy` is the efficient CuPy function to transfer data from GPU to CPU
        return cp.asnumpy(array_backend)
    return array_backend


def free_gpu_memory(use_gpu):
    """
    Manually frees up GPU memory by clearing the CuPy memory pool. This can be
    useful after large array operations to ensure memory is available for subsequent steps.

    Args:
        use_gpu (bool): Flag indicating if GPU is being used.
    """
    if not use_gpu or not _CUPY_AVAILABLE:
        return

    # CuPy uses a memory pool to speed up allocations. `free_all_blocks` clears this
    # pool, releasing the memory back to the CUDA driver for other processes to use.
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
