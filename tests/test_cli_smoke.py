"""
CLI smoke test: python -m bold_lag_mapper --help and --version exit 0.
"""

import subprocess
import sys
import os


def test_module_help_exits_zero():
    """Running python -m bold_lag_mapper --help should succeed."""
    result = subprocess.run(
        [sys.executable, "-m", "bold_lag_mapper", "--help"],
        capture_output=True,
        text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}.\n"
        f"stderr: {result.stderr}"
    )
    assert "bold" in result.stdout.lower() or "lag" in result.stdout.lower(), (
        f"Help output doesn't mention 'bold' or 'lag': {result.stdout[:200]}"
    )


def test_version_flag():
    """Running python -m bold_lag_mapper --version should print the version."""
    result = subprocess.run(
        [sys.executable, "-m", "bold_lag_mapper", "--version"],
        capture_output=True,
        text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}.\n"
        f"stderr: {result.stderr}"
    )
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from bold_lag_mapper import __version__
    assert __version__ in result.stdout, (
        f"Version {__version__} not found in output: {result.stdout}"
    )
