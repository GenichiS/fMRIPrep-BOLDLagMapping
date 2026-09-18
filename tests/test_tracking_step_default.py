"""--tracking-step-seconds auto: long-TR data (TR >= --subtr-min-tr) are tracked on a 1 s grid, short-TR data at
their own TR; 'none' keeps the acquisition TR; a number is used as given."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bold_lag_mapper import core  # noqa: E402


@pytest.mark.parametrize("value,tr,expected", [
    ("auto", 2.5, 1.0), ("auto", 1.5, 1.0), ("AUTO", 3.0, 1.0), ("auto", 1.49, None), ("auto", 0.8, None),
    ("none", 2.5, None), (None, 2.5, None), ("1.0", 0.8, 1.0), (0.5, 2.5, 0.5)])
def test_resolve_tracking_step(value, tr, expected):
    assert core.resolve_tracking_step(value, tr, 1.5) == expected


@pytest.mark.parametrize("value", ["0", -1.0])
def test_non_positive_step_is_refused(value):
    with pytest.raises(ValueError):
        core.resolve_tracking_step(value, 2.5, 1.5)
