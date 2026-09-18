"""The defaults table in README.md must match the argument parser (so the documentation cannot drift)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bold_lag_mapper import cli  # noqa: E402

README = Path(__file__).resolve().parents[1] / "README.md"


def _table_rows():
    text = README.read_text(encoding="utf-8")
    section = text.split("## Defaults", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| `(--[a-z0-9-]+)` \| `([^`]+)` \|", section, re.M)
    assert rows, "no rows found in the README defaults table"
    return rows


def test_readme_python_api_example_runs(tmp_path):
    """The Python API snippet of the README, on a small synthetic run."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_save_cleaned_bold import _synthetic_run
    from bold_lag_mapper.cli import build_parser
    from bold_lag_mapper import BOLDLagMapper
    bold, mask, _ = _synthetic_run(tmp_path)
    out = tmp_path / "out"
    args = build_parser().parse_args(["--bold-files", bold, "--mask-file", mask, "--seed-roi-file", "global"])
    BOLDLagMapper(**vars(args)).process_runs(args.bold_files, [None], args.mask_file, str(out))
    filled = [p for p in out.glob("*_lagmap.nii.gz") if "_desc-" not in p.name]    # not the prefill / raw maps
    assert len(filled) == 1 and len(list(out.glob("*_desc-stats.json"))) == 1, sorted(p.name for p in out.iterdir())


def test_every_documented_default_matches_the_parser():
    actions = {a.option_strings[0]: a for a in cli.build_parser()._actions if a.option_strings}
    for option, documented in _table_rows():
        assert option in actions, f"{option} is documented but not an option"
        default = actions[option].default
        assert str(default) == documented, f"{option}: README says {documented}, parser default is {default!r}"
