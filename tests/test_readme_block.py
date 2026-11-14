"""The README's results block must be exactly what scripts/report.py generates from results/*.json."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_results_block_is_generated():
    spec = importlib.util.spec_from_file_location("report", ROOT / "scripts" / "report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    assert report.update_readme(check=True), "README results block is stale: run python scripts/report.py --readme"
