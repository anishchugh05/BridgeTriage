"""Regression tests for planner import, selection, and export plumbing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_import_backtest_does_not_require_matplotlib() -> None:
    """Core planner APIs remain importable without the plotting dependency."""
    code = """
import builtins

real_import = builtins.__import__


def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "matplotlib" or name.startswith("matplotlib."):
        raise ModuleNotFoundError("matplotlib intentionally unavailable")
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = blocked_import
import planner.harness
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def _candidate_fixture():
    import pandas as pd

    candidates = pd.DataFrame(
        {
            "county": ["A", "B"],
            "inspect_cost": [1.0, 1.0],
            "adt": [100.0, 50.0],
            "year_built": [1950, 1960],
            "min_2018": [4.0, 5.0],
        },
        index=["candidate-a", "candidate-b"],
    )
    pool_cost_info = {
        "total_cost_pool": 2.0,
        "unique_counties": {"A", "B"},
        "exclusion_counts": {},
    }
    return candidates, pool_cost_info


def test_pick_ignores_invalid_scores_for_non_candidates() -> None:
    """R5.7 extras cannot affect validation or selection."""
    import numpy as np

    from planner.harness import pick

    candidates, pool_cost_info = _candidate_fixture()
    selected = pick(
        {"candidate-a": 2.0, "candidate-b": 1.0, "not-a-candidate": np.nan},
        1.0,
        candidates,
        pool_cost_info,
        0.0,
    )

    assert selected == ["candidate-a", "candidate-b"]


def test_pick_rejects_unknown_tiebreak() -> None:
    """Direct picker calls reject values outside the documented enum."""
    import pytest

    from planner.harness import pick

    candidates, pool_cost_info = _candidate_fixture()
    with pytest.raises(ValueError, match="tiebreak"):
        pick(
            {"candidate-a": 2.0, "candidate-b": 1.0},
            1.0,
            candidates,
            pool_cost_info,
            0.0,
            tiebreak="typo",
        )


def test_load_model_scores_validates_and_normalizes(tmp_path) -> None:
    """The shared score loader accepts only complete, finite, unique scores."""
    import numpy as np
    import pandas as pd
    import pytest

    from planner.run_export import load_model_scores

    valid_path = tmp_path / "valid.parquet"
    pd.DataFrame(
        {
            "id": [" bridge-a ", "bridge-b", "extra"],
            "score": [2.0, 1.0, 0.5],
            "intentional_extra_column": [1, 2, 3],
        }
    ).to_parquet(valid_path, index=False)
    assert load_model_scores(valid_path, ["bridge-a", "bridge-b"]) == {
        "bridge-a": 2.0,
        "bridge-b": 1.0,
        "extra": 0.5,
    }

    duplicate_path = tmp_path / "duplicate.parquet"
    pd.DataFrame(
        {"id": ["bridge-a", " bridge-a "], "score": [2.0, 1.0]}
    ).to_parquet(duplicate_path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_model_scores(duplicate_path, ["bridge-a"])

    nonfinite_path = tmp_path / "nonfinite.parquet"
    pd.DataFrame(
        {"id": ["bridge-a", "bridge-b"], "score": [2.0, np.inf]}
    ).to_parquet(nonfinite_path, index=False)
    with pytest.raises(ValueError, match="finite"):
        load_model_scores(nonfinite_path, ["bridge-a", "bridge-b"])

    incomplete_path = tmp_path / "incomplete.parquet"
    pd.DataFrame({"id": ["bridge-a"], "score": [2.0]}).to_parquet(
        incomplete_path, index=False
    )
    with pytest.raises(ValueError, match="coverage"):
        load_model_scores(incomplete_path, ["bridge-a", "bridge-b"])


def test_export_backtest_writes_required_artifacts(tmp_path, capsys) -> None:
    """The real export reports fixed-selection recall gaps and pool metadata."""
    import json

    import pandas as pd

    from planner.run_export import export_backtest

    panel_path = tmp_path / "panel.parquet"
    score_path = tmp_path / "scores_2018.parquet"
    output_dir = tmp_path / "results"

    rows = []
    starts = {"A": 5, "B": 6, "C": 4, "D": 7}
    ends = {"A": 4, "B": 5, "C": 3, "D": 7}
    for sid in starts:
        rows.append(
            {
                "structure_id": sid,
                "year": 2018,
                "county": "county-1",
                "age": 50,
                "adt": 1000.0,
                "deck": starts[sid],
                "super": starts[sid],
                "sub": starts[sid],
                "rebuilt": False,
            }
        )
        rows.append(
            {
                "structure_id": sid,
                "year": 2022,
                "county": "county-1",
                "age": 54,
                "adt": 1000.0,
                "deck": ends[sid],
                "super": ends[sid],
                "sub": ends[sid],
                "rebuilt": False,
            }
        )
    pd.DataFrame(rows).to_parquet(panel_path, index=False)
    pd.DataFrame(
        {"id": ["A", "B", "C", "D"], "score": [4.0, 3.0, 2.0, 1.0]}
    ).to_parquet(score_path, index=False)

    csv_path, summary_path = export_backtest(
        panel_path=panel_path,
        scores_path=score_path,
        output_dir=output_dir,
        budgets=(0.5, 1.0),
        county_fee=0.0,
    )

    exported = pd.read_csv(csv_path)
    assert tuple(exported.columns) == (
        "method",
        "budget",
        "tiebreak",
        "recall",
        "weighted_recall",
        "gap_lo",
        "gap_hi",
    )
    assert set(exported["method"]) == {
        "worst_first",
        "oldest_first",
        "oracle",
        "model",
    }
    assert len(exported) == 8
    worst = exported[exported["method"] == "worst_first"]
    assert (worst["gap_lo"] == 0.0).all()
    assert (worst["gap_hi"] == 0.0).all()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary == {
        "pool_size": 4,
        "decliners": 2,
        "exclusion_counts": {
            "missing_2022": 0,
            "rebuilt": 0,
            "nan_min_2018": 0,
            "nan_min_2022": 0,
            "missing_yearbuilt_adt": 0,
            "duplicate_id": 0,
        },
    }
    assert "outcome-sampling variance only, not selection variance" in (
        capsys.readouterr().out
    )


def test_default_pytest_config_collects_planner_properties() -> None:
    """Default discovery includes planner/properties.py without naming it."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--ignore=tests",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Python 3.9 may still report the known app dataclass(slots=True)
    # collection limitation; discovery itself must nevertheless include both
    # planner properties below.
    assert completed.returncode in (0, 2), completed.stdout + completed.stderr
    assert "planner/properties.py::test_budget_compliance" in completed.stdout
    assert "planner/properties.py::test_is_safety_drop_parity" in completed.stdout


def test_analysis_scripts_do_not_execute_when_imported() -> None:
    """Importing either analysis module performs no parquet I/O or analysis."""
    code = """
import pandas as pd


def forbidden_read(*args, **kwargs):
    raise RuntimeError("analysis executed during import")


pd.read_parquet = forbidden_read
import planner.run_model
import planner.run_ci
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
