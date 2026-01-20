import json
import os
import sys
from pathlib import Path

import pytest


def _load_notebook_cells(nb_path: Path):
    with nb_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Return only code cells as (source_string) list
    cells = []
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        # Some tools store source as list of lines; be robust
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        cells.append(src)
    return cells


def _should_skip(code: str) -> bool:
    # Heuristics to skip purely-visual or environment/magic cells in tests
    stripped = code.strip()
    if not stripped:
        return True
    # Skip shell and ipython magics
    if stripped.startswith("%") or stripped.startswith("!"):
        return True
    # Skip explicit blocking display calls
    visual_markers = (
        "plt.show(",
        "display(",
        "nx.draw_",
        "draw_networkx",
        "Line2D(",
    )
    if any(mark in code for mark in visual_markers):
        return True
    return False


@pytest.mark.parametrize(
    "notebook_rel",
    [
        Path("tutorial") / "Max-Cut.ipynb",
    ],
)
def test_execute_notebook_for_coverage(notebook_rel: Path):
    # Ensure we can import local package without installation
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    nb_path = repo_root / notebook_rel
    if not nb_path.exists():  # pragma: no cover - local dev guard
        pytest.skip(f"Notebook not found: {notebook_rel}")

    # Use non-interactive matplotlib backend for headless test envs
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Minimal execution namespace; mimic a fresh notebook kernel
    ns: dict = {"__name__": "__main__"}

    # Execute cells, skipping heavy visualization/magic cells
    for code in _load_notebook_cells(nb_path):
        if _should_skip(code):
            continue
        # Some notebooks assume cwd at repo root; enforce it
        cwd_before = os.getcwd()
        try:
            os.chdir(repo_root)
            compiled = compile(code, filename=str(notebook_rel), mode="exec")
            exec(compiled, ns)
        finally:
            os.chdir(cwd_before)

    # Light sanity check: notebook defined a distance/weight matrix
    # This keeps the test meaningful without over-specifying outputs
    assert any(
        k in ns for k in ("dist_matrix", "weights", "G")
    ), "Notebook did not appear to run core setup cells"
