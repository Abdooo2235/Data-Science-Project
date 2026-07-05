"""
Convert a jupytext "percent" format .py file into a clean .ipynb that
opens in Jupyter / Google Colab.

Usage:
    python scripts/py_to_nb.py notebooks/01_data_collection_preprocessing.py
        notebooks/01_data_collection_preprocessing.ipynb

Recognised cell markers:
    # %%                  -> code cell (everything until the next marker)
    # %% [markdown]       -> markdown cell (lines prefixed with "# " become text)

The leading jupytext YAML header (`# ---` ... `# ---`) is skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf


def parse_cells(text: str) -> list[tuple[str, list[str]]]:
    """Split a percent-format .py into a list of (cell_type, lines) tuples."""
    cells: list[tuple[str, list[str]]] = []
    current_type: str | None = None
    current_lines: list[str] = []

    # Skip optional leading jupytext YAML header: # --- ... # ---
    lines = text.splitlines()
    if lines and lines[0].strip() == "# ---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "# ---":
                lines = lines[i + 1 :]
                break

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("# %% [markdown]"):
            if current_type is not None:
                cells.append((current_type, current_lines))
            current_type, current_lines = "markdown", []
            continue
        if stripped.startswith("# %%"):
            if current_type is not None:
                cells.append((current_type, current_lines))
            current_type, current_lines = "code", []
            continue
        if current_type is None:
            continue
        current_lines.append(line)

    if current_type is not None:
        cells.append((current_type, current_lines))
    return cells


def markdown_source(lines: list[str]) -> str:
    """Strip leading `# ` from comment-only lines in a markdown cell."""
    out: list[str] = []
    for line in lines:
        if line.startswith("# "):
            out.append(line[2:])
        elif line.rstrip() == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out).strip()


def code_source(lines: list[str]) -> str:
    return "\n".join(lines).strip()


def convert(src_path: Path, dst_path: Path) -> None:
    text = src_path.read_text(encoding="utf-8")
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    for cell_type, lines in parse_cells(text):
        if cell_type == "markdown":
            src = markdown_source(lines)
            if src:
                nb.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            src = code_source(lines)
            if src:
                nb.cells.append(nbf.v4.new_code_cell(src))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(dst_path))
    print(f"wrote {dst_path} with {len(nb.cells)} cells")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python py_to_nb.py <src.py> <dst.ipynb>", file=sys.stderr)
        sys.exit(2)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
