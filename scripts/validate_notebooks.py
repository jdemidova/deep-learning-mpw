# to RUN:
# $ uv run python scripts/validate_notebooks.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat


def looks_like_nested_notebook_json(text: str) -> bool:
    s = text.strip()
    return s.startswith("{") and '"cells"' in s and '"nbformat"' in s


def is_valid_nested_notebook_json(text: str) -> bool:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False

    if not isinstance(data, dict):
        return False

    return "cells" in data and "nbformat" in data


def validate_notebook(path: Path) -> list[str]:
    errors: list[str] = []

    try:
        nb = nbformat.read(path, as_version=4)
    except Exception as e:
        return [f"{path}: invalid notebook: {e}"]

    for i, cell in enumerate(nb.cells, start=1):
        source = getattr(cell, "source", "")
        if not isinstance(source, str):
            continue

        if looks_like_nested_notebook_json(source):
            if is_valid_nested_notebook_json(source):
                errors.append(
                    f"{path}: cell {i} contains a nested notebook JSON blob "
                    f"(this is the broken notebook-inside-a-cell problem)"
                )
            else:
                errors.append(f"{path}: cell {i} looks like notebook JSON, but is malformed")

    return errors


def collect_notebooks(root: Path) -> list[Path]:
    return sorted(root.rglob("*.ipynb"))


def main() -> int:
    root = Path(".")
    notebooks = collect_notebooks(root)

    if not notebooks:
        print("No notebooks found.")
        return 0

    all_errors: list[str] = []

    for path in notebooks:
        all_errors.extend(validate_notebook(path))

    if all_errors:
        print("Notebook validation failed:\n")
        for err in all_errors:
            print(f"- {err}")
        return 1

    print(f"Notebook validation passed ({len(notebooks)} notebooks checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
