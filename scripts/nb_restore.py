# A specific script to restore the broken notebooks.
# The problem appears after change pulls.
# The problem is that NB is written into single NB cell in a JSON format.

# RUN:
# $ uv run python scripts/nb_restore.py notebooks/cnn/CNN-Grad-CAM.ipynb --clear-outputs
# It Creates:
# ..._restored.ipynb

# USE:
# backup broken notebook : $ cp notebooks/cnn/CNN-Grad-CAM.ipynb notebooks/cnn/CNN-Grad-CAM.backup.ipynb
# RUN recovery script
# open restored notebook in IDE
# if it looks correct, replace the broken one
# commit

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat


def find_nested_notebook_cell(nb) -> tuple[int, str] | None:
    for i, cell in enumerate(nb.cells):
        source = getattr(cell, "source", "")
        if not isinstance(source, str):
            continue

        s = source.strip()
        if s.startswith("{") and '"cells"' in s and '"nbformat"' in s:
            return i, s

    return None


def strip_outputs_and_problem_keys(nb_dict: dict) -> None:
    for cell in nb_dict.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

        # extra safety: if outputs are kept somewhere weird, strip jetTransient
        for output in cell.get("outputs", []):
            if isinstance(output, dict):
                output.pop("jetTransient", None)

    # optional cleanup
    metadata = nb_dict.get("metadata", {})
    if isinstance(metadata, dict):
        metadata.pop("widgets", None)


def recover_notebook(
    src: Path,
    dst: Path,
    force: bool = False,
    clear_outputs: bool = False,
) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source notebook does not exist: {src}")

    if dst.exists() and not force:
        raise FileExistsError(f"Destination already exists: {dst}\nUse --force to overwrite.")

    outer_nb = nbformat.read(src, as_version=4)

    found = find_nested_notebook_cell(outer_nb)
    if found is None:
        raise ValueError(
            "Could not find nested notebook JSON inside any cell.\n"
            "This notebook may be corrupted in a different way."
        )

    cell_idx, inner_text = found

    try:
        inner_dict = json.loads(inner_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Found suspicious JSON-like text in cell #{cell_idx + 1}, "
            f"but it is not valid JSON: {e}"
        ) from e

    if clear_outputs:
        strip_outputs_and_problem_keys(inner_dict)

    inner_nb = nbformat.from_dict(inner_dict)

    # validate after cleanup
    nbformat.validate(inner_nb)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        nbformat.write(inner_nb, f)

    print(f"Recovered nested notebook from cell #{cell_idx + 1}")
    print(f"Source:      {src}")
    print(f"Destination: {dst}")
    print(f"Recovered cells: {len(inner_nb.cells)}")
    print(f"Outputs cleared: {clear_outputs}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover a real notebook embedded as JSON inside a notebook cell."
    )
    parser.add_argument("src", type=Path, help="Path to the broken .ipynb notebook")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to the restored .ipynb notebook. Default: <src_stem>_restored.ipynb",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite destination if it already exists.",
    )
    parser.add_argument(
        "--clear-outputs",
        action="store_true",
        help="Strip outputs and execution counts from recovered code cells.",
    )

    args = parser.parse_args()

    src = args.src
    dst = args.output or src.with_name(f"{src.stem}_restored.ipynb")

    recover_notebook(
        src=src,
        dst=dst,
        force=args.force,
        clear_outputs=args.clear_outputs,
    )


if __name__ == "__main__":
    main()
