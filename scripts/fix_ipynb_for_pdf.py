#!/usr/bin/env python3
"""
Fix a JetBrains/PyCharm-damaged .ipynb for PDF export.

What it does:
1. Removes invalid JetBrains fields like `jetTransient`.
2. Finds Markdown links like `attachment:some-image.png`.
3. Extracts real embedded notebook attachments into a normal folder.
4. Rewrites `attachment:...` links to normal relative file paths.
5. Searches sibling notebooks and .ipynb_checkpoints to recover missing attachments.
6. Reports attachments that are truly unrecoverable.

Usage:
    python fix_ipynb_for_pdf.py notebooks/cg/cg-linear-regression-stud.ipynb

Then export:
    jupyter nbconvert notebooks/cg/cg-linear-regression-stud-fixed.ipynb --to webpdf --output cg-linear-regression-stud
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import quote, unquote


ATTACHMENT_RE = re.compile(r"attachment:([^)\]\"'\s>]+)")
IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".idea",
}


def read_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_notebook(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def source_to_text(source) -> str:
    if isinstance(source, list):
        return "".join(source)
    return str(source or "")


def text_to_source(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def safe_filename(name: str) -> str:
    return Path(unquote(name)).name


def decode_b64(value: str) -> bytes:
    if value.lstrip().startswith("data:") and "," in value:
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def extension_from_mime(mime: str) -> str:
    if mime == "image/svg+xml":
        return ".svg"
    return mimetypes.guess_extension(mime) or ""


def remove_jetbrains_metadata(obj) -> int:
    removed = 0

    if isinstance(obj, dict):
        if "jetTransient" in obj:
            obj.pop("jetTransient")
            removed += 1

        for value in obj.values():
            removed += remove_jetbrains_metadata(value)

    elif isinstance(obj, list):
        for item in obj:
            removed += remove_jetbrains_metadata(item)

    return removed


def attachment_bytes_from_cell(cell: dict, filename: str) -> tuple[bytes, str] | None:
    attachments = cell.get("attachments", {}) or {}
    filename = safe_filename(filename)

    candidates = []

    for attachment_name, mime_map in attachments.items():
        if safe_filename(attachment_name) == filename:
            candidates.append(mime_map)

    for mime_map in candidates:
        if not isinstance(mime_map, dict):
            continue

        image_items = [
            (mime, data)
            for mime, data in mime_map.items()
            if isinstance(mime, str)
            and mime.startswith("image/")
            and isinstance(data, str)
        ]

        if not image_items:
            continue

        mime, data = image_items[0]

        try:
            return decode_b64(data), mime
        except Exception:
            continue

    return None


def collect_attachment_bank(notebook_paths: list[Path]) -> dict[str, tuple[bytes, str, Path]]:
    """
    filename -> (bytes, mime, source_notebook_path)

    Last writer wins. In practice this is okay because Jupyter attachment filenames
    are usually UUID-ish and unique.
    """
    bank = {}

    for path in notebook_paths:
        try:
            nb = read_notebook(path)
        except Exception:
            continue

        for cell in nb.get("cells", []):
            attachments = cell.get("attachments", {}) or {}

            for filename, mime_map in attachments.items():
                filename = safe_filename(filename)

                if not isinstance(mime_map, dict):
                    continue

                image_items = [
                    (mime, data)
                    for mime, data in mime_map.items()
                    if isinstance(mime, str)
                    and mime.startswith("image/")
                    and isinstance(data, str)
                ]

                if not image_items:
                    continue

                mime, data = image_items[0]

                try:
                    bank[filename] = (decode_b64(data), mime, path)
                except Exception:
                    pass

    return bank


def discover_candidate_notebooks(target: Path) -> list[Path]:
    candidates = [target]

    candidates.extend(sorted(target.parent.glob("*.ipynb")))

    checkpoint_dir = target.parent / ".ipynb_checkpoints"
    if checkpoint_dir.exists():
        candidates.extend(sorted(checkpoint_dir.glob("*.ipynb")))

    # Also check one level below sibling checkpoint dirs, because IDEs can be funky.
    for p in sorted(target.parent.glob("*/.ipynb_checkpoints/*.ipynb")):
        candidates.append(p)

    seen = set()
    unique = []

    for p in candidates:
        try:
            resolved = p.resolve()
        except Exception:
            continue

        if resolved not in seen and p.is_file():
            seen.add(resolved)
            unique.append(p)

    return unique


def find_existing_file_by_name(project_root: Path, filename: str) -> Path | None:
    filename = safe_filename(filename)

    for path in project_root.rglob(filename):
        if not path.is_file():
            continue

        if any(part in IGNORED_DIRS for part in path.parts):
            continue

        return path

    return None


def unique_output_path(asset_dir: Path, filename: str) -> Path:
    filename = safe_filename(filename)
    candidate = asset_dir / filename

    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix

    for i in range(1, 10_000):
        candidate = asset_dir / f"{stem}-{i}{suffix}"
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not create unique filename for {filename}")


def write_asset(asset_dir: Path, filename: str, data: bytes, mime: str) -> Path:
    filename = safe_filename(filename)

    if not Path(filename).suffix:
        filename += extension_from_mime(mime)

    out_path = asset_dir / filename

    if out_path.exists():
        if out_path.read_bytes() == data:
            return out_path

        out_path = unique_output_path(asset_dir, filename)

    out_path.write_bytes(data)
    return out_path


def fix_notebook(target: Path, project_root: Path | None = None) -> Path:
    target = target.resolve()

    if not target.exists():
        raise FileNotFoundError(target)

    if target.suffix != ".ipynb":
        raise ValueError(f"Expected .ipynb file, got: {target}")

    project_root = project_root.resolve() if project_root else Path.cwd().resolve()

    nb = read_notebook(target)

    removed_metadata = remove_jetbrains_metadata(nb)

    asset_dir = target.parent / f"{target.stem}_attachments"
    asset_dir.mkdir(exist_ok=True)

    output_path = target.with_name(f"{target.stem}-fixed.ipynb")

    candidate_notebooks = discover_candidate_notebooks(target)
    attachment_bank = collect_attachment_bank(candidate_notebooks)

    saved_assets = 0
    rewritten_refs = 0
    recovered_from_same_cell = 0
    recovered_from_bank = 0
    recovered_from_files = 0
    missing: list[tuple[int, str]] = []

    for cell_index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue

        source = source_to_text(cell.get("source", ""))
        refs = ATTACHMENT_RE.findall(source)

        if not refs:
            continue

        replacements: dict[str, str] = {}

        for raw_ref in refs:
            filename = safe_filename(raw_ref)

            data = None
            mime = None

            # Best source: exact attachment embedded in the same Markdown cell.
            same_cell = attachment_bytes_from_cell(cell, filename)
            if same_cell is not None:
                data, mime = same_cell
                recovered_from_same_cell += 1

            # Next: any scanned sibling/checkpoint notebook.
            elif filename in attachment_bank:
                data, mime, _origin = attachment_bank[filename]
                recovered_from_bank += 1

            # Last: physical file somewhere in project.
            else:
                existing_file = find_existing_file_by_name(project_root, filename)
                if existing_file is not None:
                    data = existing_file.read_bytes()
                    mime = (
                        mimetypes.guess_type(existing_file.name)[0]
                        or "application/octet-stream"
                    )
                    recovered_from_files += 1

            if data is None or mime is None:
                missing.append((cell_index, filename))
                continue

            out_path = write_asset(asset_dir, filename, data, mime)
            saved_assets += 1

            rel_path = out_path.relative_to(target.parent).as_posix()
            replacements[raw_ref] = quote(rel_path)

        for raw_ref, rel_path in replacements.items():
            source = source.replace(f"attachment:{raw_ref}", rel_path)
            rewritten_refs += 1

        cell["source"] = text_to_source(source)

        # Now that refs are normal file paths, embedded attachments are unnecessary.
        cell.pop("attachments", None)

    write_notebook(output_path, nb)

    print()
    print("Done.")
    print(f"Input notebook:             {target}")
    print(f"Fixed notebook:             {output_path}")
    print(f"Asset folder:               {asset_dir}")
    print()
    print(f"Removed JetBrains fields:   {removed_metadata}")
    print(f"Scanned notebooks:          {len(candidate_notebooks)}")
    print(f"Attachment names in bank:   {len(attachment_bank)}")
    print(f"Saved/reused assets:        {saved_assets}")
    print(f"Rewritten attachment refs:  {rewritten_refs}")
    print()
    print(f"Recovered from same cell:   {recovered_from_same_cell}")
    print(f"Recovered from checkpoints: {recovered_from_bank}")
    print(f"Recovered from files:       {recovered_from_files}")

    if missing:
        print()
        print("Still missing. These cannot be reconstructed from the available data:")
        for cell_index, filename in missing[:100]:
            print(f"  cell {cell_index}: {filename}")

        if len(missing) > 100:
            print(f"  ... and {len(missing) - 100} more")

        print()
        print("Brutal truth: if the base64 image data is gone from the notebook/checkpoints")
        print("and no real file exists in the project, the UUID filename alone is useless.")
    else:
        print()
        print("No missing attachment references remain.")

    print()
    print("Next export command:")
    print(
        f"  jupyter nbconvert {output_path} "
        f"--to webpdf --output {target.stem}"
    )

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fix broken ipynb attachments and JetBrains metadata for PDF export."
    )
    parser.add_argument(
        "notebook",
        type=Path,
        help="Path to the .ipynb file.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root used when searching for physical image files. Defaults to current directory.",
    )

    args = parser.parse_args()

    try:
        fix_notebook(args.notebook, args.project_root)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())