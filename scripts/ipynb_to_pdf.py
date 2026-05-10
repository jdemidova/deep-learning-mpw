#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright


ATTACHMENT_RE = re.compile(r"attachment:([^)\]\"'\s>]+)")


def as_text(value) -> str:
    """Notebook multiline fields may be string or list[str]. Normalize to string."""
    if isinstance(value, list):
        return "".join(as_text(x) for x in value)
    if value is None:
        return ""
    return str(value)


def remove_jetbrains_fields(obj) -> int:
    removed = 0

    if isinstance(obj, dict):
        if "jetTransient" in obj:
            obj.pop("jetTransient")
            removed += 1

        for value in obj.values():
            removed += remove_jetbrains_fields(value)

    elif isinstance(obj, list):
        for item in obj:
            removed += remove_jetbrains_fields(item)

    return removed


def normalize_notebook_sources(nb: dict) -> int:
    """
    Normalize all cell.source values to strings.
    This avoids regex/nbconvert crashes caused by list-based sources.
    """
    changed = 0

    for cell in nb.get("cells", []):
        if "source" in cell and not isinstance(cell["source"], str):
            cell["source"] = as_text(cell["source"])
            changed += 1

    return changed


def check_attachment_refs(nb: dict) -> list[str]:
    problems = []

    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue

        source = as_text(cell.get("source", ""))
        refs = ATTACHMENT_RE.findall(source)

        attachments = cell.get("attachments", {}) or {}

        for ref in refs:
            if ref not in attachments:
                problems.append(f"cell {i}: missing attachment {ref}")

    return problems


def inject_base_href(html: str, notebook_dir: Path) -> str:
    """
    Make relative image paths resolve relative to the notebook directory.
    This matters when the script is launched from project root.
    """
    base = f'<base href="{notebook_dir.as_uri()}/">\n'

    if "<head>" in html:
        return html.replace("<head>", f"<head>\n{base}", 1)

    return base + html


def convert_ipynb_to_pdf(
    notebook_path: Path,
    output_pdf: Path | None,
    keep_html: bool,
    no_input: bool,
) -> None:
    notebook_path = notebook_path.resolve()
    notebook_dir = notebook_path.parent

    if not notebook_path.exists():
        raise FileNotFoundError(notebook_path)

    if notebook_path.suffix != ".ipynb":
        raise ValueError(f"Expected .ipynb file, got: {notebook_path}")

    output_pdf = output_pdf or notebook_path.with_suffix(".pdf")
    output_pdf = output_pdf.resolve()

    debug_html = output_pdf.with_suffix(".html")

    raw = json.loads(notebook_path.read_text(encoding="utf-8"))

    removed_jetbrains = remove_jetbrains_fields(raw)
    normalized_sources = normalize_notebook_sources(raw)

    missing_attachments = check_attachment_refs(raw)

    if missing_attachments:
        print("WARNING: notebook has broken attachment references:")
        for item in missing_attachments[:50]:
            print(f"  - {item}")
        if len(missing_attachments) > 50:
            print(f"  ... and {len(missing_attachments) - 50} more")
        print()

    nb = nbformat.from_dict(raw)

    exporter = HTMLExporter()
    exporter.template_name = "lab"

    if no_input:
        exporter.exclude_input = True
        exporter.exclude_input_prompt = True
        exporter.exclude_output_prompt = True

    resources = {
        "metadata": {
            "path": str(notebook_dir),
        }
    }

    html, _ = exporter.from_notebook_node(nb, resources=resources)
    html = inject_base_href(html, notebook_dir)

    debug_html.write_text(html, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        page.goto(debug_html.as_uri(), wait_until="networkidle")

        page.pdf(
            path=str(output_pdf),
            format="A4",
            print_background=True,
            margin={
                "top": "12mm",
                "right": "12mm",
                "bottom": "12mm",
                "left": "12mm",
            },
        )

        browser.close()

    if not keep_html:
        debug_html.unlink(missing_ok=True)

    print("Done.")
    print(f"Input notebook:          {notebook_path}")
    print(f"Output PDF:              {output_pdf}")
    print(f"Removed jetTransient:    {removed_jetbrains}")
    print(f"Normalized cell sources: {normalized_sources}")

    if keep_html:
        print(f"Debug HTML:              {debug_html}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert .ipynb to PDF via nbconvert HTML + Chromium."
    )
    parser.add_argument("notebook", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--keep-html", action="store_true")
    parser.add_argument("--no-input", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    try:
        convert_ipynb_to_pdf(
            notebook_path=args.notebook,
            output_pdf=args.output,
            keep_html=args.keep_html,
            no_input=args.no_input,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)

        if args.debug:
            traceback.print_exc()

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())