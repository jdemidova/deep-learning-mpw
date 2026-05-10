# Notebook → PDF export helpers

This directory contains scripts for converting Jupyter notebooks (`.ipynb`) to PDF reports when IDE export breaks.

## Problem

PyCharm / JetBrains notebook export may fail or produce incomplete PDFs because of notebook-format and path-resolution issues.

Common symptoms:

```text
Notebook JSON is invalid: Additional properties are not allowed ('jetTransient' was unexpected)
OSError: xelatex not found on PATH
InvalidNotebook: missing attachment: <uuid>.png
```

Or: the notebook looks correct in the IDE, but some images are missing in the exported PDF.

# Scripts
`fix_ipynb_for_pdf.py`

Repairs notebook JSON before export.

It does the following:

* removes JetBrains-specific invalid fields such as jetTransient;
* detects Markdown image links like attachment:...;
* extracts embedded notebook attachments into a normal folder;
* rewrites attachment links to normal relative image paths;
* searches sibling notebooks and .ipynb_checkpoints for missing attachments;
* reports images that cannot be recovered.

Use this first when the notebook contains embedded Markdown images or PyCharm metadata.

`ipynb_to_pdf.py`

Converts the fixed notebook to PDF without using PyCharm export.

It does the following:

* loads the notebook directly;
* cleans remaining jetTransient fields;
* renders notebook to HTML via nbconvert;
* injects the correct notebook directory as the base path;
* prints HTML to PDF using Chromium through Playwright.

This avoids the LaTeX / xelatex route and is usually more reliable for practice-work reports.

# Setup

``` bash
source .venv/bin/activate

python -m pip install -U nbconvert nbformat playwright
python -m playwright install chromium
```

# Workflow
Example NB: `notebooks/cg/cg-linear-regression-stud.ipynb`
## 1.Repair NB
``` bash
python fix_ipynb_for_pdf.py ../notebooks/cg/cg-linear-regression-stud.ipynb
```
## 2. Convert fixed NB to PDF
``` bash
python ipynb_to_pdf.py ../notebooks/cg/cg-linear-regression-stud-fixed.ipynb --keep-html
```
## 3. Custom output path PDF
``` bash
python ipynb_to_pdf.py \
  ../notebooks/cg/cg-linear-regression-stud-fixed.ipynb \
  --no-input \
  --output ../notebooks/cg-linear-regression-stud.pdf
```

## 4. Debugging
``` bash
python ipynb_to_pdf.py \
  ../notebooks/cg/cg-linear-regression-stud-fixed.ipynb \
  --keep-html \
  --debug
```

## Interpretation

* Images missing in both HTML and PDF: notebook paths/attachments are still broken.
* Images visible in HTML but missing in PDF: Chromium print/rendering issue.
* Images visible in notebook IDE but missing in HTML/PDF: IDE is resolving cached/internal attachments that are not represented correctly in notebook JSON.