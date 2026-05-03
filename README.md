# Repository Description
This repository contains the results of mandatory practice works (MPWs) for MSE module Deep Learning (Spring 2026)

## Part 1: CNN 
* Timeframe: From 18.03.2026 until 15.04.2026
* NBs: `./notebooks/cnn/`
* See README for the task: `./notebooks/cnn/README.md`

## Part 2: CG (Computational Graphs and Optimizers)
* Timeframe: From 20.04.2026 until 18.05.2026
* NBs: `./notebooks/cg/`
* See README for the task: `./notebooks/cg/README.md`

## Part 3: AL
Timeframe: From 18.05.2026 until 27.05.2026????

# Project structure
## Tasks
See `../Task/` and `../Task/./documents/*`
## Data
See `Task/data/*`
## Submissions
In a form of Jupyter NBs or PDFs. See `./notebooks/*` and `README` inside. 
## Reusable Python code
Put larger reusable methods and helpers in `./src/*`, and keep notebooks focused on experiments and reporting.

## Python environment
This project is configured for `uv` and targets Python `3.12`.

### Setup
1. Install `uv`
2. Create the environment and install dependencies:
   `uv sync --dev`
3. Run Python commands inside the managed environment:
   `uv run python`

### Dependency management
- Add a runtime dependency: `uv add <package>`
- Add a development dependency: `uv add --dev <package>`
- Update the lockfile after changes: `uv lock`
- Export `requirements.txt` only when another tool explicitly needs it

