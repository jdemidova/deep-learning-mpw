# CNN Utilities

See also: [CNN_API_demo.ipynb](notebooks/cnn/CNN_API_demo.ipynb) for a demo of the API in action.

This module contains reusable pieces for running CNN experiments in PyTorch:

- model definitions
- model registry and model builder
- experiment/config objects
- dataset loading
- dataloader creation
- training / evaluation loops
- optional Weights & Biases logging

The goal is simple:
**define a model, define a config, run an experiment**

---

## Files

### `cnn_models.py`
Contains CNN model classes.

Typical contents:
- custom `nn.Module` classes such as `ImprovedModel`, `ShallowModel`, `DepthCNN`
- model definitions only
> Always add new models here with registration in `cnn_registry.py`. 
> 
>Do not add new models in the notebook or script.

### `cnn_registry.py`
Contains model registration and model construction logic.

Typical contents:
- `MODEL_REGISTRY`
- `register_model(...)`
- `build_model(...)`

This is the single source of truth for building models from config.

### `cnn_utils.py`
Contains reusable training and data utility functions.

Typical contents:
- dataset loading
- dataset inspection
- dataloader builders
- device selection
- parameter counting
- W&B config serialization and initialization
- `train_one_epoch(...)`
- `evaluate(...)`
- `train(...)`
- `train_eval(...)`

### `configs.py`
Contains config dataclasses controlling:
- dataset settings
- model settings
- dataloader settings
- loss / optimizer / scheduler settings
- train settings
- W&B settings

### `cnn_paths.py`
Contains canonical project paths for CNN experiments:
- dataset root
- train / val / test folders
- split directory
- other CNN-related filesystem locations

Paths should be defined with `pathlib.Path`.

---

## General workflow

The intended flow is:

1. create an experiment config
2. load datasets
3. inspect dataset metadata
4. build dataloaders
5. build model from the registry
6. run `train_eval(...)`
7. inspect returned metrics / W&B logs

>The notebook or script should be the orchestration layer, not the place where training logic is reimplemented.

---

## Expected dataset structure

The utilities assume an `ImageFolder`-style structure.

Example:

```text
dataset_root/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── ...
├── validate/
│   ├── class_0/
│   ├── class_1/
│   └── ...
└── test/                 # optional
    ├── class_0/
    ├── class_1/
    └── ...