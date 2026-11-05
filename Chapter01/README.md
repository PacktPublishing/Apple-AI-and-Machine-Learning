# Chapter 1 – Converting and Validating PyTorch Models for Apple Platforms

This folder contains the code for Chapter 1. It trains a first PyTorch model,
defines two models (an image classifier and an activity recognition
transformer), converts them to Core AI `.aimodel` assets, and validates the
converted models against PyTorch.

## Requirements

- A Mac with Apple silicon (the Core AI Python runtime only runs there)
- Python 3.11 or later
- Xcode 27 or later, to open and inspect the generated assets

## Setup

Run every command from this `Chapter01` folder.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Contents

| Path | Section of the chapter |
|------|------------------------|
| `pytorch_basics/tensors.py` | Tensors, the basic building block |
| `pytorch_basics/autograd.py` | Automatic differentiation in one minute |
| `pytorch_basics/train_circle.py` | Training a first model end to end |
| `models/image_classifier.py` | Preparing an image classifier |
| `models/motion_classifier.py` | Preparing a sequence model for activity recognition |
| `snippets/export_image_classifier.py` | Exporting, reading the graph, converting, first inference |
| `snippets/export_motion_classifier.py` | Declaring a dynamic dimension |
| `convert.py` | From interactive conversion to a reproducible build step |
| `contracts/` | Contract sheets for both models |
| `validate.py` | A first comparison, and reference outputs for the Swift team |
| `parity.py` | Comparison metrics and sample generation |
| `tests/test_parity.py` | Turning validation into tests |
| `snippets/export_pitfalls.py` | When torch.export rejects your model |
| `../.github/workflows/models.yml` | Running everything in CI |

## Running the examples

```bash
# PyTorch basics
python pytorch_basics/tensors.py
python pytorch_basics/autograd.py
python pytorch_basics/train_circle.py

# Interactive steps from the chapter
python -m snippets.export_image_classifier
python -m snippets.export_motion_classifier
python -m snippets.export_pitfalls

# Build, validate, and test
python convert.py --out build
python validate.py
pytest -q
```

Generated files (assets, manifests, checkpoints, and reference outputs) are
written to `build/`, which is not committed.
