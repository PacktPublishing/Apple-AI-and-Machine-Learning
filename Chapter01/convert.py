import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
import coreai_torch

from models import image_classifier, motion_classifier


@dataclass
class ModelSpec:
    name: str
    load: callable
    example_args: tuple
    input_names: list[str]
    output_names: list[str]
    dynamic_shapes: dict | None = None
    checkpoint: str | None = None


def specs() -> list[ModelSpec]:
    seq_len = torch.export.Dim(
        "seq_len", min=1, max=motion_classifier.MAX_SEQ_LEN)
    size = image_classifier.IMAGE_SIZE
    return [
        ModelSpec(
            name="ImageClassifier",
            load=image_classifier.load_model,
            example_args=(torch.rand(1, 3, size, size),),
            input_names=["image"], output_names=["logits"]),
        ModelSpec(
            name="MotionClassifier",
            load=motion_classifier.load_model,
            example_args=(torch.randn(1, 8, motion_classifier.FEATURE_DIM),),
            input_names=["motion"], output_names=["logits"],
            dynamic_shapes={"motion": {1: seq_len}}),
    ]


def convert(spec: ModelSpec, out_dir: Path) -> Path:
    model = spec.load(spec.checkpoint)
    exported = torch.export.export(
        model, spec.example_args, dynamic_shapes=spec.dynamic_shapes)
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported,
        input_names=spec.input_names,
        output_names=spec.output_names)
    # to_coreai() returns a program that is already optimized
    program = converter.to_coreai()
    asset = out_dir / f"{spec.name}.aimodel"
    program.save_asset(asset)
    return asset


def sha256(path: str | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(spec: ModelSpec, asset: Path) -> None:
    manifest = {
        "name": spec.name,
        "asset": asset.name,
        "function": "main",
        "inputs": spec.input_names,
        "outputs": spec.output_names,
        "example_shapes": [list(a.shape) for a in spec.example_args],
        "dynamic_dims": {k: {str(i): [d.min, d.max] for i, d in v.items()}
                         for k, v in (spec.dynamic_shapes or {}).items()},
        "checkpoint_sha256": sha256(spec.checkpoint),
        "tools": {"torch": version("torch"),
                  "coreai-torch": version("coreai-torch"),
                  "python": platform.python_version()},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = asset.with_suffix(".manifest.json")
    path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="build")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    for spec in specs():
        asset = convert(spec, out_dir)
        write_manifest(spec, asset)
        print(f"{spec.name} -> {asset}")
