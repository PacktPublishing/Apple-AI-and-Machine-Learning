"""Interactive steps for the activity model, with a dynamic dimension.

Run from the Chapter01 folder: python -m snippets.export_motion_classifier
"""
import asyncio
from pathlib import Path

import numpy as np
import torch
import coreai_torch
from coreai.runtime import AIModel, NDArray

from models.motion_classifier import load_model, FEATURE_DIM, MAX_SEQ_LEN

# Declaring a dynamic dimension
model = load_model()
seq_len = torch.export.Dim("seq_len", min=1, max=MAX_SEQ_LEN)
exported = torch.export.export(
    model,
    args=(torch.randn(1, 8, FEATURE_DIM),),
    dynamic_shapes={"motion": {1: seq_len}},
)
print(exported)

# Conversion, identical to the image classifier's
decomposed = exported.run_decompositions(coreai_torch.get_decomp_table())
program = coreai_torch.TorchConverter().add_exported_program(
    decomposed, input_names=["motion"], output_names=["logits"],
).to_coreai()
Path("build").mkdir(exist_ok=True)
program.save_asset(Path("build/MotionClassifier.aimodel"))


async def smoke_test_lengths():
    model = await AIModel.load("build/MotionClassifier.aimodel")
    main = model.load_function("main")
    for length in (1, 50, 128, 256):   # 1 sample, 1 s, 2.5 s, 5 s
        x = np.random.randn(1, length, 6).astype(np.float32)
        out = await main({"motion": NDArray(data=x)})
        print(length, out["logits"].numpy().shape)   # always (1, 4)


asyncio.run(smoke_test_lengths())
