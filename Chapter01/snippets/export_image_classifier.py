"""Interactive steps for the image classifier, gathered in one script.

Run from the Chapter01 folder: python -m snippets.export_image_classifier
"""
import asyncio
from pathlib import Path

import numpy as np
import torch
import coreai_torch
from coreai.runtime import AIModel, NDArray

from models.image_classifier import load_model, IMAGE_SIZE

# Exporting with torch.export
model = load_model()
# An example image: a batch of one RGB image, values between 0 and 1
example_input = (torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE),)
exported_program = torch.export.export(model, example_input)
print(exported_program)


# Reading the graph and applying the decomposition table
def operators(program) -> list[str]:
    """Sorted list of the distinct ATen operations in an exported program."""
    return sorted({str(node.target)
                   for node in program.graph.nodes
                   if node.op == "call_function"})


print(operators(exported_program))
decomposed = exported_program.run_decompositions(
    coreai_torch.get_decomp_table()
)
print(operators(decomposed))

# Converting with coreai-torch
converter = coreai_torch.TorchConverter()
converter.add_exported_program(
    decomposed,
    input_names=["image"],
    output_names=["logits"],
)
# to_coreai() returns a program that is already optimized
program = converter.to_coreai()
Path("build").mkdir(exist_ok=True)
program.save_asset(Path("build/ImageClassifier.aimodel"))


# A first inference from Python
async def smoke_test():
    model = await AIModel.load("build/ImageClassifier.aimodel")
    main = model.load_function("main")
    image = np.random.rand(1, 3, 128, 128).astype(np.float32)
    result = await main({"image": NDArray(data=image)})
    logits = result["logits"].numpy()
    print(logits.shape, logits.dtype)   # (1, 4) float32


asyncio.run(smoke_test())
