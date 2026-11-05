import asyncio
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import AIModel, NDArray

from models import image_classifier, motion_classifier
from models.image_classifier import load_model, IMAGE_SIZE
from parity import write_references, image_samples, motion_samples


async def main():
    pt_model = load_model()
    ai_model = await AIModel.load("build/ImageClassifier.aimodel")
    function = ai_model.load_function("main")

    image = np.random.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)

    # PyTorch reference
    with torch.no_grad():
        expected = pt_model(torch.from_numpy(image)).numpy()

    # Core AI inference
    result = await function({"image": NDArray(data=image)})
    actual = result["logits"].numpy()

    max_diff = np.max(np.abs(expected - actual))
    print(f"max absolute difference: {max_diff:.2e}")
    assert max_diff < 1e-3, "Core AI output diverges from PyTorch"


asyncio.run(main())

# Reference outputs for the Swift team
rng = np.random.default_rng(seed=0)
write_references(image_classifier.load_model(), image_samples(rng, n_random=5),
                 Path("build/ImageClassifier.references.json"), "image")
write_references(motion_classifier.load_model(), motion_samples(rng, n_random=4),
                 Path("build/MotionClassifier.references.json"), "motion")
