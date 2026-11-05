import asyncio

import numpy as np
import pytest
from coreai.runtime import AIModel

from models import image_classifier, motion_classifier
from parity import compare, image_samples, motion_samples


def run(coro):
    return asyncio.run(coro)


async def load(name):
    model = await AIModel.load(f"build/{name}.aimodel")
    return model.load_function("main")


@pytest.fixture(scope="module")
def image_report():
    rng = np.random.default_rng(seed=42)
    fn = run(load("ImageClassifier"))
    return run(compare(image_classifier.load_model(), fn,
                       image_samples(rng), input_name="image"))


@pytest.fixture(scope="module")
def motion_report():
    rng = np.random.default_rng(seed=7)
    fn = run(load("MotionClassifier"))
    return run(compare(motion_classifier.load_model(), fn,
                       motion_samples(rng), input_name="motion"))


def test_image_numerics(image_report):
    assert image_report.max_abs_diff < 1e-3


def test_image_decisions(image_report):
    assert image_report.argmax_agreement == 1.0


def test_motion_numerics(motion_report):
    assert motion_report.max_abs_diff < 1e-2


def test_motion_decisions(motion_report):
    assert motion_report.argmax_agreement == 1.0
