import json
from dataclasses import dataclass

import numpy as np
import torch
from coreai.runtime import NDArray


@dataclass
class ParityReport:
    samples: int
    max_abs_diff: float
    mean_abs_diff: float
    min_psnr: float
    argmax_agreement: float   # share of identical predicted classes


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Peak signal-to-noise ratio, in decibels."""
    mse = np.mean((reference - candidate) ** 2)
    if mse == 0:
        return float("inf")
    peak = np.max(np.abs(reference))
    return 20 * np.log10(peak) - 10 * np.log10(mse)


async def compare(pt_model, function, samples, input_name,
                  output_name="logits") -> ParityReport:
    diffs, psnrs, agreements = [], [], []
    for x in samples:
        with torch.no_grad():
            expected = pt_model(torch.from_numpy(x)).numpy()
        result = await function({input_name: NDArray(data=x)})
        actual = result[output_name].numpy()
        diffs.append(np.abs(expected - actual).ravel())
        psnrs.append(psnr(expected, actual))
        agreements.append(np.argmax(expected) == np.argmax(actual))
    all_diffs = np.concatenate(diffs)
    return ParityReport(
        samples=len(samples),
        max_abs_diff=float(all_diffs.max()),
        mean_abs_diff=float(all_diffs.mean()),
        min_psnr=float(min(psnrs)),
        argmax_agreement=float(np.mean(agreements)),
    )


def image_samples(rng, n_random=50, size=128):
    samples = [rng.random((1, 3, size, size), dtype=np.float32)
               for _ in range(n_random)]
    samples += [np.zeros((1, 3, size, size), np.float32),   # black image
                np.ones((1, 3, size, size), np.float32)]    # white image
    return samples


def motion_samples(rng, n_random=50, max_len=256):
    lengths = [1, 2, 50, 128, max_len - 1, max_len]          # bounds and landmarks
    lengths += list(rng.integers(1, max_len + 1, size=n_random))
    return [rng.standard_normal((1, int(t), 6), dtype=np.float32)
            for t in lengths]


def write_references(pt_model, samples, path, input_name,
                     output_name="logits"):
    references = []
    for x in samples:
        with torch.no_grad():
            y = pt_model(torch.from_numpy(x)).numpy()
        references.append({
            input_name: {"shape": list(x.shape), "data": x.ravel().tolist()},
            output_name: {"shape": list(y.shape), "data": y.ravel().tolist()},
            "argmax": int(np.argmax(y)),
        })
    path.write_text(json.dumps(references))
