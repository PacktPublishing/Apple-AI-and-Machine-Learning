"""Troubleshooting: a branch that depends on data, and its tensor-only fix.

Run from the Chapter01 folder: python -m snippets.export_pitfalls
"""
import torch

from models.motion_classifier import MotionClassifier, FEATURE_DIM, MAX_SEQ_LEN


class WithPythonBranch(MotionClassifier):
    def forward(self, motion):
        if motion.abs().max() > 20:            # depends on the tensor's VALUE
            motion = motion / motion.abs().max()
        return super().forward(motion)


class WithTensorOps(MotionClassifier):
    def forward(self, motion):
        peak = motion.abs().amax(dim=(1, 2), keepdim=True)
        motion = torch.where(peak > 20, motion / peak, motion)   # no if
        return super().forward(motion)


seq_len = torch.export.Dim("seq_len", min=1, max=MAX_SEQ_LEN)
args = (torch.randn(1, 8, FEATURE_DIM),)
dynamic_shapes = {"motion": {1: seq_len}}

for model_class in (WithPythonBranch, WithTensorOps):
    model = model_class().eval()
    try:
        torch.export.export(model, args, dynamic_shapes=dynamic_shapes)
        print(f"{model_class.__name__}: export succeeded")
    except Exception as error:
        last_line = str(error).strip().splitlines()[0]
        print(f"{model_class.__name__}: export failed ({type(error).__name__})")
        print(f"    {last_line}")
