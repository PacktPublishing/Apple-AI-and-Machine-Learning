import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_DIM = 6               # acceleration (x, y, z) + rotation (x, y, z)
D_MODEL = 64                  # size of the internal representation
N_HEADS = 4                   # number of attention heads
HEAD_DIM = D_MODEL // N_HEADS # 16 values per head
NUM_CLASSES = 4               # 0 = resting, 1 = walking, 2 = running, 3 = cycling
MAX_SEQ_LEN = 256             # about 5 seconds at 50 Hz


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
            nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # 1. Multi-head attention
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(batch, seq_len, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(batch, seq_len, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(batch, seq_len, N_HEADS, HEAD_DIM).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, D_MODEL)
        x = x + self.proj(attn)

        # 2. Fully connected network
        return x + self.mlp(self.norm2(x))


class MotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(FEATURE_DIM, D_MODEL)
        self.position = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN, D_MODEL))
        self.block = EncoderBlock()
        self.head = nn.Linear(D_MODEL, NUM_CLASSES)

    def forward(self, motion):
        seq_len = motion.shape[1]
        x = self.embed(motion) + self.position[:, :seq_len]
        x = self.block(x)
        return self.head(x.mean(dim=1))   # average over time


def load_model(checkpoint: str | None = None) -> MotionClassifier:
    torch.manual_seed(0)
    model = MotionClassifier()
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval()
