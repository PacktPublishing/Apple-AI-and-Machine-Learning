import torch
import torch.nn as nn

IMAGE_SIZE = 128
NUM_CLASSES = 4   # 0 = shoe, 1 = bag, 2 = watch, 3 = glasses


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Normalization statistics, one value per channel (R, G, B)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, NUM_CLASSES)

    def forward(self, image):
        x = (image - self.mean) / self.std   # normalization inside the model
        x = self.features(x)                 # extract visual patterns
        x = torch.flatten(x, 1)              # [1, 64, 1, 1] -> [1, 64]
        return self.classifier(x)            # one score per category


def load_model(checkpoint: str | None = None) -> ImageClassifier:
    """Creates the model and loads its weights, or reproducible weights."""
    torch.manual_seed(0)
    model = ImageClassifier()
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval()
