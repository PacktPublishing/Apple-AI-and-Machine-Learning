from pathlib import Path

import torch
import torch.nn as nn

torch.manual_seed(0)


def make_points(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    points = torch.rand(n, 2) * 2 - 1                 # shape [n, 2]
    labels = (points.norm(dim=1) < 0.7).long()        # 1 = inside the circle
    return points, labels


class CircleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 2),                         # two logits: outside, inside
        )

    def forward(self, points):
        return self.layers(points)


train_x, train_y = make_points(2000)
test_x, test_y = make_points(500)

# Training: the five-beat rhythm
model = CircleClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(300):
    logits = model(train_x)               # 1. forward pass
    loss = loss_fn(logits, train_y)       # 2. measure the error
    optimizer.zero_grad()                 # 3. reset the gradients
    loss.backward()                       # 4. compute the gradients
    optimizer.step()                      # 5. update the weights
    if epoch % 50 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(test_x).argmax(dim=1)
    accuracy = (predictions == test_y).float().mean().item()
print(f"test accuracy: {accuracy:.1%}")

# Saving the checkpoint and reloading it
Path("build").mkdir(exist_ok=True)
torch.save(model.state_dict(), "build/circle_classifier.pt")

for name, tensor in model.state_dict().items():
    print(name, tuple(tensor.shape))
# layers.0.weight (32, 2)
# layers.0.bias (32,)
# layers.2.weight (32, 32)
# ...

restored = CircleClassifier()
restored.load_state_dict(torch.load("build/circle_classifier.pt"))
restored.eval()
