import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x    # y = x^2 + 2x
y.backward()          # computes dy/dx
print(x.grad)         # tensor(8.) because dy/dx = 2x + 2 = 8 when x = 3
