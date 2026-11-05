import numpy as np
import torch

# Creating tensors and inspecting their properties
scalar = torch.tensor(3.0)                       # 0 dimensions
vector = torch.tensor([1.0, 2.0, 3.0])           # 1 dimension
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2 dimensions
batch = torch.randn(8, 16)                       # 8 vectors of 16 random values

print(matrix.shape)   # torch.Size([2, 2])
print(matrix.dtype)   # torch.float32
print(batch.shape)    # torch.Size([8, 16])
print(batch.device)   # cpu

# Element-wise operations and linear algebra
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

print(a + b)           # element-wise addition
print(a * 2)           # multiplication by a scalar
print(a @ b)           # matrix product: tensor([[70., 100.], [150., 220.]])
print(a.mean(dim=0))   # mean of each column: tensor([2., 3.])
print(a.reshape(4))    # same data, new shape: tensor([1., 2., 3., 4.])

# Moving between NumPy and PyTorch
array = np.random.randn(1, 256).astype(np.float32)
tensor = torch.from_numpy(array)     # NumPy -> PyTorch, no copy
back = tensor.numpy()                # PyTorch -> NumPy
print(tensor.shape, tensor.dtype)    # torch.Size([1, 256]) torch.float32
