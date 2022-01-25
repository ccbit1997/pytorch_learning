import torch
import torchvision
from torch.nn import L1Loss, MSELoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

loss = L1Loss(reduction="mean")
result = loss(inputs, target)

loss_mse = MSELoss(reduction="mean")
result_mse = loss_mse(inputs, target)

inputs = torch.tensor([0.1, 0.2, 0.3])
target = torch.tensor([1])
inputs = torch.reshape(inputs, (1, 3))

ce_loss = torch.nn.CrossEntropyLoss()
result_ce = ce_loss(inputs, target)

print("ce_loss is: {}".format(result_ce))
print(result_mse)
