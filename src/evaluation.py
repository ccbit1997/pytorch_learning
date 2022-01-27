import torch

outputs = torch.tensor([[0.3, 0.5],
                        [0.2, 0.4]])

print(outputs.argmax(1)) #横向
print(outputs.argmax(0)) #纵向

inputs = torch.ones((64, 10), dtype=torch.float32)
print(sum(inputs.argmax(1)))