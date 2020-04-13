import torch

t = torch.FloatTensor([0.3118])
print(t)
newt = list(t.numpy())[0]
print(newt)
print('Hoi')