import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
a=torch.ones(1,1)
print(a.cuda())