import torch

tensor_x = torch.rand(4096, 4096).to(torch.device("cuda", 0))

tensor_w_0 = torch.rand(4096, 4096).to(torch.device("cuda", 0))
tensor_w_1 = torch.rand(3, 2).to(torch.device("cuda", 1))

tensor_w = tensor_w.to(torch.device("cuda", 0))
tensor_y = torch.matmul(tensor_x, tensor_w)

print(tensor_y)
