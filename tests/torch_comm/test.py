import torch
from timeit import default_timer as timer

tensor_x = torch.rand(4096, 4096).to(torch.device("cuda", 0))

tensor_w_0 = torch.rand(4096, 4096).to(torch.device("cuda", 0))
tensor_w_1 = torch.rand(4096, 4096).to(torch.device("cuda", 0))

# sequentially execute on 1 GPU
start = timer()
tensor_y = torch.matmul(tensor_x, tensor_w_0)
tensor_y = torch.matmul(tensor_x, tensor_w_1)
end = timer()
print(f"sequential execution time: {end - start}")

# parallel execution on 2 GPUs, assuming w1 is on gpu1
tensor_w_1 = tensor_w_1.to(torch.device("cuda", 1))

start = timer()
tensor_y = torch.matmul(tensor_x, tensor_w_0)
# move tensor_x to gpu1
tensor_x = tensor_x.to(torch.device("cuda", 1))
tensor_y = torch.matmul(tensor_x, tensor_w_1)
end = timer()

print(f"parallel execution time: {end - start}")
