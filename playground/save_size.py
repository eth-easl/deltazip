import numpy as np
import cupy as cp

data = np.random.rand(10000, 100)
data_shape = data.shape
data = cp.array(data_shape)
print(data)