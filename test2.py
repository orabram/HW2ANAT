import numpy as np
from scipy.sparse import csgraph

def laplacian(window_size,range):
    x = np.linspace(-range, range, window_size)
    xx = np.outer(x, x)
    return csgraph.laplacian(xx, normed=False)


kernel_size = 7
C = laplacian(pow(kernel_size, 2) , kernel_size / 2 )
D = np.matmul(np.transpose(C), C)
print(C)