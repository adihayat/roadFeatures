
import numpy as np
import scipy.sparse
import pyamg


def RLaplacian(I):
    region_size = I.shape
    A = scipy.sparse.identity(np.prod(region_size), format='lil')
    for y in range(region_size[0]):
            for x in range(region_size[1]):
                    index = x + y * region_size[1]
                    A[index, index] = 4
                    if index + 1 < np.prod(region_size):
                        A[index, index + 1] = -1
                    if index - 1 >= 0:
                        A[index, index - 1] = -1
                    if index + region_size[1] < np.prod(region_size):
                        A[index, index + region_size[1]] = -1
                    if index - region_size[1] >= 0:
                        A[index, index - region_size[1]] = -1
    A = A.tocsr()
    x = pyamg.solve(A, I.flatten(), verb=True, tol=1e-10)
    x = x.reshape(region_size)
    return x

