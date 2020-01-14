from enhance import *
import numpy as np
from scipy.signal import convolve2d


KERNEL_SIZE = 2
ALPHA = 2

def downsample_zeros_matrix(mat, alpha):
    (mat_shape_x, mat_shape_y) = mat.shape
    for i in range(mat_shape_x):
        if (i % alpha):
            mat[i, :] = 0
            mat[:, i] = 0
    return mat

def downscale_patches(patch, alpha):
    width = len(patch)
    height = len(patch[0])
    downscaled_width = int(width / alpha)
    downscaled_height = int(height / alpha)
    downscaled_patch = [[0] * downscaled_height for i in range(downscaled_width)]
    for i in range(downscaled_width):
        for j in range(downscaled_height):
            downscaled_patch[i][j] = patch[i * alpha][j * alpha]
    return downscaled_patch

def im2col(A, sz):
    """
    @param A: input image
    @param sz: size of blocks

    @return out: columised image (same working as im2col of matlab)
    """

    m, n = A.shape
    s1, s2 = A.strides
    rows = m - sz[0] + 1
    cols = n - sz[1] + 1
    shp = sz[0], sz[1], rows, cols
    strd = s1, s2, s1, s2

    out = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd).reshape(sz[0] * sz[1], -1)[:, ::1]
    return out


def create_S(m, n):
    S = np.zeros((math.ceil(m/ALPHA), pow(m-n+1, 2)))
    i = 1
    j = 0
    while j < S.shape[0]:
        if i % ALPHA == 0:
            i += 1
        else:
            S[j][i-1] = 1
            i += 1
            j += 1

    return S


def create_Rj(patch, k):
    #C = np.transpose(im2col(patch, (KERNEL_SIZE, KERNEL_SIZE)))
    C = im2col(patch, (KERNEL_SIZE, KERNEL_SIZE))
    S = create_S(patch.shape[0], KERNEL_SIZE)
    Rj = np.matmul(S, C)
    return Rj


def im2col_new(A, sz):
    mat = []
    for j in range(0, A.shape[1] - sz[1] + 1):
        for i in range(0, A.shape[0] - sz[0] + 1):
            window = A[i:i + sz[0], j:j + sz[1]].reshape((2, 2))
            window = np.transpose(window)
            window = np.reshape(window, (4))
            mat.append(window)
    mat = np.array(mat)
    return mat


input = [[16, 24, 32], [47, 18, 26], [68, 12, 9]]
kernel = [[5, 3], [4, 2]]
input = np.array(input)
kernel = np.array(kernel)
#print(input)

#input = np.transpose(input)

#print(input)
#print(im2col(input, (2,2)))

mat = im2col_new(input, (2, 2))
#mat = np.reshape(4, 4)
print(mat)
#Rj = create_Rj(input, kernel)
#print(Rj)

#column_kernel = im2col(kernel, (KERNEL_SIZE, KERNEL_SIZE))

#print(np.matmul(Rj, column_kernel))

#print(convolve2d(kernel, input, 'same'))



"""
delta = fftpack.fftshift(scipy.signal.unit_impulse((KERNEL_SIZE, KERNEL_SIZE)))
#curr_k = delta

patch = [[2] * 9 for i in range(9)]

curr_k_2d = np.ones((5,5), dtype=float)
curr_k_2d[3][3] = 0
#curr_k = np.ones((25, 1), dtype=int)
epsilon = pow(10, -9)
alpha = 2
extended_k = np.eye(curr_k_2d.shape[0]) * epsilon
extended_k += curr_k_2d
#for i in range(len(curr_k)):
#    extended_k[i][0] = curr_k[i][0]
k_rj = signal.convolve2d(curr_k_2d, patch, mode='same')
k_inv = np.linalg.inv(extended_k)
downsample_zeros_k_rj = downsample_zeros_matrix(k_rj,alpha)
curr_Rj = np.matmul(downsample_zeros_k_rj, k_inv)
curr_Rj = np.array(curr_Rj)

print(curr_Rj)"""