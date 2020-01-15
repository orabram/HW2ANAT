import numpy as np

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

def convmtx(v, n):
    """Generates a convolution matrix

    Usage: X = convm(v,n)
    Given a vector v of length N, an N+n-1 by n convolution matrix is
    generated of the following form:
              |  v(0)  0      0     ...      0    |
              |  v(1) v(0)    0     ...      0    |
              |  v(2) v(1)   v(0)   ...      0    |
         X =  |   .    .      .              .    |
              |   .    .      .              .    |
              |   .    .      .              .    |
              |  v(N) v(N-1) v(N-2) ...  v(N-n+1) |
              |   0   v(N)   v(N-1) ...  v(N-n+2) |
              |   .    .      .              .    |
              |   .    .      .              .    |
              |   0    0      0     ...    v(N)   |
    And then it's trasposed to fit the MATLAB return value.
    That is, v is assumed to be causal, and zero-valued after N.
    """
    N = len(v) + 2 * n - 2
    xpad = np.concatenate([np.zeros(n - 1), v[:], np.zeros(n - 1)])
    X = np.zeros((len(v) + n - 1, n))
    # Construct X column by column
    for i in range(n):
        X[:, i] = xpad[n - i - 1:N - i]

    return X.transpose()

def expand_kernel(img, kernel):
    pad_lim_x = (np.shape(img)[0] - np.shape(kernel)[0]) / 2
    pad_lim_y = (np.shape(img)[1] - np.shape(kernel)[1]) / 2
    pad_lim_x_0 = pad_lim_x
    pad_lim_x_1 = pad_lim_x
    pad_lim_y_0 = pad_lim_y
    pad_lim_y_1 = pad_lim_y
    if np.mod(np.shape(img)[0] - np.shape(kernel)[0], 2) == 1:
        pad_lim_x_0 = pad_lim_x + 0.5
        pad_lim_x_1 = pad_lim_x - 0.5
    if np.mod(np.shape(img)[1] - np.shape(kernel)[1], 2) == 1:
        pad_lim_y_0 = pad_lim_y + 0.5
        pad_lim_y_1 = pad_lim_y - 0.5
    padded_kernel = np.pad(kernel, ((int(pad_lim_x_0), int(pad_lim_x_1)), (int(pad_lim_y_0), int(pad_lim_y_1))),
                           'constant', constant_values=((0, 0), (0, 0)))
    return padded_kernel

input_mat = [[16, 24, 32], [47, 18, 26], [68, 12, 9]]
kernel = [[0, -1], [1, 0]]
input_mat = np.array(input_mat).reshape((9, 1))
print(input_mat)
kernel = np.array(kernel).flatten()
#input_mat = im2col(input_mat, (1, 3))
#print(input_mat)
#input_mat = input_mat.reshape((1, 9))
#print(input_mat)
mat = convmtx(kernel, 4)
print(mat)
print(np.matmul(mat, input_mat))