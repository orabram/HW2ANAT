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

def columnise_patches(r, q):
    col_r = []
    col_q = []
    for patch in r:
        patch = np.array(patch)
        col_r.append(np.reshape(patch, (patch.shape[0] * patch.shape[1], 1)))

    for patch in q:
        patch = np.array(patch)
        col_q.append(np.reshape(patch, (patch.shape[0] * patch.shape[1], 1)))

    return col_r, col_q


r = [[[1] * 5 for m in range(5)]]
q = [[[1] * 5 for m in range(5)]]

r = np.array(r)
print(columnise_patches(r, q))


