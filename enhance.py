from scipy.signal import convolve2d
from scipy.sparse.linalg import bicg
from scipy.sparse.linalg import gmres
import cv2
from scipy.optimize import fmin_cg
from scipy.optimize import minimize
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy.fft as fft
import numpy as np
import numpy.matlib
from skimage.transform import rotate
import sys
import sklearn
import sklearn.neighbors as neighbors

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import aslinearoperator
import scipy
from scipy.sparse.linalg import bicg, bicgstab
from scipy.signal import convolve2d
from scipy import fftpack, signal
import math
from scipy.sparse import csgraph
from PIL import Image
from scipy.fftpack import fft2, ifft2, fftshift
# from AnnoyKNN import *

ALPHA = 2
PATCH_SIZE = 50
PATCH_SIZE_RESCALED = int(PATCH_SIZE / ALPHA)
KERNEL_SIZE = 7
SIGMA = 1
WINDOW = 7
T = 10


def show_image(img, title):
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()


def show_and_save(img, title):
    show_image(img, title)
    plt.imsave(title + '.png', img, cmap='gray')


def downscale(x, alpha, show=False):
    '''
     @input x: blurred img x
     @input alpha: scaling factor

     @return: downscaled image (x/alpha)
    '''
    downscaled_image = cv2.resize(x, None, fx=1 / alpha, fy=1 / alpha, interpolation=cv2.INTER_LANCZOS4)
    if show:
        show_and_save(downscaled_image, 'downscaled_original_image')
    else:
        plt.imsave('downscaled_original_image.png', downscaled_image, cmap='gray')

    return downscaled_image


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

def col2mat(mat, sz):
    width, height = mat.shape
    #height = sz[1]
    #width = sz[0]
    mats = []
    for i in range(height):
        new_mat = [[0] * sz[0] for m in range(sz[1])]
        for j in range(sz[0]):
            for k in range(sz[1]):
                try:
                    new_mat[j][k] = mat[k + sz[0] * j][i]
                except:
                    new_mat[j][k] = 0
        #print(new_mat)
        mats.append(new_mat)
        print(i)
    return mats

def im2col_new(A, sz):
    mat = []
    for j in range(0, A.shape[1] - sz[1] + 1):
        for i in range(0, A.shape[0] - sz[0] + 1):
            window = A[i:i + sz[0], j:j + sz[1]].reshape((KERNEL_SIZE, KERNEL_SIZE))
            window = np.transpose(window)# each individual window
            window = np.reshape(window, (pow(KERNEL_SIZE, 2)))
            mat.append(window)
    return mat

def gkern(kernlen, std):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def skern(kernlen, window):
    """Returns a 2D sinc kernel array"""
    x = np.linspace(-(kernlen / 2), (kernlen / 2), window)
    xx = np.outer(x, x)
    kern = np.sinc(xx)
    return kern

def laplacian(window_size,range):
    x = np.linspace(-range, range, window_size)
    xx = np.outer(x, x)
    return csgraph.laplacian(xx, normed=False)

def RMSE(img1, img2):
    squared_diff = (img1 - img2) ** 2
    summed = np.sum(squared_diff)
    return math.sqrt(summed)

def create_patches(img, patch_size):
    patches = []
    for i in range(0, int(img.shape[0] - patch_size), patch_size):
        for j in range(0, int(img.shape[1] - patch_size), patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches


def downscale_patches(patch, alpha):
    width = PATCH_SIZE
    height = PATCH_SIZE
    downscaled_width = int(width / alpha)
    downscaled_height = int(height / alpha)
    downscaled_patch = [[0] * downscaled_height for i in range(downscaled_width)]
    for i in range(downscaled_width):
        for j in range(downscaled_height):
            downscaled_patch[i][j] = patch[i * alpha][j * alpha]
    return downscaled_patch

def calc_downscaled_patches(patches, alpha):
    downscaled_patches = []
    for patch in patches:
        downscaled_patches.append(downscale_patches(patch, alpha))
    return downscaled_patches


def convolve_patches(Rjs, kernel):
    #col_kernel = im2col(kernel, (KERNEL_SIZE, KERNEL_SIZE))
    col_r = []
    for rj in Rjs:
        col_r.append(np.matmul(rj, kernel))
    return col_r

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
"""
def convolve_patches(patches, kernel):
    convolved_patches = []
    for patch in patches:
        convolved_patch = convolve2d(patch, kernel, 'same')
        #convolved_patch = convolved_patch[PATCH_SIZE_RESCALED: PATCH_SIZE_RESCALED * 2, PATCH_SIZE_RESCALED: PATCH_SIZE_RESCALED * 2]
        convolved_patches.append(convolved_patch)
    return convolved_patches
"""
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


def calc_weights_nominative(q, r):
    return np.exp((-0.5) * math.pow((numpy.linalg.norm(q - r)) / SIGMA, 2))


def calc_weights(r, q):
    weights = np.zeros((len(q), len(r)), dtype=float)#[[0] * len(r) for i in range(len(q))]
    for i in range(len(q)):
        sum = 0
        for j in range(len(r)):
            sum += calc_weights_nominative(q[i], r[j])
        for j in range(len(r)):
            weights[i][j] = calc_weights_nominative(q[i], r[j]) / sum
    return weights

def create_S(m, n):
    S = np.zeros((math.ceil(pow(m/ALPHA, 2)), pow(m-n+1, 2)))
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

def create_Rj(patch):
    C = np.array(im2col_new(patch, (KERNEL_SIZE, KERNEL_SIZE)))
    S = np.array(create_S(patch.shape[0], KERNEL_SIZE))
    Rj = np.matmul(S, C)
    return Rj

def calc_Rj(r):
    Rj = []
    for patch in r:
        Rj.append(create_Rj(patch))
    return Rj

def unpack_q(q):
    unpacked_q = []
    for patch in q:
        #patch = np.array(patch)
        patch = np.reshape(patch, (pow(PATCH_SIZE_RESCALED, 2), 1))
        unpacked_q.append(patch)
    return unpacked_q


def calc_k(r, q, weights, C):
    first_matrix = np.zeros((pow(KERNEL_SIZE, 2), pow(KERNEL_SIZE, 2)), dtype=float)#[[0.] * pow(KERNEL_SIZE, 2) for i in range(pow(KERNEL_SIZE, 2))]
    second_matrix = np.zeros((pow(KERNEL_SIZE, 2), 1), dtype=float)#[[0.] * 1 for i in range(pow(KERNEL_SIZE, 2))]
    #first_matrix = np.array(first_matrix)
    #second_matrix = np.array(second_matrix)

    for i in range(len(q)):
        for j in range(len(r)):
            first_matrix += weights[i][j] * np.matmul(np.transpose(r[j]), r[j]) + C
            second_matrix += weights[i][j] * np.matmul(np.transpose(r[j]), q[i])
    first_matrix = first_matrix #/ pow(SIGMA, 2)
    k_hat = np.matmul(np.linalg.inv(first_matrix), second_matrix)
    return k_hat

def weiner_filter(img, kernel, K=0.01):
    padded_kernel = expand_kernel(img, kernel)
    y = np.copy(img)
    Y = fft2(y)
    H = fft2(padded_kernel)
    G = np.conj(H) / (np.abs(H) ** 2 + K)
    Y_hat = G * Y
    y_hat = np.abs(fftshift(ifft2(Y_hat)))
    return y_hat

def main():
    img = plt.imread('DIPSourceHW2.png')
    img = np.array(img)
    img = img[:, :, 0]
    delta = fftpack.fftshift(scipy.signal.unit_impulse((KERNEL_SIZE, KERNEL_SIZE)))
    k_hat = np.reshape(delta, (KERNEL_SIZE * KERNEL_SIZE, 1))
    gaussianKernel = gkern(PATCH_SIZE, SIGMA)
    sincKernel = skern(PATCH_SIZE, WINDOW)

    gaussImg = convolve2d(img, gaussianKernel, 'same')
    sincImg = convolve2d(img, sincKernel, 'same')

    downsampledGImg = downscale(gaussImg, ALPHA)
    downsampleSImg = downscale(sincImg, ALPHA)

    show_image(img, 'image_original')
    #show_and_save(gaussImg, 'image_gaussian')
    #show_and_save(sincImg, 'image_sinc')
    #show_and_save(downsampledGImg, 'image_gaussian_downsampled')
    #show_and_save(downsampleSImg, 'image_sinc_downsampled')
    #show_and_save(gaussianKernel, 'kernel_gaussian')
    #show_and_save(sincKernel, 'kernel_sinc')

    gaussImg = numpy.array(gaussImg)
    sincImg = numpy.array(sincImg)
    #ps = [PATCH_SIZE, PATCH_SIZE]
    #psr = [PATCH_SIZE_RESCALED, PATCH_SIZE_RESCALED]
    low_res_patches_r_gauss = create_patches(img, PATCH_SIZE)
    low_res_patches_q_gauss = create_patches(img, PATCH_SIZE_RESCALED)
    #low_res_patches_r_gauss = col2mat(im2col(gaussImg, ps), ps)
    #low_res_patches_q_gauss = col2mat(im2col(gaussImg, psr), psr)
    #low_res_patches_r_sinc = col2mat(im2col(sincImg,  ps), ps)
    #low_res_patches_q_sinc = col2mat(im2col(sincImg, psr), psr)
    #low_res_patches_all = [[low_res_patches_r_gauss, low_res_patches_q_gauss],
                           #[low_res_patches_r_sinc, low_res_patches_q_sinc]]

    last_rmse_dif = 1
    r = low_res_patches_r_gauss
    r = np.array(r)
    q = low_res_patches_q_gauss
    q = np.array(q)
    iter = 1
    rmserror = 0
    Rj = calc_Rj(r)
    D = laplacian(pow(KERNEL_SIZE, 2) , pow(KERNEL_SIZE, 2) / 2 )
    C = np.matmul(np.transpose(D), D)

    while last_rmse_dif > 0:
        last_k = k_hat
        col_r = convolve_patches(Rj, last_k)
        col_q = unpack_q(q)
        #convolved_patches = convolve_patches(r, last_k)
        #print("convolved patches")
        #downsampled_patches = calc_downscaled_patches(convolved_patches, ALPHA)
        #print("shrunk patches")
        #col_r, col_q = columnise_patches(downsampled_patches, q)
        print("columnised patches")
        weights = calc_weights(col_r, col_q)
        print("calculated weights")
        #Rj = calc_Rj(col_r, last_k)
        k_hat = np.reshape(calc_k(Rj, col_q, weights, C), (KERNEL_SIZE, KERNEL_SIZE))
        show_and_save(k_hat, "new_kernel_" + str(iter))
        new_convolved_image = weiner_filter(gaussImg, k_hat)
        show_and_save(new_convolved_image, "new_image_" + str(iter))
        iter += 1
        last_rmse_dif = RMSE(new_convolved_image, img) - rmserror
        print(last_rmse_dif)
        rmserror = RMSE(new_convolved_image, img)
        k_hat = k_hat.reshape((49, 1))




if __name__ == "__main__":
    main()
