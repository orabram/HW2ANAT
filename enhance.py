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
from PIL import Image
# from AnnoyKNN import *

ALPHA = 2
PATCH_SIZE = 50
PATCH_SIZE_RESCALED = int(PATCH_SIZE / ALPHA)
KERNEL_SIZE= 7
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
    width = len(patch)
    height = len(patch[0])
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


def convolve_patches(patches, kernel):
    convolved_patches = []
    for patch in patches:
        convolved_patch = convolve2d(patch, kernel, 'same')
        #convolved_patch = convolved_patch[PATCH_SIZE_RESCALED: PATCH_SIZE_RESCALED * 2, PATCH_SIZE_RESCALED: PATCH_SIZE_RESCALED * 2]
        convolved_patches.append(convolved_patch)
    return convolved_patches


def calc_weights_nominative(q, r):
    return np.exp(((-0.5) * math.pow((numpy.linalg.norm(q - r)) / SIGMA, 2)))


def calc_weights(r, q):
    weights = [[0] * len(r) for i in range(len(q))]
    for i in range(len(q)):
        sum = 0
        for j in range(len(r)):
            sum += calc_weights_nominative(q[i], r[j])
        for j in range(len(r)):
            weights[i][j] = calc_weights_nominative(q[i], r[j]) / sum
    return weights

def calc_k(r, q, weights):
    first_matrix = [[0.] * PATCH_SIZE_RESCALED for i in range(PATCH_SIZE_RESCALED)]
    second_matrix = [[0.] * PATCH_SIZE_RESCALED for i in range(PATCH_SIZE_RESCALED)]
    first_matrix = np.array(first_matrix)
    second_matrix = np.array(second_matrix)

    for i in range(len(q)):
        for j in range(len(r)):
            first_matrix += weights[i][j] * np.matmul(np.transpose(r[j]), r[j])
            second_matrix += weights[i][j] * np.matmul(np.transpose(r[j]), q[i])
    first_matrix = first_matrix / pow(SIGMA, 2)
    k_hat = np.matmul(np.linalg.inv(first_matrix), second_matrix)
    return k_hat

def main():
    img = plt.imread('DIPSourceHW2.png')
    img = np.array(img)
    img = img[:, :, 0]
    delta = fftpack.fftshift(scipy.signal.unit_impulse((KERNEL_SIZE, KERNEL_SIZE)))
    k_hat = delta
    gaussianKernel = gkern(PATCH_SIZE, SIGMA)
    sincKernel = skern(PATCH_SIZE, WINDOW)

    gaussImg = convolve2d(img, gaussianKernel, 'same')
    sincImg = convolve2d(img, sincKernel, 'same')

    downsampledGImg = downscale(gaussImg, ALPHA)
    downsampleSImg = downscale(sincImg, ALPHA)

    #show_image(img, 'image_original')
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
    q = low_res_patches_q_gauss
    iter = 1
    while last_rmse_dif > 0:
        last_k = k_hat
        downsampled_patches = calc_downscaled_patches(r, ALPHA)
        print("shrunk patches")
        convolved_patches = convolve_patches(downsampled_patches, k_hat)
        print("convolved patches")
        weights = calc_weights(convolved_patches, q)
        print("calculated weights")
        k_hat = calc_k(downsampled_patches, q, weights)
        show_and_save(k_hat, "new_kernel_" + str(iter))
        new_convolved_image = convolve2d(gaussImg, k_hat, 'same')
        show_and_save(new_convolved_image, "new_image_" + str(iter))
        iter += 1
        last_rmse_dif = RMSE(new_convolved_image, img)


if __name__ == "__main__":
    main()
