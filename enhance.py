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
#from AnnoyKNN import *

alpha = 2
patch_size = 7
patch_size_rescaled = patch_size / alpha
sigma = 1
window = 7


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

def calc_weights_nominative(q, r, sigma):
    return np.exp(((-0.5) * math.pow((numpy.linalg.norm(q - r))/sigma, 2)))

def calc_weights(i, low_res_patches, high_res_patches, k):
    q = high_res_patches[i]


def main():
    img = plt.imread('DIPSourceHW2.png')
    img = np.array(img)
    img = img[:, :, 0]
    delta = fftpack.fftshift(scipy.signal.unit_impulse((patch_size,patch_size)))
    kHat = delta
    gaussianKernel = gkern(patch_size, sigma)
    sincKernel = skern(patch_size, window)

    gaussImg = convolve2d(img, gaussianKernel)
    sincImg = convolve2d(img, sincKernel)

    downsampledGImg = downscale(gaussImg, alpha)
    downsampleSImg = downscale(sincImg, alpha)

    show_image(img, 'image_original')
    show_and_save(gaussImg, 'image_gaussian')
    show_and_save(sincImg, 'image_sinc')
    show_and_save(downsampledGImg, 'image_gaussian_downsampled')
    show_and_save(downsampleSImg, 'image_sinc_downsampled')
    show_and_save(gaussianKernel, 'kernel_gaussian')
    show_and_save(sincKernel, 'kernel_sinc')

    #knn = AnnoyKNN(patchSize, 'euclidean')
    #knn.make_index(downsampleSImg)
    #knn.build(n_trees)


if __name__ =="__main__":
    main()

