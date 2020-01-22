from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import numpy as np
import numpy.matlib
import scipy
from scipy.signal import convolve2d
from scipy import fftpack, signal
import math
from scipy.sparse import csgraph
from scipy.fftpack import fft2, ifft2, fftshift

ALPHA = 2
PATCH_SIZE = 50
PATCH_SIZE_RESCALED = int(PATCH_SIZE / ALPHA)
KERNEL_SIZE = 7
SIGMA = 1
WINDOW = 7


# Displays a grayscale image to the user
def show_image(img, title):
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()


# Displays a grayscale image to the user and also saves it
def show_and_save(img, title):
    show_image(img, title)
    plt.imsave(title + '.png', img, cmap='gray')


# This is the MATLAB im2col function
def im2col(A, sz):
    m, n = A.shape
    s1, s2 = A.strides
    rows = m - sz[0] + 1
    cols = n - sz[1] + 1
    shp = sz[0], sz[1], rows, cols
    strd = s1, s2, s1, s2

    out = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd).reshape(sz[0] * sz[1], -1)[:, ::1]
    return out


# This function creates the convolution matrix by sliding a window of size sz across the image A
def im2col_new(A, sz):
    mat = []
    for j in range(0, A.shape[1] - sz[1] + 1):
        for i in range(0, A.shape[0] - sz[0] + 1):
            window = A[i:i + sz[0], j:j + sz[1]].reshape((KERNEL_SIZE, KERNEL_SIZE))
            window = np.transpose(window)  # each individual window
            window = np.reshape(window, (pow(KERNEL_SIZE, 2)))
            mat.append(window)
    return mat


def mseCalc(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


# This function calculates the PSNR between two images
def psnr(img1, img2):
    mse = mseCalc(img1, img2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))


# This function returns a 2D Gaussian kernel
def gkern(kernlen, std):
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


# This function returns a 2D Sinc kernel
def skern(kernlen, window):
    x = np.linspace(-(kernlen / 2), (kernlen / 2), window)
    xx = np.outer(x, x)
    kern = np.sinc(xx)
    return kern


# This function returns the Laplacian matrix
def laplacian(window_size, range):
    x = np.linspace(-range, range, window_size)
    xx = np.outer(x, x)
    return csgraph.laplacian(xx, normed=False)


# This function calculates the RMSE between two images
def RMSE(img1, img2):
    squared_diff = (img1 - img2) ** 2
    summed = np.sum(squared_diff)
    return math.sqrt(summed)


# This function creates patches of size patch_size out of an image
def create_patches(img, patch_size):
    patches = []
    for i in range(0, int(img.shape[0] - patch_size), patch_size):
        for j in range(0, int(img.shape[1] - patch_size), patch_size):
            patch = img[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
    return patches


# This function calculates r_j_alpha by multiplying every Rj matrix with the kernel vector
def convolve_patches(Rjs, kernel):
    col_r = []
    for rj in Rjs:
        col_r.append(np.matmul(rj, kernel))
    return col_r


# This function adds padding to the kernel so that it'll be the same dimension as a given image. Used
# for the Wiener filter
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


# This function turns every patch within the sets r and q into column vectors
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


# This function calculates the weights for given sets of r and q
def calc_weights(r, q):
    weights = np.zeros((len(q), len(r)), dtype=float)  # [[0] * len(r) for i in range(len(q))]
    for i in range(len(q)):
        sum = 0
        for j in range(len(r)):
            sum += calc_weights_nominative(q[i], r[j])
        for j in range(len(r)):
            weights[i][j] = calc_weights_nominative(q[i], r[j]) / sum
    return weights


# The matrix S is an identity matrix with zero lines corresponding to the ALPHA constant. It is used as
# a downsampling matrix.
def create_S(m, n):
    S = np.zeros((math.ceil(pow(m / ALPHA, 2)), pow(m - n + 1, 2)))
    i = 1
    j = 0
    while j < S.shape[0]:
        if i % ALPHA == 0:
            i += 1
        else:
            S[j][i - 1] = 1
            i += 1
        j += 1

    return S


# This function creates the Rj matrix by multiplying the downscaling matrix S and the convolution matrix C
def create_Rj(patch):
    C = np.array(im2col_new(patch, (KERNEL_SIZE, KERNEL_SIZE)))
    S = np.array(create_S(patch.shape[0], KERNEL_SIZE))
    Rj = np.matmul(S, C)
    return Rj


# This function calculates a corresponding Rj matrix for each patch in r
def calc_Rj(r):
    Rj = []
    for patch in r:
        Rj.append(create_Rj(patch))
    return Rj


# This function turns every patch in q into a column vector
def unpack_q(q):
    unpacked_q = []
    for patch in q:
        patch = np.reshape(patch, (pow(PATCH_SIZE_RESCALED, 2), 1))
        unpacked_q.append(patch)
    return unpacked_q


# This function calculates the kernel(intermediate step in the algorithm)
def calc_k(r, q, weights, C):
    first_matrix = np.zeros((pow(KERNEL_SIZE, 2), pow(KERNEL_SIZE, 2)), dtype=float)
    second_matrix = np.zeros((pow(KERNEL_SIZE, 2), 1), dtype=float)

    for i in range(len(q)):
        for j in range(len(r)):
            first_matrix += weights[i][j] * np.matmul(np.transpose(r[j]), r[j]) + C
            second_matrix += weights[i][j] * np.matmul(np.transpose(r[j]), q[i])
    first_matrix = first_matrix  # / pow(SIGMA, 2)
    k_hat = np.matmul(np.linalg.inv(first_matrix), second_matrix)
    return k_hat


# This function returns an image after it has passed a weiner filter
def weiner_filter(img, kernel, K=0.01):
    padded_kernel = expand_kernel(img, kernel)
    y = np.copy(img)
    Y = fft2(y)
    H = fft2(padded_kernel)
    G = np.conj(H) / (np.abs(H) ** 2 + K)
    Y_hat = G * Y
    y_hat = np.abs(fftshift(ifft2(Y_hat)))
    return y_hat


# This function calculates the kernel used to restore an image using internal patches
def optimize_kernel(img, ref_img, r, q, k_hat):
    last_rmse = 999999
    r = np.array(r)
    q = np.array(q)
    iter = 1
    rmserror = 999999
    Rj = calc_Rj(r)
    D = laplacian(pow(KERNEL_SIZE, 2), pow(KERNEL_SIZE, 2) / 2)
    C = np.matmul(np.transpose(D), D)
    while last_rmse - rmserror >= 0:
        last_rmse = rmserror
        rmserror = 0
        last_k = k_hat
        col_r = convolve_patches(Rj, last_k)
        col_q = unpack_q(q)
        print("columnised patches")
        weights = calc_weights(col_r, col_q)
        print("calculated weights")
        # Rj = calc_Rj(col_r, last_k)
        k_hat = np.reshape(calc_k(Rj, col_q, weights, C), (KERNEL_SIZE, KERNEL_SIZE))
        show_image(k_hat, "new_kernel_" + str(iter))
        new_convolved_image = weiner_filter(ref_img, k_hat)
        show_image(new_convolved_image, "new_image_" + str(iter))
        iter += 1
        rmserror = RMSE(new_convolved_image, img)
        print(rmserror)
        k_hat = k_hat.reshape((49, 1))

    return np.reshape(k_hat, (KERNEL_SIZE, KERNEL_SIZE))


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

    # show_image(img, 'image_original')
    # show_and_save(gaussImg, 'image_gaussian')
    # show_and_save(sincImg, 'image_sinc')
    # show_and_save(downsampledGImg, 'image_gaussian_downsampled')
    # show_and_save(downsampleSImg, 'image_sinc_downsampled')
    # show_and_save(gaussianKernel, 'kernel_gaussian')
    # show_and_save(sincKernel, 'kernel_sinc')

    gaussImg = numpy.array(gaussImg)
    sincImg = numpy.array(sincImg)
    low_res_patches_r_gauss = create_patches(img, PATCH_SIZE)
    low_res_patches_q_gauss = create_patches(img, PATCH_SIZE_RESCALED)
    low_res_patches_r_sinc = create_patches(img, PATCH_SIZE)
    low_res_patches_q_sinc = create_patches(img, PATCH_SIZE_RESCALED)

    k_gauss = optimize_kernel(img, gaussImg, low_res_patches_r_gauss, low_res_patches_q_gauss, k_hat)
    k_sinc = optimize_kernel(img, sincImg, low_res_patches_r_sinc, low_res_patches_q_sinc, k_hat)

    show_and_save(k_gauss, "gaussian kernel recovered")
    show_and_save(k_sinc, "sinc kernel recovered")

    igauss_kgauss = weiner_filter(gaussImg, k_gauss)
    show_and_save(igauss_kgauss, "igauss_kgauss")
    print("PSNR igauss_kgauss: " + str(psnr(img, igauss_kgauss)))
    igauss_ksinc = weiner_filter(gaussImg, k_sinc)
    show_and_save(igauss_ksinc, "igauss_ksinc")
    print("PSNR igauss_ksinc: " + str(psnr(img, igauss_ksinc)))
    isinc_kgauss = weiner_filter(sincImg, k_gauss)
    show_and_save(isinc_kgauss, "isinc_kgauss")
    print("PSNR isinc_kgauss: " + str(psnr(img, isinc_kgauss)))
    isinc_ksinc = weiner_filter(sincImg, k_sinc)
    show_and_save(igauss_kgauss, "isinc_ksinc")
    print("PSNR isinc_ksinc: " + str(psnr(img, isinc_ksinc)))


if __name__ == "__main__":
    main()
