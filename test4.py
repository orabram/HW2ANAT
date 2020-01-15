import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
from scipy import misc
from scipy.fftpack import fft2, ifft2, fftshift
from scipy.signal import convolve2d as conv2d
import scipy.misc
from pyunlocbox import functions, solvers


def blurring_function(k):
    def make_blurred(h):
        return conv2d(h, k, 'same')

    return make_blurred


def gkern(l=5, sig=1., alpha=1):
    """
  creates gaussian kernel with side length l and a sigma of sig
  """
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-((xx * alpha) ** 2 + (yy * alpha) ** 2) / (2. * sig ** 2))

    return kernel / np.sum(kernel)


def rect(leng, size, alpha=1):
    sx, sy = size
    PSF = np.zeros((sy, sx))
    PSF[int(sy / 2 - leng / (2 * alpha)):int(sy / 2 + leng / (2 * alpha)),
    int(sx / 2 - leng / (2 * alpha)):int(sx / 2 + leng / (2 * alpha))] = 1
    return PSF / PSF.sum()


def inverse_filter(inputs, PSF):
    input_fft = fft2(inputs)
    PSF_fft = fft2(PSF)
    input_fft_scaled = input_fft / input_fft.max()
    PSF_fft_scaled = PSF_fft / PSF_fft.max()
    PSF_fft_scaled[PSF_fft_scaled == 0] = 0.000005
    recovered_fft_scaled = input_fft_scaled / PSF_fft_scaled
    recovered_fft = recovered_fft_scaled * input_fft.max()
    result = ifft2(recovered_fft)
    result = np.real(fftshift(result))
    return result


## Wiener filter

def weiner_filter(img, kernel, K=0.01):
    padded_kernel = expand_kernel(img, kernel)
    y = np.copy(img)
    Y = fft2(y)
    H = fft2(padded_kernel)
    G = np.conj(H) / (np.abs(H) ** 2 + K)
    Y_hat = G * Y
    y_hat = np.abs(fftshift(ifft2(Y_hat)))
    return y_hat


def make_blurred(inputs, PSF):
    input_fft = fft2(inputs)
    PSF_fft = fft2(PSF)
    blurred = ifft2(input_fft * PSF_fft)
    blurred = np.abs(fftshift(blurred))
    return blurred


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


def main():
    ############################ sec 1+2 - construction of PSF ####################
    img = cv2.imread('DIPSourceHW2.png', 0)
    img = cv2.resize(img, (256, 256))
    #plt.imsave('img.png', img, cmap='gray')
    alpha = 2
    psf_l_gauss = gkern(l=9, sig=1)
    #plt.imsave('psf_l_gauss.png', psf_l_gauss, cmap='gray')
    psf_l_box = rect(4, (9, 9))
    #plt.imsave('psf_l_box.png', psf_l_box, cmap='gray')
    psf_h_gauss = alpha * gkern(l=9, sig=1, alpha=alpha)
    #plt.imsave('psf_h_gauss.png', psf_h_gauss, cmap='gray')
    psf_h_box = alpha * rect(4, (9, 9), alpha)
    #plt.imsave('psf_h_box.png', psf_h_box, cmap='gray')

    ############################ sec 3 - construct corresponding low and high res  ####################
    l_gauss = conv2d(img, psf_l_gauss, 'same')
    l_box = conv2d(img, psf_l_box, 'same')
    h_gauss = conv2d(img, psf_h_gauss, 'same')
    h_box = conv2d(img, psf_h_box, 'same')
    #plt.imsave('l_gauss.png', l_gauss, cmap='gray')
    #plt.imsave('l_box.png', l_box, cmap='gray')
    #plt.imsave('h_gauss.png', h_gauss, cmap='gray')
    #plt.imsave('h_box.png', h_box, cmap='gray')

    ############################ sec 4 - construct blur kernel k ####################
    ## we have conv(psf_h,k)=psf_l
    ## we'll convert it to matrix product: toeplitz(psf_h)*k=psf_l
    ## after we have toeplitz(psf_h) and psf_l, we can multiply by inverse matrix of toeplitz(psf_h) on the left and get

    flattened_psf_hr_gauss = psf_h_gauss.flatten()
    kron_mat_psf_hr_gauss = convmtx(flattened_psf_hr_gauss, 81).T
    padded_psf_lr_gauss = np.zeros([161, 1])
    padded_psf_lr_gauss[39:120, 0] = psf_l_gauss.flatten()
    K_gauss = np.matmul(np.linalg.pinv(kron_mat_psf_hr_gauss), padded_psf_lr_gauss)
    K_gauss = np.reshape(K_gauss, [9, 9])
    K_gauss = np.roll(K_gauss, (0, 1))
    plt.imsave('k_gauss.png', K_gauss, cmap='gray')

if __name__ == '__main__':
    main()