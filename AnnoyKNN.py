from annoy import AnnoyIndex
import random
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt


patchSize = 7

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


class AnnoyKNN:
    def __init__(self, patch_size, method):
        self.patch_size = patch_size
        self.feature_dims = patch_size * patch_size
        self.KNN = AnnoyIndex(self.feature_dims, method)

    def make_index(self, img):
        """
            make indices to be used for search in KNN
            patches are flatten and treated as 1D vectors
        """
        img = im2col(img, [self.patch_size, self.patch_size])
        for i in range(img.shape[1]):
            self.KNN.add_item(i, (img.T)[i, :])

    def build(self, n_trees):
        """
            specify the no. of trees to be used
            more trees -> higher accuracy, higher search time
        """
        self.KNN.build(n_trees)

    def get_nn(self, patch):
        """
            input patch to be searched
            returns the nearest patch from indices
        """
        nn = self.KNN.get_nns_by_vector(patch.flatten(), 1)
        nn = self.KNN.get_item_vector(nn[0])
        return (np.array(nn)).reshape((self.patch_size, self.patch_size))

    def getPrior(self, img, patch_size, alpha):
        """
        @input img: size (n x m)
        @input patch_size
        @input alpha

        @return z: size (n x m)
        """

        start = time.time()
        M = patch_size
        x_alpha = downscale(img, alpha)
        n, m = img.shape

        KNN = AnnoyKNN(M, 'euclidean')
        KNN.make_index(x_alpha)
        KNN.make_index(img)
        KNN.build(10)

        z = np.zeros(img.shape)
        counts = np.zeros(img.shape)

        for i in range(M // 2, n - M // 2):
            for j in range(M // 2, m - M // 2):
                z[i - M // 2:i + M // 2 + 1, j - M // 2:j + M // 2 + 1] += KNN.get_nn(
                    img[i - M // 2:i + M // 2 + 1, j - M // 2:j + M // 2 + 1] + np.random.uniform(-0.05, 0.05,
                                                                                                  size=(M, M)))
                counts[i - M // 2:i + M // 2 + 1, j - M // 2:j + M // 2 + 1] += 1

        end = time.time()
        print('time taken:', end - start)

        return z / counts

    def call_func(self, x):
        """
        callback function for printing current iteration number in solver
        """
        global itera
        print("            Iter -> {}                  ".format(itera), end='\r')
        itera += 1

    def myConvolve(self, img, kernel_matrix, patch_size=patchSize):
        """
        @param img: input image
        @param kernel_matrix: kernel matrix containing kernel for each patch in img
        @param patch_size: patch_size to take in img

        @return outp: Kernel_matrix*img (convolution of img with kernels)
        """

        outp = np.zeros(img.shape)

        pd = patch_size // 2
        img_shape = img.shape
        pimg = np.pad(img, ((pd, pd), (pd, pd)), 'reflect')
        col_img = im2col(pimg, [patch_size, patch_size])
        outp = np.sum(col_img * kernel_matrix, axis=0)

        return outp.reshape(img_shape)

    def myConvolveLaplacian(self, img, patch_size=patchSize):
        """
        @param img: input image
        @param patch_size: patch size to take in image

        @return oimg: laplacian of img
        """
        pd = 1
        pimg = np.pad(img, ((pd, pd), (pd, pd)), 'reflect')
        col_img = im2col(pimg, [3, 3])
        laplac = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).ravel()
        oimg = np.sum(col_img * laplac[:, None], axis=0)

        return oimg

