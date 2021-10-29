import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import math

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    ssim = np.mean(np.mean(ssim_map))
#    ssim = ssim_map.sum() / np.count_nonzero(im1)
    return ssim

def compute_psnr_mean(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def compute_psnr_roi(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
#    mse = np.mean((img1 - img2) ** 2)
    mse = np.sum((img1 - img2) ** 2) /np.count_nonzero(img1)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def show():
    from tool.config_utils import process_config
    from data_process import DataReader

    try:
        config = process_config('/Users/admin/github/rabbitkeras/segmention_config.json')
    #    print(config)
    except Exception as e:
        print('[Exception] Config Error, %s' % e)
        exit(0)
    dr = DataReader(config)
    dr.init()
    no_list, lower_list, full_list = dr.get_show_data()
    opsnr = list()
    ossim = list()
    for i in range(len(no_list)):
        no_img = no_list[i]
        lower_img = lower_list[i]
        full_img = full_list[i]
        no = np.asarray(no_img)
        low = np.asarray(lower_img)
        full = np.asarray(full_img)
        print("{}: {}: {}".format(no.mean(), low.mean(), full.mean()))
        print("{}:psnr:{}, ssim:{}".format(i, compute_psnr_mean(no_img, full_img), compute_ssim(no_img, full_img)))
        opsnr.append(compute_psnr_mean(lower_img, full_img))
        ossim.append(compute_ssim(lower_img, full_img))

    ndopsnr = np.asarray(opsnr)
    ndossim = np.asarray(ossim)
    print(ndopsnr.mean(), ndopsnr.std())
    print(ndossim.mean(), ndossim.std())

if __name__ == "__main__":
#    im1 = Image.open("/Users/admin/github/rabbitkeras/datasets/regtest/10/full-contrast.png")
#    im2 = Image.open("/Users/admin/github/rabbitkeras/datasets/regtest/10/lower-contrast.png")
#
#    print(compute_ssim(np.array(im2),np.array(im1)))
#    print(compute_psnr_roi(np.array(im2),np.array(im1)))
    show()
