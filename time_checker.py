import time
import cv2 
import os
import numpy as np
import numba as nb
import pandas as pd
import scipy.fftpack as sp

def time_it(func):
    def wrapper(*args):
        start = time.time()
        res = func(*args)
        end = time.time()
        return res, end - start
    return wrapper

@time_it
def laplacian_variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()
@time_it
def gradient_magnitude(image):
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return np.mean(gradient_magnitude)
@time_it
def frequency_based(image):
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)
    return np.mean(magnitude_spectrum)
@time_it
def frequency_based_scipy(image):
    fft_image = sp.fft2(image)
    fft_shifted = sp.fftshift(fft_image)
    # Calculate magnitude spectrum and avoid log(0) issue
    magnitude_spectrum = np.log(np.abs(fft_shifted) + 1e-8)
    return np.mean(magnitude_spectrum)

@time_it
@nb.njit
def frequnecy_based_numba(image):
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)
    return np.mean(magnitude_spectrum)
@time_it
def edge_based(image):
    edges = cv2.Canny(image, 50, 150)
    return np.sum(edges) / float(image.size)


if __name__ == '__main__':
    df = pd.read_csv('processed_analyzed.csv')

    times = []
    lap = []
    grad = []
    freq = []
    edge = []
    sp_freq = []
    resize = []
    numba = []

    for i, img in enumerate(df['path']):
        # if i == 1000:
        # 	break
        bwimg = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        # print('File:', img, 'exists:', os.path.exists(img))
        start = time.time()
        bwimg = cv2.resize(bwimg, (640, 640))
        end = time.time()
        resize.append(end - start)
        # start = time.time()
        # laplacian_variance(bwimg)
        # end_lap = time.time()
        # gradient_magnitude(bwimg)
        pr_n, t_numpy = frequency_based(bwimg)
        pr_s, t_scipy = frequency_based_scipy(bwimg)
        pr_numba, t_numba = frequnecy_based_numba(bwimg)
        # edge_based(bwimg)
        # end = time.time()
        # times.append(end - start)
        # lap.append(end_lap - start)
        # grad.append(end_grad - end_lap)
        freq.append(t_numpy)
        # edge.append(end - end_freq)
        sp_freq.append(t_scipy)
        numba.append(t_numba)

    # print(f'laplacian_variance mean: {1000 * sum(lap)/len(lap)}')
    #print(f'gradient_magnitude mean: {1000 * sum(grad)/len(grad)}')
    print(f'numba frequency_based mean: {1000 * sum(numba)/len(numba)}')
    print(f'numpy frequency_based mean: {1000 * sum(freq)/len(freq)}')
    print(f'scipy frequency_based mean: {1000 * sum(sp_freq)/len(sp_freq)}')
    print(f'resize mean: {1000 * sum(resize)/len(resize)}')

#print(f'edge_based mean: {1000 * sum(edge)/len(edge)}')
#print(f"Mean time: {(sum(times)/len(times))* 1000}")