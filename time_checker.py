import time
import cv2 

import numpy as np 
import pandas as pd
import scipy
def laplacian_variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def gradient_magnitude(image):
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return np.mean(gradient_magnitude)

def frequency_based(image):
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)
    return np.mean(magnitude_spectrum)

def edge_based(image):
    edges = cv2.Canny(image, 50, 150)
    return np.sum(edges) / float(image.size)

df = pd.read_csv('analyze.csv')

times = []
lap = []
grad = []
freq = []
edge = []

for i, img in enumerate(df['path']):
	# if i == 1000:
	# 	break
	bwimg = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
	bwimg = cv2.resize(bwimg, (640, 640))

	start = time.time()
	laplacian_variance(bwimg)
	end_lap = time.time()
	gradient_magnitude(bwimg)
	end_grad = time.time()
	frequency_based(bwimg)
	end_freq = time.time()
	edge_based(bwimg)
	end = time.time()
	times.append(end - start)
	lap.append(end_lap - start)
	grad.append(end_grad - end_lap)
	freq.append(end_freq - end_grad)
	edge.append(end - end_freq)

    
print(f'laplacian_variance mean: {1000 * sum(lap)/len(lap)}')
print(f'gradient_magnitude mean: {1000 * sum(grad)/len(grad)}')
print(f'frequency_based mean: {1000 * sum(freq)/len(freq)}')
print(f'edge_based mean: {1000 * sum(edge)/len(edge)}')
print(f"Mean time: {(sum(times)/len(times))* 1000}")