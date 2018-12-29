# Convolução com filtro detector de bordas

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from skimage import io, color
from skimage import exposure

img = io.imread('lena.jpg')    # Load the image
io.imshow(img)
io.show()
print ('image matrix size: ', img.shape, img.dtype)      # print the size of image

img_gray = color.rgb2gray(img)      # Convert the image to grayscale (1 channel)
io.imshow(img_gray)
io.show()
print ('image gray matrix size: ', img_gray.shape, img_gray.dtype)      # print the size of image
print ('\n First 5 columns and rows of the image gray matrix: \n', img_gray[:5,:5]*255) 

kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]) # kernel edges

# you can use 'valid' instead of 'same', then it will not add zero padding
image_edges = scipy.signal.convolve2d(img_gray, kernel, 'same')
plt.imshow(image_edges)
plt.show()
print ('image edges matrix size: ', image_edges.shape)      # print the size of image
print ('\n First 5 columns and rows of the image_sharpen matrix: \n', image_edges[:5,:5]*255)

# Adjust the contrast of the filtered image by applying Histogram Equalization
edges_equalized = exposure.equalize_adapthist(image_edges/np.max(np.abs(image_edges)), clip_limit=0.03)
plt.imshow(edges_equalized, cmap=plt.cm.gray)    # plot the edges_clipped
plt.axis()
plt.show()