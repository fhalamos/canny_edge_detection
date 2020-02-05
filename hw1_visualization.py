## Homework 1
##
## For this assignment, you will implement basic convolution, denoising, and
## edge detection operations.  Your implementation should be restricted to
## using low-level primitives in numpy (e.g. you may not call a Python library
## routine for convolution in the implementation of your convolution function).
##
## This notebook provides examples for testing your code.
## See hw1.py for detailed descriptions of the functions you must implement.



import numpy as np
import matplotlib.pyplot as plt


from util import *
from hw1 import *

# Problem 1 - Convolution (10 Points)
#
# Implement the conv_2d() function as described in hw1.py.
#
# The example below tests your implementation by convolving with a box filter.

image = load_image('data/69015.jpg')
box = np.array([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]])
img = conv_2d(image,box)

plt.figure(); plt.imshow(image, cmap='gray')

plt.savefig("p1a.png")

plt.figure(); plt.imshow(img, cmap='gray')

plt.savefig("p1b.png")

# plt.show()


# Problem 2
# (a) Denoising with Gaussian filtering (5 Points)
#
# Implement denoise_gaussian() as described in hw1.py.
#
# The example below tests your implementation.

image = load_image('data/148089_noisy.png')
imgA  = denoise_gaussian(image, 1.0)
imgB  = denoise_gaussian(image, 2.5)

plt.figure(); plt.imshow(image, cmap='gray')
plt.savefig("p2a-a.png")
plt.figure(); plt.imshow(imgA, cmap='gray')
plt.savefig("p2a-b.png")
plt.figure(); plt.imshow(imgB, cmap='gray')
plt.savefig("p2a-c.png")

# plt.show()

# (b) Denoising with median filtering (5 Points)
#
# Implement denoise_median() as described in hw1.py.
#
# The example below tests your implementation.

image = load_image('data/143090_noisy.png')
imgA  = denoise_median(image, 1)
imgB  = denoise_median(image, 2)

plt.figure(); plt.imshow(image, cmap='gray')
plt.savefig("p2b-a.png")
plt.figure(); plt.imshow(imgA, cmap='gray')
plt.savefig("p2b-b.png")
plt.figure(); plt.imshow(imgB, cmap='gray')
plt.savefig("p2b-c.png")

image = load_image('data/143090.jpg')


# Problem 3 - Sobel gradient operator (5 Points)
#
# Implement sobel_gradients() as described in hw1.py.
#
# The example below tests your implementation.

image  = load_image('data/69015.jpg')
dx, dy = sobel_gradients(image)

plt.figure(); plt.imshow(image, cmap='gray')
plt.savefig("p3a.png")
plt.figure(); plt.imshow(dx, cmap='gray')
plt.savefig("p3b.png")
plt.figure(); plt.imshow(dy, cmap='gray')
plt.savefig("p3c.png")
# plt.show()

# Problem 4 -  (a) Nonmax suppression (10 Points)
#               (b) Edge linking and hysteresis thresholding (10 Points)
#               (c) Canny edge detection (5 Points)
#
# Implement nonmax_suppress(), hysteresis_edge_linking(), canny() as described in hw1.py
#
# The examples below test your implementation

image  = load_image('data/edge_img/easy/002.jpg')
mag, nonmax, edge = canny(image)
plt.figure(); plt.imshow(mag, cmap='gray')
plt.savefig("p4a.png")
plt.figure(); plt.imshow(nonmax, cmap='gray')
plt.savefig("p4b.png")
plt.figure(); plt.imshow(edge, cmap='gray')
plt.savefig("p4c.png")
# plt.figure(); plt.imshow(strong_and_weak_edge, cmap='gray')
# plt.savefig("p4d.png")
# plt.show()

# Extra Credits:
# (a) Improve Edge detection image quality (5 Points)
# (b) Bilateral filtering (5 Points)
# You can do either one and the maximum extra credits you can get is 5.
