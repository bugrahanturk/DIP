import cv2
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def butterworthHP(D0, imgShape, n):  # Formule
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 - 1 / (1 + (distance((y, x), center) / D0) ** (2 * n))
    return base


img = cv2.imread('question_4.tif', 0)  # 0 for grayscale

img_shape = img.shape

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

HighPassCenter = fshift * butterworthHP(60, img_shape, 2)  # Multiply F(u,v) by a filter function H(u,v)
HighPass = np.fft.ifftshift(HighPassCenter)
inverse_HighPass = np.fft.ifft2(HighPass)  # Compute the inverse DFT of the result

Result = np.abs(inverse_HighPass)

display = plt.figure()
display.add_subplot(1,2, 1)
plt.imshow(img,"gray"), plt.title("Original")
display.add_subplot(1,2, 2)
plt.imshow(Result,"gray"), plt.title("Butterworth High Pass (n=2)")
plt.show(block=True)