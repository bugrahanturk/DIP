
import cv2
import numpy as np
import matplotlib.pyplot as plt

def lowpass_filter(shape, d0=30):

    P, Q = shape
    # Initialize filter with zeros
    H = np.zeros((P, Q))

    # Traverse through filter
    for u in range(0, P):
        for v in range(0, Q):
            # Get euclidean distance from point D(u,v) to the center
            D_uv = np.sqrt((u - P/2)**2 + (v - Q/2)**2)
            if D_uv <= d0:
                H[u, v] = 1.0
    return H


img = cv2.imread('question_3.tif',0)            #0 for grayscale

img_shape = img.shape

f = np.fft.fft2(img)                             #compute the 2-dimesional FastFourierTransform ->ff2 Compute F(u,v) the DFT of the image
fshift = np.fft.fftshift(f)                      #Shift the zero-frequency component to the center of the spectrum.

LowPassCenter = fshift * lowpass_filter(img_shape, 30)  #Multiply F(u,v) by a filter function H(u,v)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)                 #Compute the inverse DFT of the result


Result = np.abs(inverse_LowPass)

display = plt.figure()
display.add_subplot(1,2, 1)
plt.imshow(img,"gray"), plt.title("Original")
display.add_subplot(1,2, 2)
plt.imshow(Result,"gray"), plt.title("Ideal Low Pass with r = 30")
plt.show(block=True)



