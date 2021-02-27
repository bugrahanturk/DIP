import cv2
import numpy as np
import matplotlib.pyplot as plt



def bandreject_filter(shape, d0=160, w=20, n=2):
    P, Q = shape
    # Initialize filter with ones
    H = np.ones((P, Q))
    for u in range(0, P):
        for v in range(0, Q):
            # Get euclidean distance from point D(u,v) to the center
            D_uv = np.sqrt((u - (P / 2)) ** 2 + (v - (Q / 2)) ** 2)

            # Butterworth bandreject filter from Gonzales book
            if D_uv == d0:  # To avoid dividing by zero
                H[u, v] = 0
            else:
                H[u, v] = 1 / (1 + ((D_uv * w) / (D_uv ** 2 - d0 ** 2)) ** (2 * n))

    return H


img = cv2.imread('question_1.tif.tif', 0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
pow_spec = np.log(np.abs(fshift) ** 2)
phase_spectrumR = np.angle(fshift)
magnitude_spectrum = 20 * np.log(np.abs(fshift))
img_shape = img.shape

H1 = bandreject_filter(img_shape, 180, 10)
H2 = bandreject_filter(img_shape, 140, 10)

NotchFilter = H1 * H2
NotchRejectCenter = fshift * NotchFilter
NotchReject = np.fft.ifftshift(NotchRejectCenter)
inverse_NotchReject = np.fft.ifft2(NotchReject)  # Compute the inverse DFT of the result

Result = np.abs(inverse_NotchReject)

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(222)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude spectrum')

plt.subplot(223)
plt.imshow(magnitude_spectrum * NotchFilter, "gray")
plt.title("Band Reject Filter")

plt.subplot(224)
plt.imshow(Result, "gray")
plt.title("Result")

plt.show()
