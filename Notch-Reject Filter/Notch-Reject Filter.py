import cv2
import numpy as np
import matplotlib.pyplot as plt



def notch_reject_filter(shape, d0=9, u_k=0, v_k=0): #IDEAL NOTCH REJECT
    P, Q = shape
    # Initialize filter with zeros
    H = np.zeros((P, Q))

    # Traverse through filter
    for u in range(0, P):
        for v in range(0, Q):
            # Get euclidean distance from point D(u,v) to the center
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)

            if D_uv <= d0 or D_muv <= d0:
                H[u, v] = 0.0
            else:
                H[u, v] = 1.0

    return H


img = cv2.imread('question_1.tif.tif', 0)

img_shape = img.shape

H1 = notch_reject_filter(img_shape, 4, 38, 30)
H2 = notch_reject_filter(img_shape, 4, -42, 27)
H3 = notch_reject_filter(img_shape, 2, 80, 30)
H4 = notch_reject_filter(img_shape, 2, -82, 28)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

NotchFilter = H1 * H2 * H3 * H4
NotchRejectCenter = fshift * NotchFilter
NotchReject = np.fft.ifftshift(NotchRejectCenter)
inverse_NotchReject = np.fft.ifft2(NotchReject)  # Compute the inverse DFT of the result

Result = np.abs(inverse_NotchReject)


display = plt.figure()
display.add_subplot(1,2, 1)
plt.imshow(img,"gray"), plt.title("Original")
display.add_subplot(1,2, 2)
plt.imshow(Result,"gray"), plt.title("Notch Reject Filter")
plt.show(block=True)