import numpy as np
import cv2
import matplotlib.pyplot as plt



def contraharmonic_filter(img, q):
    global contraharmonic
    target = np.zeros(img.shape, np.uint8)
    r = 1
    row, col = img.shape
    for i in range(r, row - r):
        for j in range(r, col - r):
            # Using formula from Gonzales book
            g = img[i - r:i + r + 1, j - r:j + r + 1]
            nominator = np.sum(np.power(g, q + 1))
            denominator = np.sum(np.power(g, q))
            if denominator != 0:
                contraharmonic = nominator / denominator
            if denominator == 0:
                target[i][j] = target[i][j]
            else:
                target[i][j] = int(contraharmonic)
    return target


img = cv2.imread("question_2.tif.tif")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Q = 1.5 for eliminating pepper noise
Result = contraharmonic_filter(gray, 1.5)

display = plt.figure()
display.add_subplot(1, 2, 1)
plt.imshow(img, "gray"), plt.title("Original")
display.add_subplot(1, 2, 2)
plt.imshow(Result, "gray"), plt.title("Contraharmonic Mean Filter")
plt.show(block=True)
