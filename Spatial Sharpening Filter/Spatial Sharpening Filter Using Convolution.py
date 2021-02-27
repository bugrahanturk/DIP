from skimage.exposure import rescale_intensity
import numpy as np
import cv2
import matplotlib.pyplot as plt


def convolve(source_image, kernel):
    # Get the spatial resolution of the source image and matrix kernel
    (iH, iW) = source_image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # We make room for the output picture and pad the edges
    # output has the same resolution as our picture but each pixel is 0 grayscale
    pad = (kW - 1) // 2
    source_image = cv2.copyMakeBorder(source_image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype=float)

    # multiply the matrix by each pixel and add
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # we select the 3x3 area / pixels we multiply matrix in the picture
            # Added +1 for prevent get zero
            pixels = source_image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # we multiply the pixels we select on the kernel matrix watch
            k = (pixels * kernel).sum()

            # We write the value we found instead of the output picture with the same coordinate.
            output[y - pad, x - pad] = k

    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output


sharpen = np.array((
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]))

img = cv2.imread("question_2.tif")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Applying spatial sharpening filter")
Result = convolve(gray, sharpen)

display = plt.figure()
display.add_subplot(1,2, 1)
plt.imshow(img,"gray"), plt.title("Original")
display.add_subplot(1,2, 2)
plt.imshow(Result,"gray"), plt.title("Spatial Sharpening Filter")
plt.show(block=True)


