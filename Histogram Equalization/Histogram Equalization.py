import cv2
import numpy
import matplotlib.pyplot as plt

def getHistogram(image):                # Find histogram of image
    resultArray = numpy.zeros(256)      # Find pixel count
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            resultArray[image[i, j]] += 1
    return resultArray


def syncHistogram(histogramArray, imageSize, L):  # Sort pixels by their values
    newValues = numpy.zeros(len(histogramArray))  # Create CDF matrix
    newValues[0] = histogramArray[0]
    for i in range(1, len(newValues), +1):
        newValues[i] = (int)(histogramArray[i]) + (int)(newValues[i - 1])
    for i in range(newValues.shape[0]):
        newValues[i] = round(((newValues[i] - min(histogramArray)) / (imageSize - min(histogramArray))) * (2 ** L - 1))
    return newValues


def syncImageHistogram(image, newValues):
    resultImage = numpy.zeros((image.shape[0], image.shape[1], 1), dtype=numpy.uint8)
    for i in range(resultImage.shape[0]):
        for j in range(resultImage.shape[1]):
            resultImage[i, j] = newValues[image[i, j]]  # Change pixel values according
    return resultImage  # to CDF pixel values


img = cv2.imread('question_1.tif')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
histogramArray = getHistogram(gray)

newValues = syncHistogram(histogramArray, gray.shape[0] * gray.shape[1], 8)
Result = syncImageHistogram(gray, newValues)



display = plt.figure()
display.add_subplot(1,2, 1)
plt.imshow(img,"gray"), plt.title("Original")
display.add_subplot(1,2, 2)
plt.imshow(Result,"gray"), plt.title("Histogram Equalized")
plt.show(block=True)
