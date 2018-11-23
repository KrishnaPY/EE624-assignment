import cv2
import numpy as np
from matplotlib import pyplot as plt
 
LEVEL = 256         # Number of reconstruction level in image

"""
Input : Path of the image as a string
Ouput : Histogram of image(as a 2D array with LEVEL number of rows and 1 column). 
        The value of histogram are scaled to fit into range [0, 1]
"""
def histogramCalculation (img):
    N = 0
    M = 0
    NUM_PIXELS = 0

    Hst = np.zeros((LEVEL, 1))

    inputImg = cv2.imread (img, 0)

    (N, M) = inputImg.shape
    NUM_PIXELS = N*M

    Hst = cv2.calcHist ([inputImg], [0], None, [256], [0, 256])

    for i in range(0, LEVEL):
        Hst[i] = (Hst[i]/NUM_PIXELS)
    
    return Hst

# **************************** MAIN CODE ****************************

givenImage = 'givenhist.jpg'
spImage = 'sphist.jpg'
ansImage = 'tranSphist.jpg'

inputEqu = np.zeros((LEVEL, 1))
outputEqu = np.zeros((LEVEL, 1))
predictedHst = np.zeros((LEVEL, 1))
umap = np.zeros((LEVEL, 1))

inputHst = histogramCalculation (givenImage)
outputHst = histogramCalculation (spImage)

temp = 0
for i in range(0, LEVEL):
    temp += inputHst[i]
    inputEqu[i] = ((LEVEL - 1)*temp)

temp = 0
for i in range(0, LEVEL):
    temp += outputHst[i]
    outputEqu[i] = ((LEVEL - 1)*temp)

for i in range (0, LEVEL):
    si = inputEqu[i]
    minDiff = 300
    minz = -1
    for j in range (0, LEVEL):

        if abs (si - outputEqu[j]) < minDiff:
            minDiff = abs (si - outputEqu[j])
            minz = j
    
    if minz != -1:
        predictedHst[minz][0] += inputHst[i]
        umap[i][0] = minz

"""
Output image genaration
"""
N = 0;
M = 0;

inputImg = cv2.imread (givenImage, 0);  
(N, M) = inputImg.shape

outputImg = np.zeros((N, M))

for i in range(0, N):
    for j in range(0, M):
        pixel = inputImg[i][j]
        outputImg[i][j] = umap[pixel][0]

plt.gray()

plt.imsave (ansImage, outputImg)
plt.imshow (outputImg)
plt.show()

# ***************************** PLOTING HISTOGRAMS *****************************

inputHst = inputHst.ravel ()
outputHst = outputHst.ravel ()
predictedHst = predictedHst.ravel ()

width = 1/1.5

# Input histogram Graph

"""
plt.bar (range (0, LEVEL), inputHst, width, color='blue', label='Input')
plt.xlabel ('Levels')
plt.ylabel ('Frequency')
plt.title ('Histrogram')
plt.legend (loc='upper right')
plt.show()
"""

# Output histogram Graph

"""
plt.bar (range (0, LEVEL), outputHst, width, color='blue', label='Output')
plt.xlabel ('Levels')
plt.ylabel ('Frequency')
plt.title ('Histrogram')
plt.legend (loc='upper right')
plt.show()
"""

# Transformed histogram Graph

"""

plt.bar (range (0, LEVEL), predictedHst, width, color='blue', label='Transformed')
plt.xlabel ('Levels')
plt.ylabel ('Frequency')
plt.title ('Histrogram')
plt.legend (loc='upper right')
plt.show()
"""
# Bar Graph with Sub-Plot as Output histogram and Transformed histogram

"""
plt.figure('Histogram Specification')

plt.subplot(211)
plt.bar (range (0, LEVEL), outputHst, width, color='blue', label='Original')
plt.ylabel ('Frequency')
plt.title ('Histrogram')
plt.legend (loc='upper right')

plt.subplot(212)
plt.bar (range (0, LEVEL), predictedHst, width, color='orange', label='Transformed')
plt.xlabel ('Levels')
plt.ylabel ('Frequency')
plt.legend (loc='upper right')
plt.show()
"""

# Bar Graph with both the plots in one Plot (i.e. both Output and transformed)

"""
plt.figure('Histogram Comparision')

plt.bar (range (0, LEVEL), outputHst, width, color='blue', label='Original')
plt.bar (range (0, LEVEL), predictedHst, width, color='orange', label='Transformed')
plt.xlabel ('Levels')
plt.ylabel ('Frequency')
plt.title ('Histrogram')
plt.legend (loc='upper right')
plt.show()
"""

# Line graph comapring Output and transformed histogram

"""
plt.figure('Histogram Comparision')

plt.plot ( range (0, 256), predictedHst, color='blue', label='Transformed')
plt.plot ( range (0, 256), outputHst, color='orange', label='Original')
plt.xlabel ('Levels')
plt.ylabel ('Frequency')
plt.title ('Histrogram')
plt.legend (loc='upper right')
plt.show ()
"""