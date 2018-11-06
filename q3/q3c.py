import numpy as np  
import matplotlib.pyplot as plt 
from scipy.ndimage import imread
import itertools


image = imread('lenna_noise.jpg',mode='L')
m,n = np.shape(image)

window = 5
patchSize = 7
sigma = 50.
pad = window+patchSize/2-1
ip = image
ip = np.pad(ip, ((pad,pad),(pad,pad)), 'edge')
target = np.zeros_like(image)

for i,j in itertools.product(range(pad+1,m+pad+1),range(pad+1,n+pad+1)):
	patch = ip[i-patchSize/2:i+patchSize/2+1,j-patchSize/2:j+patchSize/2+1]
	norm = 0
	summation = 0
	for k,l in itertools.product(range(-window/2,window/2),range(-window/2,window/2)):
		patchComp = ip[i+k-patchSize/2:i+k+patchSize/2+1,j+l-patchSize/2:j+l+patchSize/2+1]
		ssd = np.sum(np.sum(np.square(patch-patchComp)))
		weight = np.exp(-1*ssd/sigma)
		norm += weight
		summation += weight*ip[i+k,j+l]
	target[i-pad-1,j-pad-1] = summation/norm

plt.figure()
plt.subplot(2,1,1)
plt.imshow(target)
plt.subplot(2,1,2)
plt.imshow(image)
plt.show()