import numpy as np  
import matplotlib.pyplot as plt 
from scipy.ndimage import imread

image = imread('lenna_noise.jpg',mode='L')
m,n = np.shape(image)
original = image

####### anisotropic diffusion parameters #########
lbda = 0.2
K = 50 #need to calculate: use the Canny method ( FIND THE REASON : PERONA MALIK ME HEIN )

def diffCoeff(grad,K):
	return np.exp(-np.abs(grad)/K**2)

def calculateK(grad):
	grad = grad.reshape((1,-1))
	hist, bin_edges = np.histogram(grad,bins = int(np.max(grad)))
	total = np.sum(grad)
	summation = 0
	i = -1
	while (summation<=0.9*total):
		i+=1
		summation+=hist[i]*(i+1)
	return i

iterations = 50
images = np.ndarray((iterations,m,n))

for i in range(iterations):
	### in each iteration ###
	padded = np.pad(image,((1,1),(1,1)),'edge')
	Ngrad = padded[1:-1,:-2] - image
	Sgrad = padded[1:-1,2:] - image
	Egrad = padded[:-2,1:-1] - image
	Wgrad = padded[2:,1:-1] - image

	K = calculateK(np.abs(Ngrad+Sgrad+Egrad+Wgrad))
	image = image + lbda*(np.multiply(Ngrad,diffCoeff(Ngrad,K)) + 
		np.multiply(Sgrad,diffCoeff(Sgrad,K)) +
		np.multiply(Egrad,diffCoeff(Egrad,K)) +
		np.multiply(Wgrad,diffCoeff(Wgrad,K)))

	images[i,:,:] = image
	###### recalculate K for the next iteration

####### calculating stopping iteration ########
images = images.reshape((iterations,1,-1))
original = original.reshape((1,-1))
corrs = images - original
corrs = [np.sum(np.corrcoef(np.vstack((original,corrs[i,:])))) for i in range(iterations)]
optIter = np.argmin(corrs)

filteredImage = images[optIter,:].reshape((m,n))

print corrs, optIter

plt.figure()
plt.subplot(2,1,1)
plt.imshow(original.reshape((m,n)))
plt.subplot(2,1,2)
plt.imshow(filteredImage)
plt.show()






