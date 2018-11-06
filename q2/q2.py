import numpy as np  
import matplotlib.pyplot as plt 
from scipy.ndimage import imread
import itertools


b = imread('unifnoisy.jpg',mode='L')
dims = np.shape(b)

########## gaussian filter ###############
def gauss(winDim,sigma):
	a = np.zeros((winDim,winDim))
	centre = int(winDim/2)
	for i,j in itertools.product(range(winDim),range(winDim)):
		a[i,j] = (i-centre)**2 + (j-centre)**2
	a = np.exp(-a/sigma)
	norm = np.sum(np.sum(a))
	return a

def ygauss(x,winDim,sigma):
	a = np.zeros((winDim,winDim))
	centre = int(winDim/2)
	for i,j in itertools.product(range(winDim),range(winDim)):
		a[i,j] = (x[i,j]-x[centre,centre])**2 
	a = np.exp(-a/sigma)
	norm = np.sum(np.sum(a))
	return a

plt.figure()
######### filter parameters
winDim = 5
xsigma = 3
ysigma = 100

'''
CHECK THE NORMALISING OF THE TWO FILTERS, DONE SEPARATELY, THAT MAY BE CAUSING THE ERROR, DO VERIFY
'''
############ zero-padding #############
for m,n in itertools.product(range(1,5),range(1,5)):
	xsigma = m*5
	ysigma = n*5
	a = b 
	exZero = int(winDim/2)
	y = np.zeros((2*exZero+dims[0],2*exZero+dims[1]))
	print np.shape(y)
	y[exZero+1:exZero+1+dims[0],exZero+1:exZero+1+dims[1]] = a

	gaussFilter = gauss(winDim,xsigma)

	for i,j in itertools.product(range(dims[0]),range(dims[1])):
		k = i+exZero
		l = j+exZero
		patch = y[k-exZero:k+exZero+1,l-exZero:l+exZero+1]
		bifilter = np.multiply(gaussFilter,ygauss(patch,winDim,ysigma))
		bifilter /= np.sum(np.sum(bifilter))
		a[i,j] = np.sum(np.sum(np.multiply(patch,bifilter))).astype(int)

	plt.subplot(4,4,(m)+(n-1)*4)
	plt.imshow(a)
plt.show()
