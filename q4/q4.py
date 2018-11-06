import numpy as np  
import matplotlib.pyplot as plt 
from scipy.ndimage import imread
import itertools

image = imread('test.jpg',mode='L')
m,n = np.shape(image)


def gradFilter(a,dir):
	m,n = np.shape(a)
	sobel = np.asarray([[-1,-2,-1],[0,0,0],[1,2,1]])
	if(dir=='x'):
		sobel = sobel.T
	target = np.zeros_like(a)
	for i,j in itertools.product(range(1,m-1),range(1,n-1)):
		target[i,j] = np.sum(np.sum(np.multiply(sobel,a[i-1:i+2,j-1:j+2])))
	return target

Ix = gradFilter(image,'x')
Iy = gradFilter(image,'y')

Ix2 = np.multiply(Ix,Ix)
Iy2 = np.multiply(Iy,Iy)
Ixy = np.multiply(Ix,Iy)

###### corner response #######
window = 5
k = 0.06 

Ix2 = np.pad(Ix2, ((window/2,window/2), (window/2, window/2)), 'edge')
Iy2 = np.pad(Iy2, ((window/2,window/2), (window/2, window/2)), 'edge')
Ixy = np.pad(Ixy, ((window/2,window/2), (window/2, window/2)), 'edge')

Sx2 = np.zeros_like(image)
Sy2 = np.zeros_like(image)
Sxy = np.zeros_like(image)

for i,j in itertools.product(range(window,m-window),range(window,n-window)):
	Sx2[i,j] = np.sum(np.sum(Ix2[i-window/2:i+window/2,j-window/2:j+window/2]))
	Sy2[i,j] = np.sum(np.sum(Iy2[i-window/2:i+window/2,j-window/2:j+window/2]))
	Sxy[i,j] = np.sum(np.sum(Ixy[i-window/2:i+window/2,j-window/2:j+window/2]))

C = np.ndarray((m,n,4))
corner = np.zeros_like(image,dtype=float)

C[:,:,0] = Sx2
C[:,:,1] = Sxy
C[:,:,2] = Sxy
C[:,:,3] = Sy2

for i,j in itertools.product(range(m),range(n)):
	matrix = C[i,j,:].reshape((2,2))
	corner[i,j] = np.linalg.det(matrix) - k*np.trace(matrix)**2


######### TBD: SETTING THE THRESHOLD (NEED TO DO A GRID SEARCH) AND NON MAXIMAL SUPPRESSION ############
plt.figure()
reserve = corner
l=1
print np.min(corner)
print np.max(corner)
for threshold in [40000,41500,43000,45000]:
	corner = reserve
	corner[corner<threshold] = 0
	target = np.zeros_like(corner)
	corner = np.pad(corner, ((1,1), (1, 1)), 'constant',constant_values=(0,0))

	######## non maximal suppression ############	
	for i,j in itertools.product(range(1,m-1),range(1,n-1)):
		if np.max(corner[i-1:i+2,j-1:j+2]) == corner[i,j]:
			target[i,j] = corner[i,j]

	x,y = np.where(target>0)
	plt.subplot(1,4,l)
	plt.imshow(image)
	y = n-y
	plt.scatter(x,y,s=1,c='r')
	l+=1
plt.show()





