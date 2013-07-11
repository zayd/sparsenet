import scipy as sp
import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import math

def sparsenet(N=64, M=256, lambdav=0.1, eta=3.0, num_trials=1000, batch_size=100, BUFF=4):
  """
  N: # Inputs
  M: # Outputs
  lambdav: Sparsity Constraint
  eta: Learning Rate
  num_trials: Learning Iterations
  batch_size: Batch size per iteration
  BUFF: Border when extracting image patches
  """
	IMAGES = scipy.io.loadmat('./IMAGES.mat')
	IMAGES = IMAGES['IMAGES']

	(imsize, imsize, num_images) = np.shape(IMAGES)

	sz = np.sqrt(N)
	eta = eta / batch_size

	# initialize basis functions
	Phi = np.random.randn(N,M)
	Phi = np.dot(Phi,np.diag(1/np.sqrt(np.sum(Phi * Phi, axis = 0))))

	I = np.zeros((N,batch_size))

	for t in range(num_trials):

		# choose a random image
		imi = np.ceil(num_images * random.uniform(0,1))

		for i in range(batch_size):
			r = BUFF + np.ceil((imsize-sz-2*BUFF) * random.uniform(0,1))
			c = BUFF + np.ceil((imsize-sz-2*BUFF) * random.uniform(0,1))

			I[:,i] = np.reshape(IMAGES[r:r+sz, c:c+sz, imi-1],N,1)

		# Coefficient Inference
		ahat = sparsify(I,Phi,lambdav)

		# Calculate Residual Error
		R = I-np.dot(Phi,ahat)

		# Update Basis Functions
		dPhi = eta * (np.dot(R, ahat.T))
		Phi = Phi + dPhi
		Phi = np.dot(Phi,np.diag(1/np.sqrt(np.sum(Phi * Phi, axis = 0))))

		if np.mod(t,100) == 0: 
			print "Iteration " + str(t)
			image = np.zeros((sz*np.sqrt(M)+np.sqrt(M),sz*np.sqrt(M)+np.sqrt(M)))
			for i in range(np.sqrt(M).astype(int)):
				for j in range(np.sqrt(M).astype(int)):
					image[i*sz+i:i*sz+sz+i,j*sz+j:j*sz+sz+j] = np.reshape(Phi[:,i*np.sqrt(M)+j],(sz,sz))

			plt.imshow(image, cmap=cm.jet, interpolation="nearest")
			plt.show()
			plt.draw()
		
	return Phi

def sparsify(I, Phi, lambdav):
	batch_size = np.shape(I)[1]

	(N, M) = np.shape(Phi)
	sz = np.sqrt(N)

	b = np.dot(Phi.T, I)
	G = np.dot(Phi.T, Phi) - np.eye(M)

	num_iterations = 75
	eta = 0.1

	u = np.zeros((M,batch_size))

	l = 0.5 * np.max(np.abs(b), axis = 0)
	a = g(u, l)

	for t in range(num_iterations):
		u = eta * (b - np.dot(G, a)) + (1 - eta) * u
		a = g(u, l)

		l = 0.95 * l
		l[l < lambdav] = lambdav

	return a

# g - hard threshold
def g(u,theta):
 	a = u; 
	a[np.abs(a) < theta] = 0
	return a
