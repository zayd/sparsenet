import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import fista

def sparsenet(patch_dim=64, neurons=128, lambdav=0.1, eta=6.0, num_trials=3000, batch_size=100, border=4, inference='lca'):
  """
  N: # Inputs
  M: # Outputs
  lambdav: Sparsity Constraint
  eta: Learning Rate
  num_trials: Learning Iterations
  batch_size: Batch size per iteration
  border: Border when extracting image patches
  Inference: 'lca' or 'fista'  
  """
  IMAGES = scipy.io.loadmat('./IMAGES.mat')
  IMAGES = IMAGES['IMAGES']

  (imsize, imsize, num_images) = np.shape(IMAGES)

  sz = np.sqrt(patch_dim)
  eta = eta / batch_size

  # Initialize basis functions
  Phi = np.random.randn(patch_dim, neurons)
  Phi = np.dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))

  I = np.zeros((patch_dim,batch_size))

  for t in range(num_trials):

    # Choose a random image
    imi = np.ceil(num_images * random.uniform(0, 1))

    for i in range(batch_size):
      r = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
      c = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))

      I[:,i] = np.reshape(IMAGES[r:r+sz, c:c+sz, imi-1], patch_dim, 1)

    # Coefficient Inference
    if inference == 'lca':
      ahat = sparsify(I, Phi, lambdav)
    elif inference == 'fista':
      ahat = fista.fista(I, Phi, lambdav, max_iterations=50)
    else:
      print "Invalid inference option"
      return

    # Calculate Residual Error
    R = I-np.dot(Phi, ahat)

    # Update Basis Functions
    dPhi = eta * (np.dot(R, ahat.T))
    Phi = Phi + dPhi
    Phi = np.dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))

    # Plot every 100 iterations
    if np.mod(t,100) == 0:
      print "Iteration " + str(t)
      side = np.sqrt(neurons)
      image = np.zeros((sz*side+side,sz*side+side))
      for i in range(side.astype(int)):
        for j in range(side.astype(int)):
          patch = np.reshape(Phi[:,i*side+j],(sz,sz))
          patch = patch/np.max(np.abs(patch))
          image[i*sz+i:i*sz+sz+i,j*sz+j:j*sz+sz+j] = patch

      plt.imshow(image, cmap=cm.Greys_r, interpolation="nearest")

  return Phi

def sparsify(I, Phi, lambdav, eta=0.1, num_iterations=125):
  """
  LCA Inference.
  I: Image batch (dim x batch)
  Phi: Dictionary (dim x dictionary element)
  lambdav: Sparsity coefficient
  eta: Update rate
  """
  batch_size = np.shape(I)[1]

  (N, M) = np.shape(Phi)
  sz = np.sqrt(N)

  b = np.dot(Phi.T, I)
  G = np.dot(Phi.T, Phi) - np.eye(M)

  u = np.zeros((M,batch_size))

  l = 0.5 * np.max(np.abs(b), axis = 0)
  a = g(u, l)

  for t in range(num_iterations):
    u = eta * (b - np.dot(G, a)) + (1 - eta) * u
    a = g(u, l)

    l = 0.95 * l
    l[l < lambdav] = lambdav

  return a

# g:  Hard threshold. L0 approximation
def g(u,theta):
  """
  LCA threshold function
  u: coefficients
  theta: threshold value
  """
  a = u;
  a[np.abs(a) < theta] = 0
  return a
