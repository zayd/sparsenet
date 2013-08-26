import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
import theano.sandbox.linalg as tl


def sparsenet(dim=64, neurons=256, lambdav=0.1, eta=3.0, num_trials=1000, batch_size=100):
  """
  dim: # Inputs
  neurons: # Outputs
  lambdav: Sparsity Constraint
  eta: Learning Rate
  num_trials: Learning Iterations
  batch_size: Batch size per iteration
  """

  ## Data ##
  # Initialize basis functions
  Phi = np.random.randn(dim,neurons)
  Phi = np.dot(Phi,np.diag(1/np.sqrt(np.sum(Phi * Phi, axis = 0))))

  side = np.sqrt(dim)
  eta = eta / batch_size
 
  I = extract_patches()
  tI = theano.shared(I.astype(theano.config.floatX), name="tI")

  ## Model ##
  # Learning
  tPhi = theano.shared(Phi.astype(theano.config.floatX), name="tPhi")
  tA = T.matrix("tA")
  tR = T.matrix("tR")
  t = T.iscalar("t")

  tR = tI[:,t*batch_size:(t+1)*batch_size] - T.dot(tPhi, tA)
  dPhi = T.dot(tR, tA.T)

  learn = theano.function(inputs=[t, tA], 
    updates=[(tPhi, T.dot((tPhi + eta*dPhi),tl.diag(1/T.sqrt(T.sum((tPhi + eta*dPhi)**2, axis=0)))))],
    allow_input_downcast=True, mode='ProfileMode')

  # Inference 
  tb = T.matrix("tb")
  tG = T.matrix("tG")
  
  tl = theano.shared(np.zeros(()))
  tu = theano.shared(np.zeros((M, batch_size)))

  inference = theano.function(inputs=[tPhi, ], outputs=[tA, ],)

  # Reset Inference 
  def reset_inference():
    tb = T.dot(tPhi.T,tI[:,t*batch_size:(t+1)*batch_size])
    tG = T.dot(tPhi.T, tPhi) - T.eye(neurons) 

    l = 0.5 * 

  # Thresholding
  l0 = T.matrix("l0")
  l0[abs(u) < theta] = 0
    a = u;
    a[np.abs(a) < theta] = 0
    return a

  for t in range(num_trials+1):
    Phi = tPhi.get_value()
    # Coefficient Inference
    ahat = sparsify(I[:,t*batch_size:(t+1)*batch_size],Phi,lambdav)
    learn(t, ahat)
    # Calculate Residual Error
    #R = I-np.dot(Phi,ahat)

    # Update Basis Functions
    #dPhi = eta * (np.dot(R, ahat.T))
    #Phi = Phi + dPhi
    #Phi = np.dot(Phi,np.diag(1/np.sqrt(np.sum(Phi * Phi, axis = 0))))

    # Plot every 100 iterations
    if np.mod(t,100) == 0:
      print "Iteration " + str(t)
      square = np.sqrt(neurons)
      image = np.zeros((side*square+square,side*square+square))
      for i in range(square.astype(int)):
        for j in range(square.astype(int)):
          image[i*side+i:i*side+side+i,j*side+j:j*side+side+j] = np.reshape(Phi[:,i*square+j],(side,side))

      plt.imshow(image, cmap='jet', interpolation="nearest", vmin=np.min(image), vmax=np.max(image))
      plt.savefig('./figures/t' + str(t))

  return Phi

def sparsify(I, Phi, lambdav):
  """
  Inference step. Learns coefficients.
  I: Image batch to learn coefficients
  Phi: Dictionary
  lambdav: Sparsity coefficient
  """
  batch_size = np.shape(I)[1]
  (N, M) = np.shape(Phi)

  b = np.dot(Phi.T, I)
  G = np.dot(Phi.T, Phi) - np.eye(M)

  num_iterations = 75
  eta = 0.1

  u = np.zeros((M,batch_size))

  l = 0.5 * np.max(np.abs(b), axis=0)
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

def extract_patches(dim=64, num_trials=1000, batch_size=100, BUFF=4):
  IMAGES = scipy.io.loadmat('../images/IMAGES.mat')
  IMAGES = IMAGES['IMAGES']
  side = np.sqrt(dim)

  (imsize, imsize, num_images) = np.shape(IMAGES)
  I = np.zeros((dim,batch_size*num_trials))

  # Choose a random image
  for t in range(num_trials):
    imi = np.ceil(num_images * random.uniform(0,1))

    for b in range(batch_size):
      r = BUFF + np.ceil((imsize-side-2*BUFF) * random.uniform(0,1))
      c = BUFF + np.ceil((imsize-side-2*BUFF) * random.uniform(0,1))

      I[:,t*batch_size+b] = np.reshape(IMAGES[r:r+side, c:c+side, imi-1],dim,1)

  return I