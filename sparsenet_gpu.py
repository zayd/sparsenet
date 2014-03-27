import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
import theano.sandbox.linalg as tla


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
  tA = theano.shared((np.zeros((neurons, batch_size)).astype(theano.config.floatX)))
  tA2 = T.matrix("tA2")
  tR = T.matrix("tR")
  t = T.iscalar("t")

  tR = tI[:,t*batch_size:(t+1)*batch_size] - T.dot(tPhi, tA)
  dPhi = T.dot(tR, tA2.T)

  learn = theano.function(inputs=[t, tA2], 
    updates=[(tPhi, T.dot((tPhi + eta*dPhi),tla.diag(1/T.sqrt(T.sum((tPhi + eta*dPhi)**2, axis=0)))))],
    allow_input_downcast=True, mode='FAST_RUN')

  # Inference 
  zeta = 0.1
  lambdav = 0.1
  #tb = T.matrix("tb")
  #tG = T.matrix("tG")
  #tb = T.dot(tPhi.T,tI[:,t*batch_size:(t+1)*batch_size])
  #tG = T.dot(tPhi.T, tPhi) - T.eye(neurons) 
  tb = theano.shared(np.zeros((neurons, batch_size)).astype(theano.config.floatX))
  tG = theano.shared(np.zeros((neurons, neurons)).astype(theano.config.floatX))

  tl = theano.shared(np.zeros((batch_size)).astype(theano.config.floatX))
  tu = theano.shared(np.zeros((neurons, batch_size)).astype(theano.config.floatX))

  inference = theano.function(inputs=[], 
    updates=[(tu, zeta*(tb-T.dot(tG,tA)) + (1-zeta)*tu), 
    (tl, (0.95*tl-0.95*tl*T.lt(0.95*tl, lambdav)+lambdav*T.lt(0.95*tl, lambdav))), 
    (tA, tu-tu*T.lt(abs(tu), lambdav))],
    allow_input_downcast=True)

  # Reset Inference 
  reset_inference = theano.function(inputs=[t], 
    updates=[(tl, 0.5*T.max(abs(T.dot(tPhi.T,tI[:,t*batch_size:(t+1)*batch_size])), axis=0)),
    (tu, T.zeros((neurons, batch_size))),
    (tb, T.dot(tPhi.T,tI[:,t*batch_size:(t+1)*batch_size])),
    (tG, T.dot(tPhi.T, tPhi) - T.eye(neurons))])

  for t in range(num_trials):
    # Coefficient Inference
    #print np.sum(tA.get_value())
    #reset_inference(t)
    #print np.sum(tA.get_value())
    #for k in range(75):
    ahat = sparsify(I[:,t*batch_size:(t+1)*batch_size], tPhi.get_value(), 0.1)
      #print np.sum(tl.get_value())
    learn(t, ahat)
    # Plot every 100 iterations
    if np.mod(t,100) == 0:
      display(t, tPhi.get_value())

  return Phi

def extract_patches(dim=64, num_trials=5000, batch_size=100, BUFF=4):
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

def display(t, Phi):
  (dim, neurons) = Phi.shape
  side = np.sqrt(dim)
  square = np.sqrt(neurons)
  image = np.zeros((side*square+square,side*square+square))
  
  print "Iteration " + str(t)
  for i in range(square.astype(int)):
    for j in range(square.astype(int)):
      image[i*side+i:i*side+side+i,j*side+j:j*side+side+j] = np.reshape(Phi[:,i*square+j],(side,side))

  plt.imshow(image, cmap='Greys', interpolation="nearest", vmin=np.min(image), vmax=np.max(image))
  plt.savefig('./figures/t' + str(t))

def sparsify(I, Phi, lambdav):
  """
  Inference step. Learns coefficients.
  I: Image batch to learn coefficients
  Phi: Dictionary
  lambdav: Sparsity coefficient
  """
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

def g(u,theta):
  """
  LCA threshold function
  u: coefficients
  theta: threshold value
  """
  a = u;
  a[np.abs(a) < theta] = 0
  return a
