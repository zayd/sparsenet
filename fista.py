import numpy as np
import math
import scipy.sparse as sps
import scipy.sparse.linalg
import time

def fista(I, Phi, lambdav, max_iterations=150, display=False):
  """ FISTA Inference for Lasso (l1) Problem 
  I: Batches of images (dim x batch)
  Phi: Dictionary (dim x dictionary element) (nparray or sparse array)
  lambdav: Sparsity penalty
  max_iterations: Maximum number of iterations
  """
  def proxOp(x,t):
    """ L1 Proximal Operator """ 
    return np.fmax(x-t, 0) + np.fmin(x+t, 0)

  x = np.zeros((Phi.shape[1], I.shape[1]))
  Q = Phi.T.dot(Phi)
  c = -2*Phi.T.dot(I)

  L = scipy.sparse.linalg.eigsh(2*Q, 1, which='LM')[0]
  invL = 1/float(L)

  y = x
  t = 1

  for i in range(max_iterations):
    g = 2*Q.dot(y) + c
    x2 = proxOp(y-invL*g,invL*lambdav)
    t2 = (1+math.sqrt(1+4*(t**2)))/2.0
    y = x2 + ((t-1)/t2)*(x2-x)
    x = x2
    t = t2
    if display == True:
      print "L1 Objective " +  str(np.sum((I-Phi.dot(x2))**2) + lambdav*np.sum(np.abs(x2)))

  return x2