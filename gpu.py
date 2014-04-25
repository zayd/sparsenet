from __future__ import division

from numbapro import cuda
import numpy as np
import numbapro.cudalib.cublas
import numpy.random
import math
import scipy.sparse.linalg
import scipy.sparse as sps

def fista(I, Phi, lambdav, L=None, tol=10e-6, max_iterations=200, display=True, verbose=False):
	"""
	I: Images 
 	Phi: Dictionary 
 	lambdav: Sparse Penalty 
 	L = Largest eigenvalue of Phi 
 	"""

	b = numbapro.cudalib.cublas.Blas()
	(m, n) = Phi.shape
	(m, batch) = I.shape

	if L == None:
		L = scipy.sparse.linalg.svds(Phi, 1, which='LM', return_singular_vectors=False)
		print "Max eigenvalue: ." + str(L)

	L = (L**2)*2 # L = svd(Phi) -> eig(2*(Phi.T*Phi))
	invL = 1/L
	t = 1.

	if sps.issparse(Phi):
		Phi = np.array(Phi.todense())

	d_I = cuda.to_device(np.array(I, dtype=np.float32, order='F'))
	d_Phi = cuda.to_device(np.array(Phi, dtype=np.float32, order='F'))
	d_Q = cuda.device_array((n, n), dtype=np.float32, order='F')
	d_c = cuda.device_array((n, batch), dtype=np.float32, order='F')
	d_x = cuda.to_device(np.array(np.zeros((n, batch), dtype=np.float32), order='F'))
	d_y = cuda.to_device(np.array(np.zeros((n, batch), dtype=np.float32), order='F'))
	d_x2 = cuda.to_device(np.array(np.zeros((n, batch), dtype=np.float32), order='F'))

	# Temporary array variables
	d_t = cuda.device_array((m, batch), dtype=np.float32, order='F')
	d_t2 = cuda.device_array(n*batch, dtype=np.float32, order='F')

	b.gemm('T', 'N', n, n, m, 1, d_Phi, d_Phi, 0, d_Q) 	# Q = Phi^T * Phi
	b.gemm('T', 'N', n, batch, m, -2, d_Phi, d_I, 0, d_c) # c = -2*Phi^T * y

	blockdim = 32, 32
	griddim = int(math.ceil(n/blockdim[0])), int(math.ceil(batch/blockdim[1]))

	blockdim_1d = 256
	griddim_1d = int(math.ceil(n*batch/blockdim_1d))

	start = l2l1obj(b, d_I, d_Phi, d_x, d_t, d_t2, lambdav, blockdim_1d, griddim_1d)
	obj2 = start

	for i in xrange(max_iterations):

		# x2 = 2*Q*y + c
		b.symm('L', 'U', n, batch, 2, d_Q, d_y, 0, d_x2)
		b.geam('N', 'N', n, batch, 1, d_c, 1, d_x2, d_x2)
		
		# x2 = y - invL * x2
		b.geam('N', 'N', n, batch, 1, d_y, -invL, d_x2, d_x2)

		# proxOp()						
		l1prox[griddim, blockdim](d_x2, invL*lambdav, d_x2)
		t2 = (1+math.sqrt(1+4*(t**2)))/2.0
		
		# y = x2 + ((t-1)/t2)*(x2-x)
		b.geam('N', 'N', n, batch, 1+(t-1)/t2, d_x2, (1-t)/t2, d_x, d_y)

		# x = x2
		b.geam('N', 'N', n, batch, 1, d_x2, 0, d_x, d_x)
		t = t2

		# update objective
		obj = obj2
		obj2 = l2l1obj(b, d_I, d_Phi, d_x2, d_t, d_t2, lambdav, blockdim_1d, griddim_1d)

		if verbose:
			x2 = d_x2.copy_to_host()
			print "L1 Objective: " + str(obj2)

		if np.abs(obj-obj2)/float(obj) < tol:
			break

	x2 = d_x2.copy_to_host()

	if display:
		print "FISTA Iterations: " + str(i)
		print "L1 Objective: " +  str(lambdav*np.sum(np.abs(x2)) + np.sum((I-Phi.dot(x2))**2))
		print "Objective delta: " + str(obj2-start)

	return x2

def l2l1obj(b, d_I, d_Phi, d_x2, d_t, d_t2, lambdav, blockdim, griddim):
	(m, n) = d_Phi.shape
	(m, batch) = d_I.shape

	b.gemm('N', 'N', m, batch, n, 1, d_Phi, d_x2, 0, d_t)
	b.geam('N', 'N', m, batch, 1, d_I, -1, d_t, d_t)

 	l2 = b.nrm2(d_t.ravel(order='F'))**2
 	
 	gabs[griddim, blockdim](d_x2.ravel(order='F'), d_t2)
 	
 	l1 = lambdav*b.asum(d_t2)

 	return l2 + l1

@cuda.jit('void(float32[:,:], float64, float32[:,:])')
def l1prox(A, t, C):
	""" l1 Proximal operator: C = np.fmax(A-t, 0) + np.fmin(A+t, 0) 
	A: coefficients matrix (dim, batch)
	t: threshold
	C: output (dim, batch) """
	i, j = cuda.grid(2)

	if i >= A.shape[0] or j >= A.shape[1]:
		return

	if A[i, j] >= t:
		C[i, j] = A[i, j] - t 
	elif A[i, j] <= -t: 
		C[i, j] = A[i, j] + t    
	else:
		C[i, j] = 0

	return

@cuda.jit('void(float32[:], float32[:])')
def gabs(x, y):
	i = cuda.grid(1)

	if i >= x.size:
		return

	if x[i] < 0:
		y[i] = -x[i]
	else:
		y[i] = x[i]

	return