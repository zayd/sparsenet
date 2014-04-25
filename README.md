Python implementation of sparse dictionary learning. Trains on Van Hateran's or David Field's natural image datasets. Based upon Sparsenet algorithm (1996 Olshausen & Field). Standard stochastic gradient descent to learn dictionary and multiple algorithms to infer coefficients. Minimizes the following objective function:

min_{\Phi, s} || I - \Phi*s ||^2 + || s ||_1 

<b>I:</b> Data
<b>Phi:</b> Learned dictionary
<b>s:</b> Coefficients  

<table>
	<tr> <td> L1 optimization </td> <td> LCA: Locally Competitive Algorithm (2006 Rozell et al.) </td> </tr>
 	<tr>  <td> </td> <td> FISTA: Fast Iterative Shrinkage and Thresholding Algorithm (2009 Nesterov et al.) </td> </tr>
	<tr> <td> L0 optimization </td> <td> L1 Initialization and IHT: Iterative Hard Thresholding (2008 Blumensath & Davies) </td> </tr>
</table>

![2x Overcomplete Dictionary](./sparse.png)
2x Overcomplete Dictionary Trained on Natural Image Patches

