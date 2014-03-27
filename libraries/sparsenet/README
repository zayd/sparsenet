
Instructions for running sparsenet (for Matlab version 5.0 or higher)
---------------------------------------------------------------------

To run sparsenet, you will first need to compile the conjugate
gradient (cgf) routine so that it may be called through matlab.  To do
this, you will first need to build the library libnrfopt.a, which
contains the actual conjugate gradient subroutines.  Go to the nrf/
subdirectory and type

    make libnrfopt.a

Then go back up the to sparsenet directory and compile the cgf routine
so that it may be called by matlab.  First modify the makefile
definition for MEXT to be the proper suffix for the mex file for your
machine (for SGI's it is mexsg).  Then type

    make

If successful, this should create a file named cgf.xxx, where xxx is a
suffix that depends on what machine you are on.

At this point, you are in business.  Now startup Matlab.

First load the training data.  You can get the array IMAGES from the 
sparsenet web page at 

    http://redwood.ucdavis.edu/bruno/sparsenet.html

Once you download this array, you load into Matlab by typing

    load IMAGES

This will bring in a matrix of 10 images, each 512x512 (it is a 
512^2 x 10 array).  This consumes about 20 Meg, so hopefully you 
have enough memory.  To make your own training dataset, see the
instructions in the file "make-your-own-images."

The next step is to define a matrix of basis functions, A.  For
example, to learn 64 bases on 8x8 patches, define A as follows:

>> A = rand(64)-0.5;
>> A = A*diag(1./sqrt(sum(A.*A)));

This will create a 64x64 matrix initialized to random values.  Each
column is a different basis function that has unit length.

Set the colormap of Figure 1 to greyscale:

>> figure(1), colormap(gray)

Now simply run the simulation by typing

>> sparsenet


Once the simulation starts running, it will display the bases in
Figure No. 1 every 10 batches, and it will show the variance of the
coefficients in Figure No. 2.  You can stop the simulation at any
point by typing control-c.  You an change the parameters in
sparsenet.m and then resume execution by typing "sparsenet" again.

The learning rate (eta) is initialized to 5.0, which is a large value.
This quickly gets the solution in the right ballpark, but it is too
large to come to a clean stable solution.  Once it looks like you have
something interesting emerging, start reducing eta, eventually to
about 1.0.  A full set of 8x8 bases takes about 15 min. to learn
(depending on how fast your workstation is).


Notation:
---------

A          Basis functions (Phi in Nature/Vision Research. paper)
X          Input image (I in Nature/Vision Research paper)
S          Coefficients (a in Nature/Vision Research paper)
noise_var  noise variance (sigma_N^2 in Vision Research paper, eq. 6)
beta       steepness of prior (beta in Vision Research paper, eq. 8)
sigma      scaling parameter for prior (sigma in Nature paper)
eta        learning rate (eta in Nature/Vision Research paper)
tol        tolerance for conjugate gradient routine

VAR_GOAL   variance goal for the coefficients 
S_var      actual variance of the coefficients
var_eta    average rate for S_var
alpha      gain adaptation rate
gain       L2 norm of basis functions


Note that in both the Vision Research and Nature papers, sigma_N^2
(noise_var) and beta are combined into a single constant, lambda
(eq. 14 VR, eq. 2 Nature).  The scale parameter for the coefficients
does not appear in the Vision Research paper, but it does appear in
the Nature paper (eqs. 4,5).


References:

Olshausen BA, Field DJ (1997). Sparse coding with an overcomplete
basis set: A strategy employed by V1?  Vision Research, 37, 3311-3325.
ftp://redwood.ucdavis.edu/pub/papers/VR.pdf

Olshausen BA, Field DJ (1996). Emergence of simple-cell receptive
field properties by learning a sparse code for natural images.
Nature, 381, 607-609.
ftp://redwood.ucdavis.edu/pub/papers/nature-paper.pdf

