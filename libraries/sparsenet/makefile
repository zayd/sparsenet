MEX = /Applications/MATLAB_R2012a.app/bin/mex 
NRFDIR = ./nrf
MEXT = mexsg

cgf.$(MEXT): cgf.c
	$(MEX) -I$(NRFDIR) -L$(NRFDIR) -lnrfopt cgf.c
