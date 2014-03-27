/*
 * diff_net.c:	computes sparse basis function decomposition
 *
 *		uses variance normalization and conjugate gradient descent
 *
 * written by:	Bruno Olshausen, June 1995
 *		Copyright, Cornell University
 */

#include <stdio.h>
#include <fcntl.h>
#include <math.h>
#include "net.h"

/*
 * Goal for variance normalization
 */
#define	VAR_GOAL 0.1
#define SIGMA 0.31622777

extern double	drand48(),atof();

int		num_inputs,	/* dimensions of input and output */
		num_outputs,
		size;
double		*I,		/* input array */
		*I_rec,		/* reconstruction array */
		*diff,		/* residual array */
		*a,		/* output array (a_i) */
		*b,		/* b_i */
		**W,		/* weights (phi_i) */
		**C,		/* weights overlaps (C_ij) */
		*gain,		/* gain of phi_i */
		*a_mean,	/* running average of a_i */
		*a_var,		/* running variance of a_i */
		*a_kurt,	/* running kurtosis of a_i */
		**W_stat;	/* averages used for computing weight update */
double		mse,		/* mean square error of input */
		mse_ave,	/* average mse */
		I_var,		/* variance of input */
		I_rec_var;	/* variance of input reconstruction */
int		settle_time;	/* max. iterations for a_i */
double		eta_w,		/* learning rate */
		beta;		/* sparseness factor */
int		image_rows=512,	/* training image size */
		image_cols=512; /* training image size */

/*
 * simulation initialization, array allocation, etc.
 */
init_sim(ni,no)
int	ni,no;
{
  int	i;

  num_outputs=no;
  num_inputs=ni;
  size=sqrt(num_inputs);

  I=(double *)calloc(num_inputs,sizeof(double));
  MCHECK(I);
  I_rec=(double *)calloc(num_inputs,sizeof(double));
  MCHECK(I_rec);
  diff=(double *)calloc(num_inputs,sizeof(double));
  MCHECK(diff);
  a=(double *)calloc(num_outputs,sizeof(double));
  MCHECK(a);
  a_mean=(double *)calloc(num_outputs,sizeof(double));
  MCHECK(a_mean);
  a_var=(double *)calloc(num_outputs,sizeof(double));
  MCHECK(a_var);
  a_kurt=(double *)calloc(num_outputs,sizeof(double));
  MCHECK(a_kurt);
  gain=(double *)calloc(num_outputs,sizeof(double));
  MCHECK(gain);
  b=(double *)calloc(num_outputs,sizeof(double));
  MCHECK(b);
  C=(double **)calloc(num_outputs,sizeof(double *));
  MCHECK(C);
  W=(double **)calloc(num_outputs,sizeof(double *));
  MCHECK(W);
  W_stat=(double **)calloc(num_outputs,sizeof(double *));
  MCHECK(W_stat);
  for (i=0; i<num_outputs; i++) {
    C[i]=(double *)calloc(num_outputs,sizeof(double));
    MCHECK(C[i]);
    W[i]=(double *)calloc(num_inputs,sizeof(double));
    MCHECK(W[i]);
    W_stat[i]=(double *)calloc(num_inputs,sizeof(double));
    MCHECK(W_stat[i]);
  }

  init_stats();
  clear_Wstat();
}

/*
 * main loop for training
 */
train_network(num_trials,image)
int	num_trials;
float	*image;
{
  int	t;

  for (t=0; t<num_trials; t++) {
    load_image_data(image);
    compute_network();
    update_a_stats();
    update_w_stats();
    display_trial_no(t);
  }
  update_weights();
}

/*
 * computes outputs and reconstructed input
 */
compute_network()
{
  compute_output();
  reconstruct_input();
}

/*
 * Extract an image patch at random from the training image and load it
 * into input array
 */
load_image_data(image) 
float	*image;
{
  int		i,j,k,start_row,start_col, buff;
  double	sum;
  float		*iptr;
  double	*Iptr;

  buff=4;
  start_row=buff+drand48()*(image_rows-size-2*buff);
  start_col=buff+drand48()*(image_cols-size-2*buff);
  iptr= image+image_cols*start_row+start_col;
  Iptr= I;
  for (j=0; j<size; j++) {
    for (i=0; i<size; i++)
      *Iptr++ = iptr[i];
    iptr += image_cols;
  }

  sum=0;
  for (i=0; i<num_inputs; i++)
    sum += I[i]*I[i];
  I_var=sum/num_inputs;
}

/* 
 * compute the a_i, starting from the b_i
 */
compute_output() 
{
  register int		i;
  register double	sum;
  extern double		input();

  for (i=0; i<num_outputs; i++) {
    b[i]=input(i);
    a[i]= b[i]/C[i][i];
  }

  minimize_a();
}

/*
 * compute the net input to a unit 
 */
double input(i)
int	i;
{
  register double	sum, *Wptr, *Iptr;
  register int		j;
  
  sum=0;
  Wptr=W[i];
  Iptr=I;
  for (j=0; j<num_inputs; j++) {
    sum += *Wptr++ * *Iptr++;
  }
  sum*=gain[i];
  return(sum);
}

#include "nrutil.h"

extern int	ITMAX;

/*
 * Minimize E with respect to the a_i, using conjugate gradient descent
 */
minimize_a()
{
  int		i,n,iter;
  float		*p,ftol,fret;
  extern float	func_a();
  extern void	dfunc_a();

  ITMAX=settle_time;
  n=num_outputs;
  p=vector(1,n);
  for (i=1; i<=num_outputs; i++)
    p[i]=a[i-1];

  ftol=.01;
  frprmn(p,n,ftol,&iter,&fret,func_a,dfunc_a);

  for (i=0; i<num_outputs; i++)
    a[i]=p[i+1];
  
  free_vector(p,1,n);
}

/*
 * Function evaluation used by conj grad descent
 */
float func_a(p)
float *p;
{
  register int		i,j;
  register double	sum,*cptr;
  register float	fval,*p1;
  extern double		sparse();

  fval=0;
  p1=&p[1];
  for (i=0; i<num_outputs; i++) {
    cptr=C[i];
    sum=0;
    for (j=0; j<num_outputs; j++) {
      sum += p1[j] * *cptr++;
    }
    sum *= p1[i];
    fval += sum;
  }
  fval *= 0.5;

  sum=0;
  for (i=0; i<num_outputs; i++)
    sum += p1[i]*b[i];
  fval -= sum;

  sum=0;
  for (i=0; i<num_outputs; i++)
    sum += SIGMA*sparse((double)p1[i]/SIGMA);
  fval += beta*sum;

  return(fval);
}

/*
 * Gradient evaluation used by conj grad descent
 */
void dfunc_a(p,grad)
float	*p,*grad;
{
  register int		i,j;
  register double	sum,*cptr;
  register float	*p1;
  extern double		sparse_prime();

  p1=&p[1];

  for (i=0; i<num_outputs; i++) {
    cptr=C[i];
    sum=0;
    for (j=0; j<num_outputs; j++) {
      sum += p1[j] * *cptr++;
    }
    grad[i+1] = sum - b[i] + beta*sparse_prime((double)p1[i]/SIGMA);
  }
}


/*
 * sparseness cost function
 */
double sparse(x)
double	x;
{
  return(log(1.0+x*x));
}

/*
 * sparseness cost function gradient
 */
double sparse_prime(x)
double	x;
{
  return(2*x/(1.0+x*x));
}

/*
 * alternative sparseness cost function and gradient
 *
double sparse(x)
double	x;
{
  return(-exp(-x*x));
}

double sparse_prime(x)
double	x;
{
  return(2*x*exp(-x*x));
}
*/


/*
 * accumulate running stats on a_i
 */
update_a_stats()
{
  register int		i,j;
  register double	eta,d2;

  eta=.001;

  for (i=0; i<num_outputs; i++) {
    a_mean[i] = (1.0-eta)*a_mean[i] + eta*a[i];
    d2=a[i]*a[i];
    a_var[i] = (1.0-eta)*a_var[i] + eta*(d2);
    a_kurt[i] = (1.0-eta)*a_kurt[i] + eta*(d2*d2);
  }

  update_histo();

  mse_ave = (.999)*mse_ave + (.001)*mse;
}

/*
 * reset a_i stats
 */
init_stats()
{
  register int		i;

  for (i=0; i<num_outputs; i++) {
    gain[i]=1.0;
    a_mean[i] = 0;      
    a_var[i] = VAR_GOAL;
    a_kurt[i]=3*a_var[i]*a_var[i];
  }
  mse_ave = 0;
}


/* 
 * compute image reconstruction from a_i
 */
reconstruct_input()
{
  register int		i,j;
  register double	sum, *aptr, *gptr, **Wptr;

  for (j=0; j<num_inputs; j++) {
    aptr=a;
    gptr=gain;
    Wptr=W;
    sum=0;
    for (i=0; i<num_outputs; i++)
      sum += *aptr++ * *gptr++ * *(*Wptr++ + j);
    I_rec[j]= sum;
    diff[j]=I[j]-I_rec[j];
  }
  mse=0;
  I_rec_var=0;
  for (i=0; i<num_inputs; i++) {
    mse += diff[i]*diff[i];
    I_rec_var += I_rec[i]*I_rec[i];
  }
  I_rec_var/=num_inputs;
  mse/=num_inputs;
}

static int	nstat;

/*
 * accumulate averages used for weight update
 */
update_w_stats()
{
  register int		i,j;
  register double	*aptr;

  for (i=0; i<num_outputs; i++) {
    for (j=0; j<num_inputs; j++) {
      W_stat[i][j] += a[i]*diff[j];
    }
  }

  nstat++;
}

/*
 * reset averages used for weight update
 */
clear_Wstat()
{
  register int		i,j;

  for (i=0; i<num_outputs; i++) {
    for (j=0; j<num_inputs; j++) {
      W_stat[i][j] = 0;
    }
  }
  nstat=0;
}

/*
 * make a weight update
 */
update_weights() 
{
  register int		i,j;
  register double	eta, *Wptr, *Wsptr;
 
  eta = eta_w/nstat;

  for (i=0; i<num_outputs; i++) {
    Wptr=W[i];
    Wsptr=W_stat[i];
    for (j=0; j<num_inputs; j++) {
      *Wptr++ += eta * *Wsptr++;
    }
    normalize(W[i],num_inputs,1.0);
    gain[i] += 0.1*(a_var[i]-VAR_GOAL);
  }
  update_C();
  clear_Wstat();
}

/*
 * recompute weight vector overlaps
 */
update_C()
{
  register int		i,j,k,n,*iptr;
  register double	c,*w,*cptr,thresh;
  extern double		overlap();

  for (i=0; i<num_outputs; i++) {
    cptr=C[i];
    for (j=0; j<=i; j++) {
      *cptr++ = gain[i]*gain[j]*overlap(W[i],W[j]);
    }
  }
  for (i=0; i<num_outputs; i++) {
    cptr=C[i]+i+1;
    for (j=i+1; j<num_outputs; j++) {
      *cptr++ = C[j][i];
    }
  }
}

double	overlap(w1,w2)
double	*w1,*w2;
{
  register int		k;
  register double 	sum;

  sum=0;
  for (k=0; k<num_inputs; k++) {
    sum += *w1++ * *w2++;
  }
  return(sum);
}

/*
 * initialize weight vectors to random numbers
 */
init_weights()
{
  register int		i,j;
 
  for (i=0; i<num_outputs; i++) {
    for (j=0; j<num_inputs; j++) {
      W[i][j] = drand48()-0.5;
    }
    normalize(W[i],num_inputs,1.0);
  }
  update_C();
}

/*
 * weight vector normalization
 */
normalize(v,n,mag)
double *v,mag;
int	n;
{
  register int		i;
  register double	l,sf,*vlast,*vptr;
  extern double		magnitude();

  l=magnitude(v,n);
  if (l>0) {
    sf=mag/l;
    vlast=v+n;
    for (vptr=v; vptr<vlast; vptr++)
      *vptr *= sf;
  }
}

double magnitude(v,n)
double	*v;
int	n;
{
  register int		i;
  register double	*vptr, *vlast, sum;

  sum=0;
  vlast=v+n;
  for (vptr=v; vptr<vlast; vptr++)
    sum += *vptr * *vptr;
  return(sqrt(sum));
}
