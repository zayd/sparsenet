/*
 * gen_data.c:	generates sparse test data
 *
 * usage:	gen_data [lambda] > image_name
 */

#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define SIZE 512

extern double	drand48(), atof();

main(argc, argv)
int	argc;
char	**argv;
{
  int		i,j;
  double	lambda=5.0;
  float		image[SIZE*SIZE],*iptr;
  struct timeval tp;
  extern double	sparse_distr();
  
  if (argc>1)
    lambda=atof(argv[1]);
  
  gettimeofday(&tp);
  srand48(tp.tv_sec);

  iptr=image;
  for (i=0; i<SIZE; i++) {
    for (j=0; j<SIZE; j++) {
      *iptr++ = (float) (((drand48()>=0.5)? 1 : -1)*sparse_distr(lambda));
    }
  }

  write(1,image,SIZE*SIZE*sizeof(float));
}

double sparse_distr(lambda)
double	lambda;
{
  double	x,x_min;

  x_min=exp(-lambda);
  x=x_min+(1.0-x_min)*drand48();
  return(-log(x)/lambda);
}
