/*
 * routines for keeping histograms of outputs
 */

#include <stdio.h>
#include <math.h>
#include "net.h"

static int	**histo;	/* histogram */
static int	num_trials;	/* trials count */
static int	num_bins=150;	/* number of bins in histogram */
static double	max=3.0,	/* max and min range */
		min=-3.0;
static double	interval;	/* bin interval */


/*
 * allocate histogram and initialize
 */
init_histo()
{
  int	i;

  histo=(int **)malloc(num_outputs*sizeof(int *));
  MCHECK(histo);
  for (i=0; i<num_outputs; i++) {
    histo[i]=(int *)calloc(num_bins,sizeof(int));
    MCHECK(histo[i]);
  }

  num_trials=0;

  interval=(max-min)/num_bins;
}

/*
 * update histogram 
 */
update_histo()
{
  register int	i,j;

  for (i=0; i<num_outputs; i++) {
    j=(a[i] - min)/interval;
    if (j<0)
      j=0;
    else if (j>=num_bins)
      j=num_bins-1;

    histo[i][j]++;
  }

  num_trials++;
}

/*
 * dump histogram i to file "histogram"
 */
dump_histo(i)
int	i;
{
  register double	H,p;
  register int		j;
  FILE			*fp;

  if (i<0 || i>=num_outputs)
    return;
  
  if ((fp=fopen("histogram","w"))==NULL) {
    fprintf(stderr,"sim: couldn't open histogram file");
    return;
  }

  H=0;
  for (j=0; j<num_bins; j++) {
    p=(double)histo[i][j]/num_trials;
    fprintf(fp,"%lf %lf\n", min+interval*j, p);
    if (p>0)
      H -= p*log(p);
  }
  
  fclose(fp);
  
  printf("unit %d: entropy=%lf nats\n", i, H);
}

/*
 * dump an average of the histograms on all units to file "total_histo"
 */
dump_total_histo()
{
  register double	H,p,p_ave;
  register int		i,j;
  FILE			*fp;

  if ((fp=fopen("total_histo","w"))==NULL) {
    fprintf(stderr,"sim: couldn't open histogram file");
    return;
  }

  H=0;
  for (j=0; j<num_bins; j++) {
    p_ave=0;
    for (i=0; i<num_outputs; i++) {
      p=(double)histo[i][j]/num_trials;
      if (p>0)
	H -= p*log(p);
      p_ave+=p;
    }
    fprintf(fp,"%lf %lf\n", min+interval*j, p_ave/num_outputs);
  }
  
  fclose(fp);
  
  printf("sum entropy=%lf nats\n", H);
}

