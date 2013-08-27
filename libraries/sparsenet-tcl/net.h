#define MCHECK(m) if (!m) fatal_error("couldn't allocate core")


extern int	num_inputs, num_outputs, size;
extern double	*I, *I_rec, *diff, *a, *gain, *a_mean, *a_var, *a_kurt, **W;
extern double	alpha;
extern double	mse, mse_ave, I_var, I_rec_var;
extern int	settle_time;
extern double	eta_w, beta, lambda;
extern int	tau;
extern int	image_rows, image_cols;
