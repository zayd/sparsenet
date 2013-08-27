/*
 * sim.c:	TCL/TK shell for running network simulation
 *
 */
#include <stdio.h>
#include <fcntl.h>
#include <sys/time.h>
#include <tcl.h>
#include <tk.h>

#include "net.h"

#define FLOAT_TO_INT(x)  ((int)(100*(x)))
#define INT_TO_FLOAT(i)  ((double)(i)/100.0)

/*
 * images used for training
 */
#define IMAGE_FILE "trainset.test/image%d"
#define NUM_IMAGES 1

extern double	drand48();

static float	*image_list[NUM_IMAGES], *image;
static struct timeval	tp;

extern int main();
int *tclDummyMainPtr = (int *) main;

/*
main(argc, argv)
int	argc;
char	**argv;
{
  Tk_Main(argc, argv, Tcl_AppInit);
}
*/

/*
 * initialize the simulation by setting the number of inputs and outputs,
 * reading the training images, and initializing display.
 */
int InitSimCmd(ClientData clientData, Tcl_Interp *interp,
	       int argc, char *argv[]) {
  int	ni,no;

  if (argc < 3) {
    interp->result = "wrong # args";
    return TCL_ERROR;
  }
  ni=atoi(argv[1]);
  no=atoi(argv[2]);
  init_sim(ni,no);
  init_weights();
  init_histo();

  read_images();
  image=image_list[0];

  // gettimeofday(&tp);
  // srand48(tp.tv_sec);

  init_display();
  return TCL_OK;
}

/*
 * Step the simulation either one step, or n steps with a weight update
 * at the end.
 */
int StepNetCmd(ClientData clientData, Tcl_Interp *interp,
	       int argc, char *argv[]) {
  int	i,n;

  if (argc > 2) {
    interp->result = "wrong # args";
    return TCL_ERROR;
  }
  n=0;
  if (argc>1)
    n=atoi(argv[1]);

  if (n) {
    i=NUM_IMAGES*drand48();
    image=image_list[i];
    train_network(n,image);
  }
  else {
    load_image_data(image);
    compute_network();
  }

  display_output();
  display_weights();
  display_input();
  check_events();
  
  return TCL_OK;
}

/* 
 * Read the training images into memory
 */
read_images()
{
  int		i, fd;
  char		image_name[80];

  for (i=0; i<NUM_IMAGES; i++) {
    sprintf(image_name,IMAGE_FILE,i);
    if ((fd=open(image_name,O_RDONLY))==-1)
      fatal_error("couldn't open image file");
    image_list[i]=(float *)malloc(image_rows*image_cols*sizeof(float));
    MCHECK(image_list[i]);
    read(fd,image_list[i],image_rows*image_cols*sizeof(float));
    close(fd);
  }
}

/*
 * Load the saved state from a previous run - expects a file named
 * "weights" that matches the inputs/outputs of the current simulation,
 * and file named "gain" that matches the number of outputs.
 */
int LoadStateCmd(ClientData clientData, Tcl_Interp *interp,
		   int argc, char *argv[]) {
  int		rows,cols,num_frame,i,j,fd;
  float		*tmp_array,*tptr;

  if ((fd=open("weights",O_RDONLY))==-1) {
    interp->result="couldn't open weights file";
    return(TCL_ERROR);
  }
  read(fd,&rows,sizeof(int));
  read(fd,&cols,sizeof(int));
  read(fd,&num_frame,sizeof(int));
  if (rows!=size || cols!=size || num_frame!=num_outputs) {
    interp->result="weights file doesn't match network";
    return(TCL_ERROR);
  }
  MCHECK((tmp_array=(float *)malloc(num_inputs*sizeof(float))));
  for (i=0; i<num_outputs; i++) {
    read(fd,tmp_array,num_inputs*sizeof(float));
    tptr=tmp_array;
    for (j=0; j<num_inputs; j++)
      W[i][j] = *tptr++;
  }
  free(tmp_array);
  close(fd);

  if ((fd=open("gain",O_RDONLY))==-1) {
    update_C();
    interp->result="couldn't open gain file";
    return(TCL_ERROR);
  }
  read(fd,&cols,sizeof(int));
  if (cols!=num_outputs) {
    update_C();
    interp->result="gain file doesn't match network";
    return(TCL_ERROR);
  }
  MCHECK((tmp_array=(float *)malloc(num_outputs*sizeof(float))));
  read(fd,tmp_array,num_outputs*sizeof(float));
  tptr=tmp_array;
  for (i=0; i<num_outputs; i++) {
    gain[i] = *tptr++;
  }
  free(tmp_array);
  close(fd);

  update_C();
  update_net();

  return TCL_OK;
}

/*
 * Save the state of the network into files "weights" and "gain"
 */
int SaveStateCmd(ClientData clientData, Tcl_Interp *interp,
		   int argc, char *argv[]) {
  int		i,j,fd;
  float		*tmp_array,*tptr;

  if ((fd=open("weights",O_WRONLY|O_CREAT,0644))==-1) {
    interp->result="couldn't open weight file";
    return(TCL_ERROR);
  }
  write(fd,&size,sizeof(int));
  write(fd,&size,sizeof(int));
  write(fd,&num_outputs,sizeof(int));
  tmp_array=(float *)malloc(num_inputs*sizeof(float));
  MCHECK(tmp_array);
  for (i=0; i<num_outputs; i++) {
    tptr=tmp_array;
    for (j=0; j<num_inputs; j++)
      *tptr++=W[i][j];
    write(fd,tmp_array,num_inputs*sizeof(float));
  }
  free(tmp_array);
  close(fd);

  if ((fd=open("gain",O_WRONLY|O_CREAT,0644))==-1) {
    interp->result="couldn't open gain file";
    return(TCL_ERROR);
  }
  write(fd,&num_outputs,sizeof(int));
  MCHECK((tmp_array=(float *)malloc(num_outputs*sizeof(float))));
  tptr=tmp_array;
  for (i=0; i<num_outputs; i++) {
    *tptr++=gain[i];
  }
  write(fd,tmp_array,num_outputs*sizeof(float));
  free(tmp_array);
  close(fd);

  dump_total_histo();

  return TCL_OK;
}

/*
 * Set a parameter in the simulation
 */
int SetVarCmd(ClientData clientData, Tcl_Interp *interp,
	      int argc, char *argv[]) {
  int	val;

  if (argc < 3) {
    interp->result = "wrong # args";
    return TCL_ERROR;
  }
  val=atoi(argv[2]);
  if (strcmp(argv[1],"beta")==0) {
    beta=INT_TO_FLOAT(val);
    update_net();
  }
  else if (strcmp(argv[1],"eta")==0) {
    eta_w=INT_TO_FLOAT(val);
  }
  else if (strcmp(argv[1],"settle")==0) {
    settle_time=val;
    update_net();
  }

  return TCL_OK;
}

/*
 * update the network
 */
update_net()
{
  compute_network();
  display_output();
  display_weights();
  display_reconstruction();
  check_events();
}

/*
 * Initialize the TCL interpreter
 */
int
Tcl_AppInit(interp)
    Tcl_Interp *interp;		/* Interpreter for application. */
{
  Tk_Window main;

  main = Tk_MainWindow(interp);

  if (Tcl_Init(interp) == TCL_ERROR) {
    return TCL_ERROR;
  }
  if (Tk_Init(interp) == TCL_ERROR) {
    return TCL_ERROR;
  }

  Tcl_CreateCommand(interp, "initSim", InitSimCmd,
		    (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
  Tcl_CreateCommand(interp, "stepNet", StepNetCmd,
		    (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
  Tcl_CreateCommand(interp, "loadState", LoadStateCmd,
		    (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
  Tcl_CreateCommand(interp, "saveState", SaveStateCmd,
		    (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
  Tcl_CreateCommand(interp, "setVar", SetVarCmd,
		    (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
  

/*  tcl_RcFileName = "~/.wishrc";      */
  return TCL_OK;
}


fatal_error(s)
char *s;
{
  fprintf(stderr, "%s\n", s);
  exit(1);
}
