/*
 * network display
 */

#include <stdio.h>
#include <math.h>
#include "net.h"
#include "display.h"

/*
 * redraw everything
 */
redraw()
{
  XClearWindow(display,win);
  display_output();
  display_weights();
  display_input();
  XFlush(display);
}

/*
 * display the output of the network
 */
display_output()
{
  int	i,x,y,cy,xoff;

  XSetForeground(display,gc,white);
  XFillRectangle(display,win,gc,
		 chart_pos.x, chart_pos.y, chart_width, chart_height);
  XSetForeground(display,gc,black);

  XSetLineAttributes(display,gc,BAR_WIDTH,LineSolid,CapButt,JoinMiter);
  cy=chart_height/3;
  xoff=BAR_WIDTH/2;
  for (i=0; i<num_outputs; i++) {
    y=a[i]*cy;
    x=xoff+i*BAR_WIDTH;
    XDrawLine(display,win,gc,
	      chart_pos.x+x, chart_pos.y+cy, chart_pos.x+x, chart_pos.y+cy-y);
  }

  XSetLineAttributes(display,gc,BAR_WIDTH/2,LineSolid,CapButt,JoinMiter);
  XSetForeground(display,gc,blue);
  for (i=0; i<num_outputs; i++) {
    y=sqrt(a_var[i])*cy;
    x=xoff+i*BAR_WIDTH;
    XDrawLine(display,win,gc,
	      chart_pos.x+x, chart_pos.y+chart_height-1,
	      chart_pos.x+x, chart_pos.y+chart_height-1-y);
  }
  XSetForeground(display,gc,red);
  for (i=0; i<num_outputs; i++) {
/*    y=pow(a_kurt[i],0.25)*cy; */
    y=gain[i]*cy;
    x=xoff+(i+0.5)*BAR_WIDTH;
    XDrawLine(display,win,gc,
	      chart_pos.x+x, chart_pos.y+chart_height-1,
	      chart_pos.x+x, chart_pos.y+chart_height-1-y);
  }
}

/*
 * display the weights
 */
display_weights()
{
  int		i,j,n,m,p;
  double	w_max;
  char		line[80];

  w_max=0.5;
  for (i=0; i<num_outputs; i++) {
    for (j=0; j<num_inputs; j++) {
      if (fabs(W[i][j])>w_max)
	w_max=fabs(W[i][j]);
    }
  }
  for (i=0; i<num_outputs; i++) {
    j=0;
    for (n=0; n<size; n++) {
      for (m=0; m<size; m++) {
	p=(NUM_GREY_LEVELS-1)*(W[i][j]+w_max)/(2*w_max);
	XSetForeground(display,gc,grey_table[p]);
	XFillRectangle(display,win,gc,
		       weight_pos[i].x+m*PIX_SIZE, weight_pos[i].y+n*PIX_SIZE,
		       PIX_SIZE, PIX_SIZE);
	j++;
      }
    }
  }
}

/* 
 * display the input image
 */
display_input()
{
  int	i,n,m,p;
  char	var[80];

  i=0;
  for (n=0; n<size; n++) {
    for (m=0; m<size; m++) {
      p=(NUM_GREY_LEVELS-1)*0.5*(I[i]+1.0);
      p= (p<0) ? 0 : (p>=NUM_GREY_LEVELS) ? NUM_GREY_LEVELS-1 : p;
      XSetForeground(display,gc,grey_table[p]);
      XFillRectangle(display,win,gc,
		     input_pos.x+m*PIX_SIZE, input_pos.y+n*PIX_SIZE,
		     PIX_SIZE, PIX_SIZE);
      i++;
    }
  }
  sprintf(var,"var=%lf",I_var);
  XSetForeground(display,gc,white);
  XDrawImageString(display,win,gc,
		   input_pos.x+size*PIX_SIZE+BUFF,
		   input_pos.y+size*PIX_SIZE/2,
		   var, strlen(var));
  
  display_reconstruction();
}

/*
 * display the reconstruction image
 */
display_reconstruction()
{
  int	i,n,m,p;
  char	var[80];

  i=0;
  for (n=0; n<size; n++) {
    for (m=0; m<size; m++) {
      p=NUM_GREY_LEVELS*(I_rec[i]+1.0)/2.0;
      p= (p<0) ? 0 : (p>=NUM_GREY_LEVELS) ? NUM_GREY_LEVELS-1 : p;
      XSetForeground(display,gc,grey_table[p]);
      XFillRectangle(display,win,gc,
		     input_pos.x+m*PIX_SIZE,
		     input_pos.y+size*PIX_SIZE+BUFF/2+n*PIX_SIZE,
		     PIX_SIZE, PIX_SIZE);
      i++;
    }
  }
  sprintf(var,"var=%lf",I_rec_var);
  XSetForeground(display,gc,white);
  XDrawImageString(display,win,gc,
		   input_pos.x+size*PIX_SIZE+BUFF,
		   input_pos.y+(3*size*PIX_SIZE+BUFF)/2,
		   var, strlen(var));

  display_diff();
}

/*
 * display the difference image 
 */
display_diff()
{
  int	i,n,m,p;
  char	error[80];

  i=0;
  for (n=0; n<size; n++) {
    for (m=0; m<size; m++) {
      p=NUM_GREY_LEVELS*(diff[i]+1.0)/2.0;
      p= (p<0) ? 0 : (p>=NUM_GREY_LEVELS) ? NUM_GREY_LEVELS-1 : p;
      XSetForeground(display,gc,grey_table[p]);
      XFillRectangle(display,win,gc,
		     input_pos.x+m*PIX_SIZE,
		     input_pos.y+2*size*PIX_SIZE+BUFF+n*PIX_SIZE,
		     PIX_SIZE, PIX_SIZE);
      i++;
    }
  }
  XSetForeground(display,gc,white);
  sprintf(error,"mse=%lf",mse);
  XDrawImageString(display,win,gc,
		   input_pos.x+size*PIX_SIZE+BUFF,
		   (int)(input_pos.y+2.5*size*PIX_SIZE+BUFF),
		   error, strlen(error));
  sprintf(error,"mse_ave=%lf",mse_ave);
  XDrawImageString(display,win,gc,
		   input_pos.x+size*PIX_SIZE+BUFF,
		   (int)(input_pos.y+3*size*PIX_SIZE+BUFF),
		   error, strlen(error));
}

/*
 * display the iteration number
 */
display_trial_no(t)
int	t;
{
  char	line[20];

  sprintf(line,"%3d",t);
  XSetForeground(display,gc,white);
  XDrawImageString(display,win,gc, 10, input_pos.y+input_height+BUFF-1,
		   line, strlen(line));
  XFlush(display);
}

/*
 * check for input events
 */
check_events()
{
  if (XCheckWindowEvent(display,win,event_mask,&event)==True) {
    if (event.type==Expose)
      redraw();
    else if (event.type==ButtonPress)
      process_button_press(&event);
  }
}

/*
 * process button events
 */
process_button_press(button_event)
XButtonEvent	*button_event;
{
  int	x,y,i;

  x=button_event->x - chart_pos.x;
  y=button_event->y - chart_pos.y;

  if (x>=0 && x<chart_width && y>0 && y<chart_height) {
    i = x/BAR_WIDTH;
    dump_histo(i);
  }
}
