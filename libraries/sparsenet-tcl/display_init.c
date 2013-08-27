/*
 * display initialization
 */

#include <stdio.h>
#include <math.h>
#include "net.h"
#include "display.h"

Display 	*display;			/* X display */
int     	screen;				/* screen */
Window  	win;				/* window */
GC		gc;				/* graphics context */
XEvent		event;				/* event struct */
unsigned long	event_mask;			/* event mask */
unsigned long	grey_table[NUM_GREY_LEVELS];	/* table for grey values */
unsigned long	white,black,light_grey,		/* various colors */
		dark_grey,blue,red;

Coord		*weight_pos,			/* weight positions */
		input_pos,			/* input position */
		chart_pos;			/* chart position */
int		weight_width, weight_height,	/* weight dimensions */
		chart_height, chart_width,	/* output chart dimensions */
		input_height, input_width;	/* input dimensions */

/*
 * initialize the display
 */
init_display()
{
  unsigned int	width,height;
  unsigned int	foreground,background,border_width,border_color;
  Window	parent;
  Visual	*visual;
  XGCValues	gc_values;
  int		num_down,num_across,i,x,y,weight_offsety;

  if ((display = XOpenDisplay(NULL)) == NULL) {
    fprintf(stderr,"Can't open display %s\n",XDisplayName(NULL));
    exit(1);
  }
  screen = DefaultScreen(display);

  /* object positions positions and sizes */

  weight_pos=(Coord *)malloc(num_outputs*sizeof(Coord));
  MCHECK(weight_pos);

  if (num_outputs>num_inputs) {
    num_down=size;
    num_across=num_outputs/num_down;
  }
  else {
    num_across=size;
    num_down=num_outputs/num_across;
  }
  if (num_across*num_down<num_outputs)
    num_down++;

  chart_height=1.5*100;
  chart_width=num_outputs*BAR_WIDTH+BAR_WIDTH/2;
  chart_pos.x=BUFF;
  chart_pos.y=BUFF;
  weight_width=size*PIX_SIZE;
  weight_height=size*PIX_SIZE;
  weight_offsety=2*BUFF+chart_height;
  input_width=size*PIX_SIZE;
  input_height=3*input_width+BUFF;
  input_pos.x=BUFF+(chart_width-input_width)/2;
  input_pos.y=BUFF+chart_height+BUFF +num_down*(weight_height+BUFF/2)+BUFF;

  width=BUFF+num_across*(weight_width+BUFF/2)+BUFF;
  height=BUFF+chart_height+BUFF +num_down*(weight_height+BUFF/2)+BUFF
    +input_height+BUFF;

  for (i=0; i<num_outputs; i++) {
    y=i/num_across;
    x=i%num_across;
    weight_pos[i].x=BUFF+x*(weight_width+BUFF/2);
    weight_pos[i].y=weight_offsety+y*(weight_height+BUFF/2);
  }
  
  allocate_colors();

  foreground=white;
  background=dark_grey;
  parent=RootWindow(display,screen);
  border_width=4;
  border_color=foreground;
  win = XCreateSimpleWindow(display,parent,100,100,
			    width,height, border_width,
			    border_color, background);
  event_mask=ExposureMask|ButtonPressMask|KeyPressMask;
  XSelectInput(display, win, event_mask);
  XStoreName(display,win,"net");
  gc_values.background=background;
  gc=XCreateGC(display,win,GCBackground, &gc_values);

  XMapWindow(display,win);

  XWindowEvent(display,win,ExposureMask,&event);
  redraw();
}


allocate_colors()
{
  int		shift,i,n;
  Colormap	cmap;
  XColor	color,exact_color;
  
  cmap=XDefaultColormap(display,screen);
  shift=8;
  n=256/NUM_GREY_LEVELS;
  for (i=0; i<NUM_GREY_LEVELS; i++) {
    color.red   = (i*n) <<shift;
    color.green = (i*n) <<shift;
    color.blue  = (i*n) <<shift;
    if (!XAllocColor(display,cmap,&color))
      fatal_error("not enough colors");
    grey_table[i]=color.pixel;
  }
  black=grey_table[0];
  white=grey_table[NUM_GREY_LEVELS-1];
  light_grey=grey_table[2*NUM_GREY_LEVELS/3];
  dark_grey=grey_table[NUM_GREY_LEVELS/3];
  XAllocNamedColor(display,cmap,"red",&color,&exact_color);
  red=color.pixel;
  XAllocNamedColor(display,cmap,"blue",&color,&exact_color);
  blue=color.pixel;
}
