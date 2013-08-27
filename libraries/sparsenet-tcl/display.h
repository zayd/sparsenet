#include <X11/Xlib.h>

#define NUM_COLOR_LEVELS 64
#define NUM_GREY_LEVELS 64
#define BUFF 15
#define PIX_SIZE 4
#define BAR_WIDTH 4

extern Display 		*display;
extern int     		screen;
extern Window  		win;
extern GC		gc;
extern XEvent		event;
extern unsigned long	event_mask;
extern unsigned long	grey_table[NUM_GREY_LEVELS];
extern unsigned long	white,black,light_grey,dark_grey,blue,red;

typedef struct {
  int	x;
  int	y;
} Coord;

extern Coord		*weight_pos, input_pos, chart_pos, corr_pos;
extern int		weight_width;
extern int		chart_height, chart_width, corr_height, corr_width,
			input_width, input_height;

