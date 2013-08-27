CC=cc
NRDIR=../nr
TKINCL=/usr/include/tcl/
TKLIB=/usr/include/tcl/
CFLAGS=-c -O
OBJS=diff_net_cg.o histo.o display.o display_init.o
SIM_OBJS=sim.o $(OBJS)

sim:	$(SIM_OBJS)
	$(CC) -o sim $(SIM_OBJS) -L$(TKLIB) -L$(NRDIR) -ltk -ltcl -lX11 -lm -lnropt


sim.o:	sim.c net.h
	$(CC) $(CFLAGS) -I$(TKINCL) sim.c

diff_net_cg.o: diff_net_cg.c net.h
	$(CC) $(CFLAGS) -I$(NRDIR) diff_net_cg.c

histo.o: histo.c net.h
	$(CC) $(CFLAGS) histo.c

display.o: display.c net.h display.h
	$(CC) $(CFLAGS) display.c

display_init.o:	display_init.c display.h net.h
	$(CC) $(CFLAGS) display_init.c


gen_data: gen_data.c
	$(CC) -o gen_data -O gen_data.c -lm
