CC= cc
CFLAGS=	-O -I.

OPTOBJS=	frprmn.o linmin.o brent.o mnbrak.o nrutil.o

.c.o:
	${CC} ${CFLAGS} -c $*.c

libnrfopt.a:	$(OPTOBJS)
		ar cr libnrfopt.a $(OPTOBJS)

