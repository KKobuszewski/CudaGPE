CC        = g++
NVCC      = /usr/local/cuda-7.0/bin/nvcc
LD        = $(NVCC)
CFLAGS    = -Wall -O3 -I /usr/local/cuda/include -march=native -msse2 -ffast-math -fassociative-math -ftree-vectorizer-verbose=2
LIBS      = 

#NVCCPARMS += -arch sm_13

PROG=cu_add_in_place

OBJS =  main.o kernels.o

all: $(PROG)
       
$(PROG):	$(OBJS)
		$(LD) -o $@ $(OBJS) $(LIBS)
       

NVCCINC=-I $(CUDASDK)/common/inc

.SUFFIXES:

%.o:	%.cu
		$(NVCC) $(NVCCPARMS) -o $@ -c $<

%.o:	%.c
		$(CC) $(CFLAGS) -o $@ -c $<

clean:
	rm -f *.o $(PROG) *.linkinfo
