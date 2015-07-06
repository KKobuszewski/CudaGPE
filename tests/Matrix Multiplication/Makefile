CC        = g++
NVCC      = /usr/local/cuda-7.0/bin/nvcc
LD        = $(NVCC)
#CFLAGS    = -Wall -I /usr/local/cuda/include
CFLAGS    = -std=c99 -Wall -I /usr/local/cuda/include -march=native -msse2 -ffast-math -fassociative-math -O3 -ftree-vectorizer-verbose=2
LIBS      = 

#NVCCPARMS += -arch sm_13

PROG=cu_matrixtest

OBJS =  main.o cpu_matrix_mul.o gpu_matrix_mul.o

all: $(PROG)
       
$(PROG):	$(OBJS)
		$(LD) -o $@ $(OBJS) $(LIBS)
       

NVCCINC=-I $(CUDASDK)/common/inc

.SUFFIXES:

%.o:	%.cu
		$(NVCC) $(NVCCPARMS) -o $@ -c $<

%.o:	%.c
		$(CC) $(CFLAGS) -o $@ -c $<

.PHONY: clean

clean:
	rm -f *.o $(PROG) *.linkinfo
