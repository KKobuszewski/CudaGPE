CC        = g++
NVCC      = /usr/local/cuda-7.0/bin/nvcc
LD        = $(NVCC)
CFLAGS    = -std=c99 -Wall -I /usr/local/cuda/include -march=native -O3 -msse2 -ffast-math -fassociative-math -ftree-vectorizer-verbose=1
CLIBS     = -lfftw3 -lm
CULIBS      = -lcufft

NVCCPARMS += -g -G -Xptxas="-v" -I /usr/local/cuda-7.0/include
#-O3 -use_fast_math
# -arch sm_13 <- byc moze potrzeba zeby dobrze double liczyl 

PROG=dft2_test

OBJS =  main.o fftw_dft.o cuda_dft.o cudautils.o

fftw3: main.c fftw_dft.c
	gcc -o $@ main.c fftw_dft.c $(CLIBS) $(CFLAGS)

all: $(PROG)
       
$(PROG):	$(OBJS)
		$(LD) -o $@ $(OBJS) $(CULIBS) $(CLIBS)
       

NVCCINC=-I $(CUDASDK)/common/inc

.SUFFIXES:

%.o:	%.cu
		$(NVCC) -o $@ -c $< $(NVCCPARMS) $(CULIBS)

%.o:	%.c
		$(CC) $(CFLAGS) -o $@ -c $<

clean:
	rm -f *.o $(PROG) *.linkinfo
