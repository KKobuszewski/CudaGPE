CC        = g++ 
#-std=c++11
#-std=c99
CFLAGS    =  -Wall -Wundef -m64 -march=native -O3 -msse4 -ffast-math -fassociative-math -ftree-vectorizer-verbose=1
CINC      = -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples/common/inc
CLIBS     = -lgsl -lm -lpthread -L /usr/local/cuda/lib64
#-lfftw3 
NVCC      = /usr/local/cuda-7.0/bin/nvcc
LD        = $(NVCC)
NVCCPARMS += -g -G -Xptxas="-v" -arch=sm_52 -m64 -O3 -lm -Xcompiler -fopenmp
#-O3 -use_fast_math
# -arch sm_52 <- byc moze potrzeba zeby dobrze double liczyl
# przyspieszanie odstepu pamieci:
# -Xptxas=-dlcm=cg  <- http://stackoverflow.com/questions/11592450/how-to-adjust-the-cuda-number-of-block-and-of-thread-to-get-optimal-performances
# cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
CUINCL    = -I /usr/local/cuda-7.0/include -I /usr/local/cuda/samples/common/inc
# -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples/common/inc -L /usr/local/cuda/lib
NVCCINC=-I $(CUDASDK)/common/inc
CULIBS    = -lcufft -lcublas -lgomp

# one must define evolution type: EVOLUTION = REAL_TIME/IMAG_TIME (these values are predefined macros in code)
DEF       = -DV_EXT -DV_CON 
#-DV_DIP
# -DLOOSES
# ...

PROG=../bin/gpe
PROG2=../bin/gpe_debug
PROG_IMAG=../bin/gpe_imaginary

OBJS =  ../bin/main.o ../bin/fileIO.o ../bin/cudautils.o ../bin/simulation.o ../bin/kernels.o

#c: main.c
#	gcc -o $@ main.c $(CLIBS) $(CFLAGS)

all: $(PROG) $(PROG2) $(PROG_IMAG)

real: DEF += -D REAL_TIME
real: $(PROG)
       
$(PROG):	$(OBJS)
		$(LD) -o $@ $(OBJS) $(CULIBS) $(CLIBS) -arch=sm_52
       



debug: DEF += -DDEBUG -D REAL_TIME
debug: $(PROG2)
       
$(PROG2):	$(OBJS)
		$(LD) -o $@ $(OBJS) $(CULIBS) $(CLIBS)
       


imag: DEF += -D IMAG_TIME
imag: $(PROG_IMAG)
       
$(PROG_IMAG):	$(OBJS)
		$(LD) -o $@ $(OBJS) $(CULIBS) $(CLIBS)
       



.SUFFIXES:

../bin/%.o:	%.cu
		$(NVCC) -o $@ -c $< $(NVCCPARMS) $(CUINCL) $(CULIBS) $(DEF)

../bin/%.o:	%.c
		$(CC) $(CFLAGS) $(CINC) $(CLIBS) $(DEF) -o $@ -c $<

#.PHONY:
clean:
	rm -f ../bin/*.o ../bin/*.linkinfo