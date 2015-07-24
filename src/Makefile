CC        = g++
#-std=c99
CFLAGS    =  -Wall -m64 -march=native -O3 -msse2 -ffast-math -fassociative-math -ftree-vectorizer-verbose=1 -D DEBUG
CINCL     = -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples/common/inc
CLIBS     = -lfftw3 -lm -lpthread -L /usr/local/cuda/lib 

NVCC      = /usr/local/cuda-7.0/bin/nvcc
LD        = $(NVCC)
NVCCPARMS += -g -G -Xptxas="-v" -arch=sm_52 -m64 -O3 
#-D DEBUG
#-O3 -use_fast_math
# -arch sm_52 <- byc moze potrzeba zeby dobrze double liczyl
# przyspieszanie odstepu pamieci:
# -Xptxas=-dlcm=cg  <- http://stackoverflow.com/questions/11592450/how-to-adjust-the-cuda-number-of-block-and-of-thread-to-get-optimal-performances
# cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
CUINCL    = -I /usr/local/cuda-7.0/include -I /usr/local/cuda/samples/common/inc
# -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples/common/inc -L /usr/local/cuda/lib
CULIBS    = -lcufft -lcublas

PROG=gpe
PROG2=gpe_debug

OBJS =  main.o fileIO.o cudautils.o simulation.o kernels.o

#c: main.c
#	gcc -o $@ main.c $(CLIBS) $(CFLAGS)

# !!!!!!! DEBUG <- mozna ustawiac macra #define przez flage kompilatora -D ... !!!!!!!

all: $(PROG)

       
$(PROG):	$(OBJS)
		$(LD) -o $@ $(OBJS) $(CULIBS) $(CLIBS) -arch=sm_52
       

debug: $(PROG2)
       
$(PROG2):	$(OBJS)
		$(LD) -o $@ $(OBJS) $(CULIBS) $(CLIBS)
       

NVCCINC=-I $(CUDASDK)/common/inc

.SUFFIXES:

%.o:	%.cu
		$(NVCC) -o $@ -c $< $(NVCCPARMS) $(CUINCL) $(CULIBS)

%.o:	%.c
		$(CC) $(CFLAGS) $(CINCL) $(CLIBS) -o $@ -c $<

#.PHONY:
clean:
	rm -f *.o *.linkinfo