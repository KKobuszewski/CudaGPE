CC        = g++
#-std=c99
CFLAGS    =  -Wall -m64 -march=native -O3 -msse2 -ffast-math -fassociative-math -ftree-vectorizer-verbose=1 -D DEBUG
CINCL     = -I /usr/local/cuda/include -I /usr/local/cuda/samples/common/inc
CLIBS     = -lfftw3 -lm -lpthread

NVCC      = /usr/local/cuda-7.0/bin/nvcc
LD        = $(NVCC)
NVCCPARMS += -g -G -Xptxas="-v" -m64
#-O3 -use_fast_math
# -arch sm_?? <- byc moze potrzeba zeby dobrze double liczyl
CUINCL    = -I /usr/local/cuda-7.0/include -I /usr/local/cuda/samples/common/inc
CULIBS    = -lcufft 

PROG=gpe

OBJS =  main.o cudautils.o simulation.o

#c: main.c
#	gcc -o $@ main.c $(CLIBS) $(CFLAGS)

# !!!!!!! DEBUG <- mozna ustawiac macra #define przez flage kompilatora -D ... !!!!!!!

all: $(PROG)
       
$(PROG):	$(OBJS)
		$(LD) -o $@ $(OBJS) $(CULIBS) $(CLIBS)
       

NVCCINC=-I $(CUDASDK)/common/inc

.SUFFIXES:

%.o:	%.cu
		$(NVCC) -o $@ -c $< $(NVCCPARMS) $(CUINCL) $(CULIBS)

%.o:	%.c
		$(CC) $(CFLAGS) $(CINCL) $(CLIBS) -o $@ -c $<

#.PHONY:
clean:
	rm -f *.o $(PROG) *.linkinfo