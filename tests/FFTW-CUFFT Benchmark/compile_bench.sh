#!/bin/bash

export NFFTS="1000"

# non pow 2, out of place
#gcc -std=c99 -DONED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.np2.oop.serial  
#gcc -std=c99 -DONED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.np2.oop.threads 
#/usr/local/cuda-7.0/bin/nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.cuda -DHOSTMEM 

# pow 2, out of place
gcc -std=c99 -DONED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.p2.oop.serial  -DPOW2 
gcc -std=c99 -DONED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.p2.oop.threads -DPOW2 
/usr/local/cuda-7.0/bin/nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.cuda -DHOSTMEM -DPOW2 

# non pow 2, in place
#gcc -std=c99 -DONED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.np2.ip.serial  -DINPLACE 
#gcc -std=c99 -DONED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.np2.ip.threads -DINPLACE 
#/usr/local/cuda-7.0/bin/nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.cuda -DHOSTMEM -DINPLACE 

# pow 2, in place
gcc -std=c99 -DONED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.p2.ip.serial  -DINPLACE -DPOW2 
gcc -std=c99 -DONED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.p2.ip.threads -DINPLACE -DPOW2 
/usr/local/cuda-7.0/bin/nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.cuda -DHOSTMEM -DINPLACE -DPOW2 

# device memory only
/usr/local/cuda-7.0/bin/nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.dev.cuda  -DINPLACE -DPOW2 
# /usr/local/cuda-7.0/bin/nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.dev.cuda  -DINPLACE 
/usr/local/cuda-7.0/bin/nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.dev.cuda  -DPOW2
# /usr/local/cuda-7.0/bin/nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.dev.cuda 

## two dimensional

# non pow 2, out of place
#gcc -std=c99 -DTWOD -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.np2.oop.serial.2d
#gcc -std=c99 -DTWOD -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.np2.oop.threads.2d
#/usr/local/cuda-7.0/bin/nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.cuda.2d -DHOSTMEM

# pow 2, out of place
gcc -std=c99 -DTWOD -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.p2.oop.serial.2d  -DPOW2
gcc -std=c99 -DTWOD -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.p2.oop.threads.2d -DPOW2
/usr/local/cuda-7.0/bin/nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.cuda.2d -DHOSTMEM -DPOW2

# non pow 2, in place
#gcc -std=c99 -DTWOD -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.np2.ip.serial.2d  -DINPLACE
#gcc -std=c99 -DTWOD -Dexport NFFTS="1000"

# non pow 2, out of place
#gcc -std=c99 -DONED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.np2.oop.serial  
#gcc -std=c99 -DONED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.np2.oop.threads 
#/usr/local/cuda-7.0/bin/nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.cuda -DHOSTMEM 

# pow 2, out of place
gcc -std=c99 -DONED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.p2.oop.serial  -DPOW2 
gcc -std=c99 -DONED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.p2.oop.threads -DPOW2 
/usr/local/cuda-7.0/bin/nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.cuda -DHOSTMEM -DPOW2 

# non pow 2, in place
#gcc -std=c99 -DONED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.np2.ip.serial  -DINPLACE 
#gcc -std=c99 -DONED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.np2.ip.threads -DINPLACE 
#/usr/local/cuda-7.0/bin/nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.cuda -DHOSTMEM -DINPLACE 

# pow 2, in place
gcc -std=c99 -DONED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.p2.ip.serial  -DINPLACE -DPOW2 
gcc -std=c99 -DONED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.p2.ip.threads -DINPLACE -DPOW2 
/usr/local/cuda-7.0/bin/nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.cuda -DHOSTMEM -DINPLACE -DPOW2 

# device memory only
/usr/local/cuda-7.0/bin/nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.dev.cuda  -DINPLACE -DPOW2 
# /usr/local/cuda-7.0/bin/nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.dev.cuda  -DINPLACE 
/usr/local/cuda-7.0/bin/nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.dev.cuda  -DPOW2
# /usr/local/cuda-7.0/bin/nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.dev.cuda 

## two dimensional

# non pow 2, out of place
#gcc -std=c99 -DTWOD -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.np2.oop.serial.2d
#gcc -std=c99 -DTWOD -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.np2.oop.threads.2d
#/usr/local/cuda-7.0/bin/nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.cuda.2d -DHOSTMEM

# pow 2, out of place
gcc -std=c99 -DTWOD -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.p2.oop.serial.2d  -DPOW2
gcc -std=c99 -DTWOD -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.p2.oop.threads.2d -DPOW2
/usr/local/cuda-7.0/bin/nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.cuda.2d -DHOSTMEM -DPOW2

# non pow 2, in place
#gcc -std=c99 -DTWOD -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.np2.ip.serial.2d  -DINPLACE
#gcc -std=c99 -DTWOD -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.np2.ip.threads.2d -DINPLACE
#/usr/local/cuda-7.0/bin/nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.cuda.2d -DHOSTMEM -DINPLACE

# pow 2, in place
gcc -std=c99 -DTWOD -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.p2.ip.serial.2d  -DINPLACE -DPOW2
gcc -std=c99 -DTWOD -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.p2.ip.threads.2d -DINPLACE -DPOW2
/usr/local/cuda-7.0/bin/nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.cuda.2d -DHOSTMEM -DINPLACE -DPOW2

# device memory only
/usr/local/cuda-7.0/bin/nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.dev.cuda.2d  -DINPLACE -DPOW2
# /usr/local/cuda-7.0/bin/nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.dev.cuda.2d  -DINPLACE
/usr/local/cuda-7.0/bin/nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.dev.cuda.2d  -DPOW2
# /usr/local/cuda-7.0/bin/nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.dev.cuda.2d

### three dimensional

# non pow 2, out of place
#gcc -std=c99 -DTHREED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.np2.oop.serial.3d
#gcc -std=c99 -DTHREED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.np2.oop.threads.3d
#/usr/local/cuda-7.0/bin/nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.cuda.3d -DHOSTMEM

# pow 2, out of place
gcc -std=c99 -DTHREED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.p2.oop.serial.3d  -DPOW2
gcc -std=c99 -DTHREED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.p2.oop.threads.3d -DPOW2
/usr/local/cuda-7.0/bin/nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.cuda.3d -DHOSTMEM -DPOW2

# non pow 2, in place
#gcc -std=c99 -DTHREED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.np2.ip.serial.3d  -DINPLACE
#gcc -std=c99 -DTHREED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.np2.ip.threads.3d -DINPLACE
#/usr/local/cuda-7.0/bin/nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.cuda.3d -DHOSTMEM -DINPLACE

# pow 2, in place
gcc -std=c99 -DTHREED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.p2.ip.serial.3d  -DINPLACE -DPOW2
gcc -std=c99 -DTHREED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.p2.ip.threads.3d -DINPLACE -DPOW2
/usr/local/cuda-7.0/bin/nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.cuda.3d -DHOSTMEM -DINPLACE -DPOW2

# device memory only
/usr/local/cuda-7.0/bin/nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.dev.cuda.3d  -DINPLACE -DPOW2
# /usr/local/cuda-7.0/bin/nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.dev.cuda.3d  -DINPLACE
/usr/local/cuda-7.0/bin/nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.dev.cuda.3d  -DPOW2
# /usr/local/cuda-7.0/bin/nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.dev.cuda.3d

#### end compileN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.np2.ip.threads.2d -DINPLACE
#/usr/local/cuda-7.0/bin/nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.cuda.2d -DHOSTMEM -DINPLACE

# pow 2, in place
gcc -std=c99 -DTWOD -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.p2.ip.serial.2d  -DINPLACE -DPOW2
gcc -std=c99 -DTWOD -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.p2.ip.threads.2d -DINPLACE -DPOW2
/usr/local/cuda-7.0/bin/nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.cuda.2d -DHOSTMEM -DINPLACE -DPOW2

# device memory only
/usr/local/cuda-7.0/bin/nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.dev.cuda.2d  -DINPLACE -DPOW2
# /usr/local/cuda-7.0/bin/nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.dev.cuda.2d  -DINPLACE
/usr/local/cuda-7.0/bin/nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.dev.cuda.2d  -DPOW2
# /usr/local/cuda-7.0/bin/nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.dev.cuda.2d

### three dimensional

# non pow 2, out of place
#gcc -std=c99 -DTHREED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.np2.oop.serial.3d
#gcc -std=c99 -DTHREED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.np2.oop.threads.3d
#/usr/local/cuda-7.0/bin/nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.cuda.3d -DHOSTMEM

# pow 2, out of place
gcc -std=c99 -DTHREED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.p2.oop.serial.3d  -DPOW2
gcc -std=c99 -DTHREED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.p2.oop.threads.3d -DPOW2
/usr/local/cuda-7.0/bin/nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.cuda.3d -DHOSTMEM -DPOW2

# non pow 2, in place
#gcc -std=c99 -DTHREED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.np2.ip.serial.3d  -DINPLACE
#gcc -std=c99 -DTHREED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.np2.ip.threads.3d -DINPLACE
#/usr/local/cuda-7.0/bin/nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.cuda.3d -DHOSTMEM -DINPLACE

# pow 2, in place
gcc -std=c99 -DTHREED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3 -lm -o bench.p2.ip.serial.3d  -DINPLACE -DPOW2
gcc -std=c99 -DTHREED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -L /usr/lib/x86_64-linux-gnu/ -lfftw3_threads -lfftw3 -lpthread -lm -o bench.p2.ip.threads.3d -DINPLACE -DPOW2
/usr/local/cuda-7.0/bin/nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.cuda.3d -DHOSTMEM -DINPLACE -DPOW2

# device memory only
/usr/local/cuda-7.0/bin/nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.dev.cuda.3d  -DINPLACE -DPOW2
# /usr/local/cuda-7.0/bin/nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.dev.cuda.3d  -DINPLACE
/usr/local/cuda-7.0/bin/nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.dev.cuda.3d  -DPOW2
# /usr/local/cuda-7.0/bin/nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -arch=sm_52 -I /usr/local/cuda/include -I /usr/local/cuda-7.0/samples//common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.dev.cuda.3d

#### end compile