#!/bin/bash
export NFFTS="10"

# non pow 2, out of place
gcc -DONED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f -lm -o bench.np2.oop.serial  
gcc -DONED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f_threads -lfftw3f -lpthread -lm -o bench.np2.oop.threads 
nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.cuda -DHOSTMEM 

# pow 2, out of place
gcc -DONED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f -lm -o bench.p2.oop.serial  -DPOW2 
gcc -DONED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f_threads -lfftw3f -lpthread -lm -o bench.p2.oop.threads -DPOW2 
nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.cuda -DHOSTMEM -DPOW2 

# non pow 2, in place
gcc -DONED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f -lm -o bench.np2.ip.serial  -DINPLACE 
gcc -DONED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f_threads -lfftw3f -lpthread -lm -o bench.np2.ip.threads -DINPLACE 
nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.cuda -DHOSTMEM -DINPLACE 

# pow 2, in place
gcc -DONED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f -lm -o bench.p2.ip.serial  -DINPLACE -DPOW2 
gcc -DONED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f_threads -lfftw3f -lpthread -lm -o bench.p2.ip.threads -DINPLACE -DPOW2 
nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.cuda -DHOSTMEM -DINPLACE -DPOW2 

# device memory only
nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.dev.cuda  -DINPLACE -DPOW2 
nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.dev.cuda  -DINPLACE 
nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.dev.cuda  -DPOW2
nvcc -DONED -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.dev.cuda 

## two dimensional

# non pow 2, out of place
gcc -DTWOD -DN_FFTS=${NFFTS} timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f -lm -o bench.np2.oop.serial.2d
gcc -DTWOD -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f_threads -lfftw3f -lpthread -lm -o bench.np2.oop.threads.2d
nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.cuda.2d -DHOSTMEM

# pow 2, out of place
gcc -DTWOD -DN_FFTS=${NFFTS} timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f -lm -o bench.p2.oop.serial.2d  -DPOW2
gcc -DTWOD -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f_threads -lfftw3f -lpthread -lm -o bench.p2.oop.threads.2d -DPOW2
nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.cuda.2d -DHOSTMEM -DPOW2

# non pow 2, in place
gcc -DTWOD -DN_FFTS=${NFFTS} timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f -lm -o bench.np2.ip.serial.2d  -DINPLACE
gcc -DTWOD -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f_threads -lfftw3f -lpthread -lm -o bench.np2.ip.threads.2d -DINPLACE
nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.cuda.2d -DHOSTMEM -DINPLACE

# pow 2, in place
gcc -DTWOD -DN_FFTS=${NFFTS} timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f -lm -o bench.p2.ip.serial.2d  -DINPLACE -DPOW2
gcc -DTWOD -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f_threads -lfftw3f -lpthread -lm -o bench.p2.ip.threads.2d -DINPLACE -DPOW2
nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.cuda.2d -DHOSTMEM -DINPLACE -DPOW2

# device memory only
nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.dev.cuda.2d  -DINPLACE -DPOW2
nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.dev.cuda.2d  -DINPLACE
nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.dev.cuda.2d  -DPOW2
nvcc -DTWOD -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.dev.cuda.2d

### three dimensional

# non pow 2, out of place
gcc -DTHREED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f -lm -o bench.np2.oop.serial.3d
gcc -DTHREED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f_threads -lfftw3f -lpthread -lm -o bench.np2.oop.threads.3d
nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.cuda.3d -DHOSTMEM

# pow 2, out of place
gcc -DTHREED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f -lm -o bench.p2.oop.serial.3d  -DPOW2
gcc -DTHREED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f_threads -lfftw3f -lpthread -lm -o bench.p2.oop.threads.3d -DPOW2
nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.cuda.3d -DHOSTMEM -DPOW2

# non pow 2, in place
gcc -DTHREED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f -lm -o bench.np2.ip.serial.3d  -DINPLACE
gcc -DTHREED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f_threads -lfftw3f -lpthread -lm -o bench.np2.ip.threads.3d -DINPLACE
nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.cuda.3d -DHOSTMEM -DINPLACE

# pow 2, in place
gcc -DTHREED -DN_FFTS=${NFFTS} timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f -lm -o bench.p2.ip.serial.3d  -DINPLACE -DPOW2
gcc -DTHREED -DN_FFTS=${NFFTS} -DTHREADED timer.c fftw_bench.c -I ~/lib/fftw-3.1.2_sse/include/ -L ~/lib/fftw-3.1.2_sse/lib/ -lfftw3f_threads -lfftw3f -lpthread -lm -o bench.p2.ip.threads.3d -DINPLACE -DPOW2
nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.cuda.3d -DHOSTMEM -DINPLACE -DPOW2

# device memory only
nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.ip.dev.cuda.3d  -DINPLACE -DPOW2
nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.ip.dev.cuda.3d  -DINPLACE
nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.p2.oop.dev.cuda.3d  -DPOW2
nvcc -DTHREED -DN_FFTS=${NFFTS} -DCUDA -I /usr/local/cuda/include -I ~/NVIDIA_CUDA_SDK/common/inc -L /usr/local/cuda/lib fftw_bench.cu timer.cu -lcufft -o bench.np2.oop.dev.cuda.3d

#### end compile

unset NFFTS
echo "./bench.np2.oop.serial"
./bench.np2.oop.serial > out/bench.np2.oop.serial
echo "./bench.np2.oop.threads"
./bench.np2.oop.threads > out/bench.np2.oop.threads
echo "./bench.np2.oop.cuda"
./bench.np2.oop.cuda > out/bench.np2.oop.cuda

echo "./bench.p2.oop.serial"
./bench.p2.oop.serial > out/bench.p2.oop.serial
echo "./bench.p2.oop.threads"
./bench.p2.oop.threads > out/bench.p2.oop.threads
echo "./bench.p2.oop.cuda"
./bench.p2.oop.cuda > out/bench.p2.oop.cuda

echo "./bench.np2.ip.serial"
./bench.np2.ip.serial > out/bench.np2.ip.serial
echo "./bench.np2.ip.threads"
./bench.np2.ip.threads > out/bench.np2.ip.threads
echo "./bench.np2.ip.cuda"
./bench.np2.ip.cuda > out/bench.np2.ip.cuda

echo "./bench.p2.ip.serial"
./bench.p2.ip.serial > out/bench.p2.ip.serial
echo "./bench.p2.ip.threads"
./bench.p2.ip.threads > out/bench.p2.ip.threads
echo "./bench.p2.ip.cuda"
./bench.p2.ip.cuda > out/bench.p2.ip.cuda

echo "./bench.p2.ip.dev.cuda"
./bench.p2.ip.dev.cuda > out/bench.p2.ip.dev.cuda
echo "./bench.np2.ip.dev.cuda"
./bench.np2.ip.dev.cuda > out/bench.np2.ip.dev.cuda
echo "./bench.p2.oop.dev.cuda"
./bench.p2.oop.dev.cuda > out/bench.p2.oop.dev.cuda
echo "./bench.np2.oop.dev.cuda"
./bench.np2.oop.dev.cuda > out/bench.np2.oop.dev.cuda

#TWOD

echo "./bench.np2.oop.serial.2d"
./bench.np2.oop.serial.2d > out/bench.np2.oop.serial.2d
echo "./bench.np2.oop.threads.2d"
./bench.np2.oop.threads.2d > out/bench.np2.oop.threads.2d
echo "./bench.np2.oop.cuda.2d"
./bench.np2.oop.cuda.2d > out/bench.np2.oop.cuda.2d

echo "./bench.p2.oop.serial.2d"
./bench.p2.oop.serial.2d > out/bench.p2.oop.serial.2d
echo "./bench.p2.oop.threads.2d"
./bench.p2.oop.threads.2d > out/bench.p2.oop.threads.2d
echo "./bench.p2.oop.cuda.2d"
./bench.p2.oop.cuda.2d > out/bench.p2.oop.cuda.2d

echo "./bench.np2.ip.serial.2d"
./bench.np2.ip.serial.2d > out/bench.np2.ip.serial.2d
echo "./bench.np2.ip.threads.2d"
./bench.np2.ip.threads.2d > out/bench.np2.ip.threads.2d
echo "./bench.np2.ip.cuda.2d"
./bench.np2.ip.cuda.2d > out/bench.np2.ip.cuda.2d

echo "./bench.p2.ip.serial.2d"
./bench.p2.ip.serial.2d > out/bench.p2.ip.serial.2d
echo "./bench.p2.ip.threads.2d"
./bench.p2.ip.threads.2d > out/bench.p2.ip.threads.2d
echo "./bench.p2.ip.cuda.2d"
./bench.p2.ip.cuda.2d > out/bench.p2.ip.cuda.2d

echo "./bench.p2.ip.dev.cuda.2d"
./bench.p2.ip.dev.cuda.2d > out/bench.p2.ip.dev.cuda.2d
echo "./bench.np2.ip.dev.cuda.2d"
./bench.np2.ip.dev.cuda.2d > out/bench.np2.ip.dev.cuda.2d
echo "./bench.p2.oop.dev.cuda.2d"
./bench.p2.oop.dev.cuda.2d > out/bench.p2.oop.dev.cuda.2d
echo "./bench.np2.oop.dev.cuda.2d"
./bench.np2.oop.dev.cuda.2d > out/bench.np2.oop.dev.cuda.2d

#THREED

echo "./bench.np2.oop.serial.3d"
./bench.np2.oop.serial.3d > out/bench.np2.oop.serial.3d
echo "./bench.np2.oop.threads.3d"
./bench.np2.oop.threads.3d > out/bench.np2.oop.threads.3d
echo "./bench.np2.oop.cuda"
./bench.np2.oop.cuda.3d > out/bench.np2.oop.cuda.3d

echo "./bench.p2.oop.serial.3d"
./bench.p2.oop.serial.3d > out/bench.p2.oop.serial.3d
echo "./bench.p2.oop.threads.3d"
./bench.p2.oop.threads.3d > out/bench.p2.oop.threads.3d
echo "./bench.p2.oop.cuda.3d"
./bench.p2.oop.cuda.3d > out/bench.p2.oop.cuda.3d

echo "./bench.np2.ip.serial.3d"
./bench.np2.ip.serial.3d > out/bench.np2.ip.serial.3d
echo "./bench.np2.ip.threads.3d"
./bench.np2.ip.threads.3d > out/bench.np2.ip.threads.3d
echo "./bench.np2.ip.cuda.3d"
./bench.np2.ip.cuda.3d > out/bench.np2.ip.cuda.3d

echo "./bench.p2.ip.serial.3d"
./bench.p2.ip.serial.3d > out/bench.p2.ip.serial.3d
echo "./bench.p2.ip.threads.3d"
./bench.p2.ip.threads.3d > out/bench.p2.ip.threads.3d
echo "./bench.p2.ip.cuda.3d"
./bench.p2.ip.cuda.3d > out/bench.p2.ip.cuda.3d

echo "./bench.p2.ip.dev.cuda.3d"
./bench.p2.ip.dev.cuda.3d > out/bench.p2.ip.dev.cuda.3d
echo "./bench.np2.ip.dev.cuda.3d"
./bench.np2.ip.dev.cuda.3d > out/bench.np2.ip.dev.cuda.3d
echo "./bench.p2.oop.dev.cuda.3d"
./bench.p2.oop.dev.cuda.3d > out/bench.p2.oop.dev.cuda.3d
echo "./bench.np2.oop.dev.cuda.3d"
./bench.np2.oop.dev.cuda.3d > out/bench.np2.oop.dev.cuda.3d
