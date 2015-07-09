#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

#include "cudautils.h"



__global__ void cudaGauss_1d(cufftDoubleComplex* data, const uint64_t N) {
  // get the index of thread
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  // allocate constants in shared memory
  const double x0 = (-5*SIGMA);
  const double dx = (10*SIGMA)/((double) N);
  
  if (ii < N) {
    data[ii] = make_cuDoubleComplex( INV_SQRT_2PI*exp(-(x0 + ii*dx)*(x0 + ii*dx)/2/SIGMA)/SIGMA, 0. );
  }
  
  __syncthreads();
  //printf("Kernel sie wykonuje\n");
}


