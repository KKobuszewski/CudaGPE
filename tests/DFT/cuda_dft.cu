#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuComplex.h>
#include <driver_types.h>

#include "cuda_dft.cuh"
#include "book.h"

#include "cudautils.cuh"

// timing
cudaEvent_t start;
cudaEvent_t stop;


__global__ void cudaGauss_1d(cufftDoubleComplex* data, const unsigned long long N) {
  // get the index of thread
  unsigned long long ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  // allocate constants in shared memory
  const double x0 = (-5*SIGMA);
  const double dx = (10*SIGMA)/((double) N);
  
  if (ii < N) {
    data[ii] = make_cuDoubleComplex( INV_SQRT_2PI*exp(-(x0 + ii*dx)*(x0 + ii*dx)/2/SIGMA)/SIGMA, 0. );
  }
  
  __syncthreads();
  //printf("Kernel sie wykonuje\n");
}


void perform_cufft_1d(const uint64_t N, FILE** array_timing) {
  
  // initilizing files to save data
  const uint8_t filename_str_lenght = 128;
  const uint8_t dim = 1;
    
  char filename1d[filename_str_lenght];
  FILE *file1d;
  
  sprintf(filename1d,"./data/cufft_%dd_N%lu.bin",dim,N );
  printf("1d cufft example save in: %s\n",filename1d);
  file1d = fopen(filename1d, "wb");
  if (file1d == NULL)
  {
      printf("Error opening file %s!\n",filename1d);
      exit(EXIT_FAILURE);
  }
  
  
  
  
  // allocate memory
  cufftDoubleComplex *data_dev;
  cufftDoubleComplex *data_host;
  printf("sizeof(cufftDoubleComplex): %lu\n", sizeof(cufftDoubleComplex));
  printf("memory: %lu kB\n", sizeof(cufftDoubleComplex)*N/1024);
  HANDLE_ERROR( cudaMalloc((void**) &data_dev, sizeof(cufftDoubleComplex)*N) );
  if (N < 65536) {
    HANDLE_ERROR( cudaHostAlloc((void**) &data_host, sizeof(cufftDoubleComplex)*N, cudaHostAllocDefault) ); // when to use pinned memory: http://www.cs.virginia.edu/~mwb7w/cuda_support/pinned_tradeoff.html
  }
  else {
    data_host = (cufftDoubleComplex*) malloc(sizeof(cufftDoubleComplex)*N);
  }
  
  
  // fill array
  int threadsPerBlock = 512;
  printf("%lu\n",(N + threadsPerBlock - 1)/threadsPerBlock);
  dim3 dimGrid( (N + threadsPerBlock - 1)/threadsPerBlock, 1, 1 ); // (numElements + threadsPerBlock - 1) / threadsPerBlock
  dim3 dimBlock(threadsPerBlock,1,1);
  cudaGauss_1d<<<dimGrid,dimBlock>>>(data_dev, N);
  HANDLE_ERROR( cudaGetLastError() );
  
  
  HANDLE_ERROR( cudaMemcpy(data_host, data_dev, N*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost) );
  HANDLE_ERROR( cudaDeviceSynchronize() );
  
  if (N < 65536) {
  for (uint16_t ii = 0; ii < N; ii++) {
    fwrite(data_host+ii, sizeof(cuDoubleComplex),1,file1d);
  }
  }
  
  cufftHandle plan_forward;
  CUDATIMEIT_START;
  cufftPlan1d(&plan_forward, N, CUFFT_Z2Z, 1); // N - samples in array, 1 - number of arrays, must be splitted
  //cufftPlan1d(&plan_forward, (N <= 512) ? N : 512, CUFFT_Z2Z, (N <= 512) ? 1 : N/512); Maybe there is more efficient way ???
  CUDATIMEIT_STOP;
  fprint_cudatimeit(array_timing[1]);
  
  // inplace
  CUDATIMEIT_START;
  if (cufftExecZ2Z(plan_forward, data_dev, data_dev, CUFFT_FORWARD) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed!\n");
    exit( EXIT_FAILURE );
  }
  CUDATIMEIT_STOP;
  fprint_cudatimeit(array_timing[1]);
  
  
  HANDLE_ERROR( cudaMemcpy(data_host, data_dev, N*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost) );
  HANDLE_ERROR( cudaDeviceSynchronize() );
  
  for (uint16_t ii = 0; ii < N; ii++) {
    fwrite(data_host+ii, sizeof(cuDoubleComplex),1,file1d);
  }
  
  //  cleaning up the mesh
  HANDLE_ERROR( cudaFree(data_dev) );
  HANDLE_ERROR( cudaFreeHost(data_host) );
  
}

/*
// 


#define NX 64
#define NY 64
#define NZ 128

cufftHandle plan;
cufftComplex *data1, *data2;
cudaMalloc((void**)&data1, sizeof(cufftComplex)*NX*NY*NZ);
cudaMalloc((void**)&data2, sizeof(cufftComplex)*NX*NY*NZ);
// Create a 3D FFT plan. 
cufftPlan3d(&plan, NX, NY, NZ, CUFFT_C2C);

// Transform the first signal in place.
cufftExecC2C(plan, data1, data1, CUFFT_FORWARD);

// Transform the second signal using the same plan.
cufftExecC2C(plan, data2, data2, CUFFT_FORWARD);

// Destroy the cuFFT plan.
cufftDestroy(plan);
cudaFree(data1); cudaFree(data2);
*/
