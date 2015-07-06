#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas.h>
#include <cufft.h>
#include <cudaProfiler.h>

#include "cuda_dft.cuh"


__global__ void cudaGauss_1d(cufftDoubleComplex* data, const uint64_t N) {
  // get the index of thread
  int ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  // allocate constants in shared memory
  const double x0 = (-5*SIGMA);
  const double dx = (10*SIGMA)/((double) N);
  
  if (ii < N)
    data[ii] = make_cuDoubleComplex( INV_SQRT_2PI*exp(-(x0 + ii*dx)*(x0 + ii*dx)/2/SIGMA)/SIGMA, 0.);//GAUSSIAN
}


void cuda_dft_1d(const uint64_t N, FILE** array_timing) {
   
  // initilizing files to save data
  const uint8_t filename_str_lenght = 128;
  const uint8_t dim = 1;
  
  // co jest??
  double  *data_dev, *data_host; // UWAGA - SMIESZNIE SIE ALOKUJE PAMIEC!!!
  
  char filename1d[filename_str_lenght];
  sprintf(filename1d,"./data/cufft_%dd_N%lu.bin",dim,N );
  printf("1d cufft example save in: %s\n",filename1d);
  FILE *file1d = fopen(filename1d, "wb");
  if (file1d == NULL)
  {
      printf("Error opening file %s!\n",filename1d);
      exit(EXIT_FAILURE);
  }
  
  checkCudaErrors( cudaMalloc( (void**)&data_dev,  sizeof(cufftDoubleComplex)*N ) );
  checkCudaErrors( cudaHostAlloc( (void**)&data_host, sizeof(cufftDoubleComplex)*N ,cudaHostAllocDefault) );
  
  // fill array
  int threadsPerBlock = 512;
  dim3 dimGrid( (N + threadsPerBlock - 1)/threadsPerBlock, 1, 1 ); // (numElements + threadsPerBlock - 1) / threadsPerBlock
  dim3 dimBlock(threadsPerBlock,1,1);
  checkCudaErrors( cudaGauss_1d<< dimGrid,dimBlock >>(data_dev) );
  
  checkCudaErrors( cudaMemcpy(data_host, data_dev, N*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost) );
  cudaDeviceSynchronize();
  for (uint16_t ii = 0; ii < N; ii++) {
    //fprintf(file1d,"",creal(data_host[ii]),cimag(data_host[ii]), sizeof(cufftDoubleComplex));
    fwrite(data_host+2*ii, sizeof(cufftDoubleComplex),1,file1d);
  }
  
  
  /* Create   a  1D  FFT  plan. */
  //cufftSetCompatibilityMode(CUFFT_COMPATIBILITY_NATIVE);
  //cufftHandle  plan; 
  //cufftPlan1d(&plan, NX, CUFFT Z2Z,1); // we want to make transform of only one 1D array
  
  /* Use  the  CUFFT  plan  to  transform  the  signal  in place. */ 
  //HANDLE_ERROR( cufftExecZ2Z(plan, data_dev, data_dev, CUFFT FORWARD) ); 
  
  
  
  
  //  cleaning up the mesh
  //HANDLE_ERROR( cufftDestroy(plan) ); 
  checkCudaErrors( cudaFree(data_dev) );
  checkCudaErrors( cudaFreeHost(data_host) );
  
}

/*
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