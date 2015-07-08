#include <stdio.h>
#include <cuda.h>
#include <time.h>

#include "kernels.h"

#define BLOCK_SIZE 1024 // max 1024 <- it can be 3d like 1024x1x1, 512x2x1, 256x2x2 etc. etc.
#define GRID_SIZE 1

/*
 * compile: nvcc -o prog saxpy.cu
 */

/*
 * Simple kernel adding a vector scaled by constant to another vector (overwriting second vec)
 */
__global__
void saxpy(int n, const double a, double *x, double *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] += a*x[i];
  
}

/*
 * Funtion to checking errors from CUDA functions.
 * cudaError_t err - structure with error code, cudaSuccess means no error
 * char* action - description of action (to be easier to find in code)
 */
void cudaCheckErrors(cudaError_t err,const char* action){
  if (err != cudaSuccess)
  {
        fprintf(stderr, "Failed to:<< %s >>(error code %s)!\n", action, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
  }
}

/*
 * This function provides interface to CUDA
 * return status of action (if there was a crash)
 */
void perform_cuda_kernel(int N, const double a, double *x, double *y) {
  
  printf("\n\n\nGPU\n");
  
  double *x_dev, *y_dev;
  
  // GPU MEM
  clock_t temp_t = clock();
  clock_t start = clock();
  cudaMalloc(&x_dev, N*sizeof(double)); 
  cudaMalloc(&y_dev, N*sizeof(double));
  clock_t gpu_mem_t = clock() - start;
  
  start = clock();
  cudaMemcpy(x_dev, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_dev, y, N*sizeof(double), cudaMemcpyHostToDevice);
  clock_t gpu_cpy_t = clock() - start;
  printf("\nMemory allocation on DEVICE:\t\t\t\t %lf s\n", gpu_mem_t/((double)CLOCKS_PER_SEC));
  printf("Memory copying time on DEVICE:\t\t\t\t %lf s\n", gpu_cpy_t/((double)CLOCKS_PER_SEC));
  printf("Copying + memalloc on DEVICE:\t\t\t\t %lf s\n", (gpu_cpy_t+gpu_mem_t)/((double)CLOCKS_PER_SEC));
  
  
  
  
  
  // GPU COMPUTING
  printf("\n<<%d, %d>>, N=%d\n",GRID_SIZE, BLOCK_SIZE,N);
  
  // for timing purposes
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  // start timer
  cudaEventRecord(start_event,0);
  
  //saxpy<<<(N+255)/256, 256>>>(N, a, x_dev, y_dev); <- tu cos nie dziala tak jak trzeba
  for (int ii = 0; ii<FOR_LOOPS; ii++) { // FOR_LOOPS defined in kernels.h
    saxpy<<<GRID_SIZE, BLOCK_SIZE>>>(N, a, x_dev, y_dev);
    cudaDeviceSynchronize();
  }
  
  cudaEventRecord(stop_event,0);
  cudaEventSynchronize(stop_event);
  float computationTime;
  cudaEventElapsedTime(&computationTime, start_event, stop_event);
  
  cudaCheckErrors(cudaGetLastError(),"cuda kernel");
  
  
  printf("\nGPU operation time:\t\t\t\t\t %f s\n", computationTime/1000.0);
  
  
  
  
  // copying data back to host
  start = clock();
  
  cudaCheckErrors(
    cudaMemcpy(y, y_dev, N*sizeof(double), cudaMemcpyDeviceToHost),
  "copying data back to host");
  
  clock_t gpu_copyback_t = clock() - start;
  printf("\nCopying data from GPU back to host:\t\t\t %lf s\n", gpu_copyback_t/((double)CLOCKS_PER_SEC));
  
  
  
  
  // FREE MEM ON DEVICE
  start = clock();
  
  cudaCheckErrors(	cudaFree(x_dev)	,"free x_dev vec");
  cudaCheckErrors(	cudaFree(y_dev)	,"free y_dev vec");
  
  clock_t gpu_cufree_t = clock() - start;
  printf("\nFreeing memory on DEVICE:\t\t\t\t %lf s\n", gpu_cufree_t/((double)CLOCKS_PER_SEC));
  //printf("\nTotal time on DEVICE:\t\t\t\t %lf s\n", (clock() - temp_t)/((double)CLOCKS_PER_SEC));
  
}