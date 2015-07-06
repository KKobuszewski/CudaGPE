#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include "kernels.h"

#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))



int main(void)
{
  int N = 1<<27; // max int: 2**31-1
  double *x, *y, *x_copy, *y_copy;
  
  printf("\n");
  printf(" * ***************************************** *\n");
  printf(" *                                           *\n");
  printf(" *              Test CUDA vs C               *\n");
  printf(" *                                           *\n");
  printf(" * ***************************************** *\n");
  printf("All operations are made on two arrays of type double and size %d\n",N);
  printf("Operation is adding a vector scaled by constant to another vector\n\n");
  printf("CPU\n");
  
  // CPU MEM
  clock_t start = clock();
  
  x = (double*)malloc(N*sizeof(double));
  y = (double*)malloc(N*sizeof(double));
  
  
  
  clock_t cpu_mem_t = clock() - start;
  printf("\nMemory allocation time on HOST:\t\t\t\t %lf s\n", (double) cpu_mem_t/((double)CLOCKS_PER_SEC));
  
  // copying array on host
  //clock_t cpu_copy_t = clock();
  
  x_copy = (double*)malloc(N*sizeof(double));
  y_copy = (double*)malloc(N*sizeof(double));
  for (int i = 0; i < N; i++) {
    x_copy[i] = x[i];
    y_copy[i] = y[i];
  }
  
  //clock_t temp_t = clock() - cpu_copy_t;
  //printf("\nCopying array on HOST:\t\t\t\t\t %lf s\n", (double) (temp_t)/((double)CLOCKS_PER_SEC));
  //printf("Copying + memalloc on HOST:\t\t\t\t %lf s\n", (double) (cpu_mem_t + temp_t)/((double)CLOCKS_PER_SEC));
  
  
  // CPU COMPUTING
  start = clock();
  for (int ii=0; ii<FOR_LOOPS; ii++) { // FOR_LOOPS defined in kernels.h
    int N_copy = N;
    while(N_copy) {
      N_copy--;
      y_copy[N_copy] += 2.0*x_copy[N_copy];
    }
  }
  clock_t cpu_perf_t = clock() - start;
  printf("\nCPU operation time:\t\t\t\t\t %lf s\n", cpu_perf_t/((double)CLOCKS_PER_SEC));
  
  
  printf("%d\n",N);
  // pass controll to GPU device
  perform_cuda_kernel(N, 2.0, x, y);
  
  
  
  
  // CHECK ERROS IN COMPUTATION
  double maxError = 0.0f;
  uint64_t numError = 0.;
  for (int i = 0; i < N; i++) {
    maxError = MAX(maxError, abs(y[i]-y_copy[i]));
    if (y[i]-y_copy[i]>1e-14) numError++;
  }
  
  
  // FREE MEM ON HOST
  start = clock();
  
  free(x);
  free(y);
    
  clock_t cpu_free_t = clock() - start;
  printf("Freeing memory on HOST:\t\t\t\t\t %lf s\n", cpu_free_t/((double)CLOCKS_PER_SEC));
  
  free(x_copy);
  free(y_copy);
  
  
  printf("\n\nMax error of GPU operations: %lf \n", maxError);
  printf("Number of errors: %lu \n\n", numError);
  return 0;
}
