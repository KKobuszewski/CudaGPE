#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#include <cuda_runtime.h>

#include "cpu_matrix_mul.h"

#define TIMEIT_START (start_t=clock())
#define TIMEIT_END(x) x = (end_t=clock()-start_t)



void print_timeit(clock_t time_to_print, const char* message){
  printf( "time of %-60s %lf s\n", message, time_to_print/((double)CLOCKS_PER_SEC) );
}


int main(){
  
  const uint64_t N = 1000;
  clock_t start_t = 0;
  clock_t end_t = 0;
  
  double *A_matrix, *B_matrix, *A_matrix_dev, *B_matrix_dev, *result_cpu, *result_gpu;
  
  TIMEIT_START;
  // allocate memmory on HOST
  A_matrix = (double*)malloc(N*N*sizeof(double));
  B_matrix = (double*)malloc(N*N*sizeof(double));
  result_cpu = (double*)malloc(N*N*sizeof(double));
  result_gpu = (double*)malloc(N*N*sizeof(double));
  TIMEIT_END(clock_t host_alloc_t);
  print_timeit(host_alloc_t,"allocating memory on host:");
  
  fill_matrix(A_matrix,N);
  fill_matrix(B_matrix,N);
  //print_matrix(A_matrix,N);
  printf("\n");
  
  TIMEIT_START;
  mult_matrix(A_matrix,B_matrix,result_cpu,N);
  TIMEIT_END(clock_t cpu_mult_t);
  print_timeit(cpu_mult_t,"multiplying matrix by a matrix with into another one:");
  //print_matrix(result_cpu,N);
  
  TIMEIT_START;
  cudaMalloc((void **) &A_matrix_dev,N*N*sizeof(double));
  cudaMalloc((void **) &B_matrix_dev,N*N*sizeof(double));
  cudaMalloc((void **) &result_gpu,N*N*sizeof(double));
  TIMEIT_END(clock_t gpu_alloc_t);
  print_timeit(gpu_alloc_t,"allocation memory on device:");
  
  TIMEIT_START;
  cudaMemcpy(A_matrix_dev, A_matrix, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(B_matrix_dev, B_matrix, N*N*sizeof(double), cudaMemcpyHostToDevice);
  TIMEIT_END(clock_t gpu_copy_t);
  print_timeit(gpu_copy_t,"allocation memory on device:");
  
  free(A_matrix);
  free(B_matrix);
  free(result_cpu);
  free(result_gpu);
  
  return EXIT_SUCCESS;
}