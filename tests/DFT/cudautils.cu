#include <stdio.h>
#include <cuda.h>

#include "cudautils.cuh"


extern cudaEvent_t start;
extern cudaEvent_t stop;

double print_cudatimeit(const char* message) {
  float computationTime;
  cudaEventElapsedTime(&computationTime, start, stop);
  printf( "time of %-60s %lf s\n", message, (double) computationTime );
  return (double) computationTime;
}

double fprint_cudatimeit(FILE* file) {
  float computationTime;
  cudaEventElapsedTime(&computationTime, start, stop);
  fwrite(&computationTime, (size_t) sizeof(double), 1, file);
  return (double) computationTime;
}

// void cudaCheckErrors(cudaError_t err,const char* action){
//   if (cudaGetLastError() != cudaSuccess)
//   {
//         fprintf(stderr, "Failed to:<< %s >>(error code %s)!\n", action, cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//   }
// }