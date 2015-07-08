#ifndef __CUDAUTILS_H__
#define __CUDAUTILS_H__

//#include <stdio.h>
//#include <cuda.h>

// !!! these macros need cudaEvent_t start, stop; to be defined globally!!!
#define CUDATIMEIT_START \
			  cudaEventCreate(&start_t); \
			  cudaEventCreate(&stop_t); \
			  cudaEventRecord(start_t,0)

#define CUDATIMEIT_STOP \
			  cudaEventRecord(stop_t,0); \
			  cudaEventSynchronize(stop_t)

double print_cudatimeit(const char* message);
double fprint_cudatimeit(FILE* file);
//void cudaCheckErrors(cudaError_t err,const char* action);

#endif