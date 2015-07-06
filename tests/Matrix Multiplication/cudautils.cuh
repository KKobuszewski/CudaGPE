#ifndef __CUDAUTILS_H__
#define __CUDAUTILS_H__

//#include <stdio.h>
//#include <cuda.h>


// !!! these macros need cudaEvent_t start, stop; to be defined !!!
#define CUDATIMEIT_START \
			  cudaEventCreate(&start); \
			  cudaEventCreate(&stop); \
			  cudaEventRecord(start,0)

#define CUDATIMEIT_STOP \
			  cudaEventRecord(stop,0); \
			  cudaEventSynchronize(stop)

double print_cudatimeit(const char* message);
void cudaCheckErrors(cudaError_t err,const char* action);

#endif