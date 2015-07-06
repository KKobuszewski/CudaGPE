#ifndef __KERNELS_H__
#define __KERNELS_H__

#define FOR_LOOPS 1

// util functions
void cudaCheckErrors(cudaError_t err,const char* action);

// calling cuda actions
void perform_cuda_kernel(const int N, const double a, double *x, double *y);

#endif