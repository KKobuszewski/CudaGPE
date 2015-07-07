
#ifndef __CUDA_DFT_CUH__
#define __CUDA_DFT_CUH__


#include "cuda_dft.h"

// macro constants
//#define M_PI (3.14159265358979323846)
#define M_2PI (6.283185307179586)
#define SQRT_2PI (2.5066282746310002)
#define INV_SQRT_2PI (0.3989422804014327)
#define SIGMA (1)


__global__ void cudaGauss_1d(cufftDoubleComplex* data, const unsigned long long N);

#endif