#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include "kernels.h"

//kernels
__global__ void saxpy(int n, const double a, double *x, double *y);


#endif