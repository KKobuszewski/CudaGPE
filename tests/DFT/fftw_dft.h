
#ifndef __FFTW_DFT_H__
#define __FFTW_DFT_H__

#include <fftw3.h>

// macro constants
//#define M_PI (3.14159265358979323846)
#define M_2PI (6.283185307179586)
#define SQRT_2PI (2.5066282746310002)
#define INV_SQRT_2PI (0.3989422804014327)
#define SIGMA (1)

// functions
void perform_fftw3_1d(const uint64_t N, FILE** array_timing);

#endif