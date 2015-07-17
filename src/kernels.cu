#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <math.h>
#include <complex.h>
#include <cuda.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cufftXt.h>

#include "global.h"
#include "kernels.cuh"


// tests with callbacks
#define M_2PI (6.283185307179586)
#define SQRT_2PI (2.5066282746310002)
#define INV_SQRT_2PI (0.3989422804014327)
#define SIGMA (1)

static __device__ cufftDoubleComplex cufftSgn(void *dataIn, 
					      size_t offset, 
					      void *callerInfo, 
					      void *sharedPtr) 
{
    if (offset < (NX*NY*NZ)/2)
      return make_cuDoubleComplex(-1.,0.);
    else
      return make_cuDoubleComplex(1.,0.);
}

static __device__ cufftDoubleComplex cudaGauss_1d(void *dataIn, 
						  size_t offset, 
						  void *callerInfo, 
						  void *sharedPtr) 
{
  // allocate constants in shared memory <- how to do that???
  const double x0 = (-5*SIGMA);
  const double dx = (10*SIGMA)/((double) NX*NY*NZ);
  return make_cuDoubleComplex( INV_SQRT_2PI*exp(-(x0 + offset*dx)*(x0 + offset*dx)/2/SIGMA)/SIGMA, 0. );
}


__global__ void cudaGauss_1d(cufftDoubleComplex* data, const uint64_t N) {
  // get the index of thread
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  // allocate constants in shared memory
  const double x0 = (-5*SIGMA);
  const double dx = (10*SIGMA)/((double) N);
  
  if (ii < N) {
    data[ii] = make_cuDoubleComplex( INV_SQRT_2PI*exp(-(x0 + ii*dx)*(x0 + ii*dx)/2/SIGMA)/SIGMA, 0. );
  }
  
  __syncthreads();
  //printf("Kernel sie wykonuje\n");
}








// util functions
//__global__ 


// cross sections of wavefunction
__global__ void get_cross_sectionX();
__global__ void get_cross_sectionY();
__global__ void get_cross_sectionZ();
__global__ void get_cross_sectionXY();
__global__ void get_cross_sectionXZ();
__global__ void get_cross_sectionYZ();


// integrals
__global__ void mean_kinetic_energy();
__global__ void mean_potential_energy();
__global__ void mean_contact_interaction_energy();
__global__ void mean_dipolar_interaction_energy();
__global__ void mean_momentum();
__global__ void get_norm();
// angular momnetum???

// DEVICE ONLY FUNCTIONS

static __device__ void normalize();
//__device__



// callbacks
static __device__ cuDoubleComplex propagate_Vext();
static __device__ cuDoubleComplex propagate_T();
static __device__ cuDoubleComplex propagate_Vcon();
static __device__ cuDoubleComplex propagate_Vdip();

// pointer to callbacks' functions
__device__ cufftCallbackLoadZ CB_LD_MOMENTUM_SPACE_FORWARD();
__device__ cufftCallbackStoreZ CB_ST_MOMENTUM_SPACE_FORWARD();
__device__ cufftCallbackLoadZ CB_LD_MOMENTUM_SPACE_INVERSE();
__device__ cufftCallbackStoreZ CB_ST_MOMENTUM_SPACE_INVERSE();
__device__ cufftCallbackLoadD CB_LD_DIPOLAR_FORWARD();
__device__ cufftCallbackStoreZ CB_ST_DIPOLAR_FORWARD();
__device__ cufftCallbackLoadZ CB_LD_DIPOLAR_INVERSE();
__device__ cufftCallbackStoreZ CB_ST_DIPOLAR_INVERSE();