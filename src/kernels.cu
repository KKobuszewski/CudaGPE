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
#include "cuda_complex_ext.cuh"


// tests with callbacks
#define M_2PI (6.283185307179586)
#define SQRT_2PI (2.5066282746310002)
#define INV_SQRT_2PI (0.3989422804014327)
#define SIGMA (1)


/* ************************************************************************************************************************************* *
 * 																	 *
 * 							LOAD CALLBACKS									 *
 * 																	 *
 * ************************************************************************************************************************************* */

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

static __device__ cufftDoubleComplex cufftGauss_1d(void *dataIn, 
						  size_t offset, 
						  void *callerInfo, 
						  void *sharedPtr) 
{
  // allocate constants in shared memory <- how to do that???
  const double x0 = (-5*SIGMA);
  const double dx = (10*SIGMA)/((double) NX*NY*NZ);
  return make_cuDoubleComplex( INV_SQRT_2PI*exp(-(x0 + offset*dx)*(x0 + offset*dx)/2/SIGMA)/SIGMA, 0. );
}



/* ************************************************************************************************************************************* *
 * 																	 *
 * 							KERNELS TYPE Z									 *
 * 																	 *
 * ************************************************************************************************************************************* */


__global__ void ker_gauss_1d(cuDoubleComplex* data) {
  // get the index of thread
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  const uint64_t N = NX*NY*NZ;
  
  // allocate constants in shared memory
  const double x0 = (-5*SIGMA);
  const double dx = (10*SIGMA)/((double) N);
  
  if (ii < N) {
    data[ii] = make_cuDoubleComplex( INV_SQRT_2PI*exp(-(x0 + ii*dx)*(x0 + ii*dx)/2/SIGMA)/SIGMA, 0. );
  }
  
  __syncthreads();
  //printf("Kernel sie wykonuje\n");
}



/*
 * Divides the result of inverse cufft by number of samples (to get unitary form of DFT).
 * 
 * NOTE: cufftDoubleComplex is just typdef for cuDoubleComplex ! (no need to include <cuComplex.h> if only cufft necessary)
 * 	 here used to associate it with inverse cufft
 */
__global__ void ker_normalize(cufftDoubleComplex* cufft_inverse_data) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  // in both kernel as well as callback we use predefined N to have comparable performance results
  
  if (ii < NX*NY*NZ) {
    cufft_inverse_data[ii] = make_cuDoubleComplex( cuCreal(cufft_inverse_data[ii])/((double) N), cuCimag(cufft_inverse_data[ii])/((double) N) );
  }
}

// dla funckji roznych dla innych DIM mozna zrobic makro wybierajace odpowiednia funckja lub makro 'krojace funcje' na opcje w zaleznosci od wymiaru
__global__ void ker_create_propagator_T(cuDoubleComplex* propagator_T_dev) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  // tutaj mozna sporbowac uzyc shared memory na k !
  
  // CASE DIM 1D
#if (DIM == 1)
  if (ii < NX*NY*NZ/2) {
    // range [0, KMAX]
    const double kx_ii = DKx*ii;
    propagator_T_dev[ii] = make_cuDoubleComplex( cos(kx_ii*kx_ii*DT/2),-sin(kx_ii*kx_ii*DT/2)  ); // array of constants e^(-I*k^2/2*dt) = cos( -kx^2/2dt ) + I*sin( kx^2/2dt ) = cos( +kx^2/2dt ) - I*sin( +kx^2/2dt )
  }
  else if (ii < NX*NY*NZ) {
    // range [KMIN = -KMAX, -DK]
    const double kx_ii = 2*KxMIN + DKx*ii;
    propagator_T_dev[ii] = make_cuDoubleComplex( cos(kx_ii*kx_ii*DT/2),-sin(kx_ii*kx_ii*DT/2)  );
#ifdef DEBUG
    if (ii < NX*NY*NZ/2) printf("\nError in kernel creating propagator T!\tWrong index in 'higher part' of FFT.\n");
#endif
  }
  
  // CASE DIM 2D
#elif (DIM == 2)
  
  // CASE DIM 3D
#elif (DIM == 3)


#endif // case dimension for propagator T
  
}


/* ************************************************************************************************************************************* *
 * 																	 *
 * 							KERNELS TYPE ZD									 *
 * 																	 *
 * ************************************************************************************************************************************* */

__global__ void ker_modulus_pow2_wf_1d(cuDoubleComplex* complex_arr_dev, double* double_arr_dev) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (ii < NX*NY*NZ) {
    //double_arr_dev = cuCabs(complex_arr_dev[ii])*cuCabs(complex_arr_dev[ii]);
    double_arr_dev = cuCreal(complex_arr_dev[ii])*cuCreal(complex_arr_dev[ii]) + cuCimag(complex_arr_dev[ii])*cuCimag(complex_arr_dev[ii]);
  }
  
}

__global__ void ker_arg_wf_1d(cuDoubleComplex* complex_arr_dev, double* double_arr_dev) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (ii < NX*NY*NZ) {
    //double_arr_dev[ii] = atan2( cuCimag(complex_arr_dev[ii]), cuCreal(complex_arr_dev[ii]) ); // in case line below doesn't work
    double_arr_dev = cuCarg(complex_arr_dev[ii]); // this function is declared in cuda_complex_ext.cuh
  }
  
}

__global__ void ker_count_norm_wf_1d(cuDoubleComplex* complex_arr_dev, double* norm_dev) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  
}


/* ************************************************************************************************************************************* *
 * 																	 *
 * 							KERNELS TYPE ZZ									 *
 * 																	 *
 * ************************************************************************************************************************************* */

__global__ void ker_popagate_T(cuDoubleComplex* wf_momentum_dev, cuDoubleComplex* popagator_T_dev) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (ii < NX*NY*NZ) {
    wf_momentum_dev[ii] *= propagator_T_dev[ii];
  }
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