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


// util functions
__global__ 


// cross sections of wavefunction
__global__ get_cross_sectionX();
__global__ get_cross_sectionY();
__global__ get_cross_sectionZ();
__global__ get_cross_sectionXY();
__global__ get_cross_sectionXZ();
__global__ get_cross_sectionYZ();


// integrals
__global__ mean_kinetic_energy();
__global__ mean_potential_energy();
__global__ mean_contact_interaction_energy();
__global__ mean_dipolar_interaction_energy();
__global__ mean_momentum();
__global__ get_norm();
// angular momnetum???

// DEVICE ONLY FUNCTIONS

static __device__ normalize();
//__device__



// callbacks
static __device__ cuDoubleComplex propagate_Vext();
static __device__ cuDoubleComplex propagate_T();
static __device__ cuDoubleComplex propagate_Vcon();
static __device__ cuDoubleComplex propagate_Vdip();

__device__ cufftCallbackLoadZ CB_LD_MOMENTUM_SPACE_FORWARD();
__device__ cufftCallbackStoreZ CB_ST_MOMENTUM_SPACE_FORWARD();
__device__ cufftCallbackLoadZ CB_LD_MOMENTUM_SPACE_INVERSE();
__device__ cufftCallbackStoreZ CB_ST_MOMENTUM_SPACE_INVERSE();
__device__ cufftCallbackLoadD CB_LD_DIPOLAR_FORWARD();
__device__ cufftCallbackStoreZ CB_ST_DIPOLAR_FORWARD();
__device__ cufftCallbackLoadZ CB_LD_DIPOLAR_INVERSE();
__device__ cufftCallbackStoreZ CB_ST_DIPOLAR_INVERSE();