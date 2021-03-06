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





/* ************************************************************************************************************************************* *
 * 																	 *
 * 							KERNELS TYPE Z									 *
 * 																	 *
 * ************************************************************************************************************************************* */

#define SIGMA ( XMAX*sqrt(2./(3.14159265358979323846*NX)) )
#ifndef IMAG_TIME
    #define OFFSET_X ((double) 0.1)
#else
    #define OFFSET_X ((double) 0.0)
#endif
__global__ void ker_gauss_1d(cuDoubleComplex* data) {
  // get the index of thread
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  const uint64_t N = NX*NY*NZ;
  
  // allocate constants in shared memory
  //const double x0 = (-5*SIGMA);
  //const double dx = (10*SIGMA)/((double) N);
  
  if (ii < N) {
    data[ii] = make_cuDoubleComplex( sqrt(INV_SQRT_2PI/SIGMA)*exp(-(XMIN + ii*DX + OFFSET_X)*(XMIN + ii*DX + OFFSET_X)/4/(SIGMA*SIGMA)), 0. );
  }
  
  __syncthreads();
  if ( ii == 0)
    printf("sigma: %.15f\n",SIGMA);
}


__global__ void ker_const_1d(cuDoubleComplex* wf) {
  // get the index of thread
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
    
  while (ii < NX) {
    wf[ii] = make_cuDoubleComplex( (1./(XMAX-XMIN)), 0. );
    ii += blockDim.x * gridDim.x;
  }
}


/*
 * Divides the result of inverse cufft by number of samples (to get unitary form of DFT).
 * 
 * NOTE: cufftDoubleComplex is just typdef for cuDoubleComplex ! (no need to include <cuComplex.h> if only cufft necessary)
 * 	 here used to associate it with inverse cufft
 */
__global__ void ker_normalize_1d(cufftDoubleComplex* cufft_inverse_data) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  // in both kernel as well as callback we use predefined N to have comparable performance results
  
  while (ii < NX) {
    //cufft_inverse_data[ii] = make_cuDoubleComplex( cuCreal(cufft_inverse_data[ii])/((double) NX), cuCimag(cufft_inverse_data[ii])/((double) NX) );
    // check division Intrinsics ddiv_rz <- round to zero mode (maybe less problems with norm ??? & faster )
    cufft_inverse_data[ii] = make_cuDoubleComplex( __ddiv_rd(cuCreal(cufft_inverse_data[ii]),(double) NX) ,
                                                   __ddiv_rd(cuCimag(cufft_inverse_data[ii]),(double) NX) );
    ii += blockDim.x * gridDim.x;
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
    
    // make sure that tramsform will be unitary
    /*propagator_T_dev[ii] = make_cuDoubleComplex( cuCreal(propagator_T_dev[ii]) / cuCabs(propagator_T_dev[ii]),
						 cuCimag(propagator_T_dev[ii]) / cuCabs(propagator_T_dev[ii]) );*/
    
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


__global__ void ker_print_Z(cuDoubleComplex* arr_dev)
{
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  ii *= 32;
  while (ii < NX) {
    printf("%lu\t%.15f + %.15fj\t%.15f * exp( j*%.15f )\n", ii, cuCreal(arr_dev[ii]), cuCimag(arr_dev[ii]), cuCabs(arr_dev[ii]), cuCarg(arr_dev[ii]) );
    ii += blockDim.x * gridDim.x;
  }
}


/* ************************************************************************************************************************************* *
 * 																	 *
 * 							KERNELS TYPE ZD									 *
 * 																	 *
 * ************************************************************************************************************************************* */

__global__ void ker_multiplyZD(cuDoubleComplex* complex_arr_dev, double* double_arr_dev) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (ii < NX*NY*NZ) {
    // WYTESTOWAC CZY SZYBSZE NIE BEDZIE OBLICZANIE PROPAGATORA
    complex_arr_dev[ii] = cuCmul( complex_arr_dev[ii], double_arr_dev[ii] );
    ii += blockDim.x * gridDim.x;
  }
}

__global__ void ker_modulus_wf_1d(cuDoubleComplex* complex_arr_dev, double* double_arr_dev) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (ii < NX) {
    //double_arr_dev = cuCabs(complex_arr_dev[ii])*cuCabs(complex_arr_dev[ii]);
    double_arr_dev[ii] = sqrt( cuCreal(complex_arr_dev[ii])*cuCreal(complex_arr_dev[ii]) + cuCimag(complex_arr_dev[ii])*cuCimag(complex_arr_dev[ii]) );
    ii += blockDim.x * gridDim.x;
  }
  
}

__global__ void ker_modulus_pow2_wf_1d(cuDoubleComplex* complex_arr_dev, double* double_arr_dev) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
#ifdef DEBUG
  if(ii%4 == 0) printf("x:%.15f\twf:%15f + %15fj\tRe^2:%15f\tIm^2:%15f\n", XMIN + ii*DX, complex_arr_dev[ii].x, complex_arr_dev[ii].y, complex_arr_dev[ii].x*complex_arr_dev[ii].x, complex_arr_dev[ii].y*complex_arr_dev[ii].y);
#endif
  
  while (ii < NX) {
    //double_arr_dev = cuCabs(complex_arr_dev[ii])*cuCabs(complex_arr_dev[ii]);
    double_arr_dev[ii] = complex_arr_dev[ii].x*complex_arr_dev[ii].x + complex_arr_dev[ii].y*complex_arr_dev[ii].y;
    ii += blockDim.x * gridDim.x;
  }
  
}
// TODO: Check what is quicker (above or this):
__global__ void ker_density_wf_1d(cuDoubleComplex* complex_arr_dev, double* double_arr_dev) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  double re_wf_reg;
  double im_wf_reg;
#ifdef DEBUG
  if(ii%4 == 0) printf("x:%.15f\twf:%15f + %15fj\tRe^2:%15f\tIm^2:%15f\n", XMIN + ii*DX, complex_arr_dev[ii].x, complex_arr_dev[ii].y, complex_arr_dev[ii].x*complex_arr_dev[ii].x, complex_arr_dev[ii].y*complex_arr_dev[ii].y);
#endif
  
  while (ii < NX) {
    re_wf_reg = complex_arr_dev[ii].x;
    im_wf_reg = complex_arr_dev[ii].y;
    double_arr_dev[ii] = re_wf_reg*re_wf_reg + im_wf_reg*im_wf_reg;
    ii += blockDim.x * gridDim.x;
  }
  
}

__global__ void ker_arg_wf_1d(cuDoubleComplex* complex_arr_dev, double* double_arr_dev) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (ii < NX*NY*NZ) {
    //double_arr_dev[ii] = atan2( cuCimag(complex_arr_dev[ii]), cuCreal(complex_arr_dev[ii]) ); // in case line below doesn't work
    double_arr_dev[ii] = cuCarg(complex_arr_dev[ii]); // this function is declared in cuda_complex_ext.cuh
  }
  
}

// TODO: check if cublasZdscale is not more efficient?
__global__ void ker_normalize_1d(cuDoubleComplex* data, double* norm) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  // SPRAWDZIC CZY NIE DA SIE PRZYSPIESZYC POPRZEZ CONSTANT / SHARED MEMOMRY (SKOPIOWAC TAM WARTOSC NORMY) !!!
  
  while (ii < NX*NY*NZ) {
    data[ii] = make_cuDoubleComplex( cuCreal(data[ii])/(*norm)/sqrt(DX), cuCimag(data[ii])/(*norm)/sqrt(DX) );
    ii += blockDim.x * gridDim.x;
  }
}



/*
 * PROPAGATION VIA CONTACT INTERACTIONS
 */
__global__ void ker_propagate_Vcon_1d(cuDoubleComplex* wf, double* density) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  //double gpsi2dt_reg;
  
  while (ii < NX*NY*NZ) {
    //gpsi2dt_reg =  G_CONTACT*density[ii]*DT;
#ifdef REAL_TIME
    wf[ii] = cuCmul(  wf[ii], make_cuDoubleComplex(  cos( G_CONTACT*density[ii]*DT ),-sin( G_CONTACT*density[ii]*DT )  )  );
    //wf[ii] = cuCmul(  wf[ii], make_cuDoubleComplex(cos(factor[tid]*DT),-sin(factor[tid]*DT))  );
#endif
#ifdef IMAG_TIME
    wf[ii] = cuCmul( wf[ii], exp(-G_CONTACT*density[ii]*DT) );
#endif
    
    ii += blockDim.x * gridDim.x;
  }
}


/*
 * PROPAGATION VIA INTERACTIONS (with predefined interactions` potential)
 * 
 * (this evolution is made in positions` space)
 * cuDoubleComplex* wf - wavefunction
 * cuDoubleComplex* Vint - interactions` potential (integral is counted via convolution with FFT)
 */
__global__ void ker_propagate_Vint_1d(cuDoubleComplex* wf, cuDoubleComplex* Vint) {
  //extern __shared__ double factor[];
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  double Re_Vint = Vint[ii].x;
  double Im_Vint = Vint[ii].y;
  
  while (ii < NX*NY*NZ) {
    
    //factor[tid] = G_CONTACT*density[ii]*NX*NY*NZ;// gN|\psi|^2
    
#ifdef DEBUG
    if (Im_Vint > 0) printf("x: %.15f\twavefunction before progration Vint: %.15f + %.15fj\tVint: %.15f + %.15fj\n", XMIN + ii*DX,cuCreal(wf[ii]),cuCimag(wf[ii]),Re_Vint,Im_Vint);
    __syncthreads();
#endif
#ifdef REAL_TIME
    // TODO: Check if taking imaginary part is good? 
    wf[ii] = cuCmul(  wf[ii], make_cuDoubleComplex(  exp(Im_Vint*DT)*cos( Re_Vint*DT ),-exp(Im_Vint*DT)*sin( Re_Vint*DT )  )  );
#endif
#ifdef IMAG_TIME
    wf[ii] = cuCmul( wf[ii], make_cuDoubleComplex(  exp(-Re_Vint*DT)*cos( Re_Vint*DT ),-exp(Im_Vint)*sin( Re_Vint*DT )  ) );
#endif
#ifdef DEBUG
    if (ii < 10) printf("wavefunction after progration Vcon: %.15f + %.15fj\tVint: %.15f + %.15fj\n", XMIN + ii*DX,cuCreal(wf[ii]),cuCimag(wf[ii]),Re_Vint,Im_Vint);
#endif
    
    ii += blockDim.x * gridDim.x;
  }
}

/*
 *          PHASE IMPRINTING
 * double* phase is an array contaning phase on grid in radians
 */
__global__ void ker_phase_imprint_1d(cuDoubleComplex* wf) {
    uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
    
    // register variables <- quicker?
    double mod_wf;
    double phase_reg;
    
    while (ii < NX*NY*NZ) {
        mod_wf = cuCabs(wf[ii]);
        /* SINGLE SOLITON IN CENTRE 
        if ( (XMIN + ii*DX) < 0. ) phase_reg = -M_PI/2.;
        else phase_reg = M_PI/2.;
        */
#ifdef V_EXT
        /* SINGLE SOLITON MOVED RIGHT */
        if ( (XMIN + ii*DX) < SIGMA*0.5 ) phase_reg = -M_PI/2.;
        else phase_reg = M_PI/2.;
#else
        /* TWO SYMETRICAL SOLITONS 
        if ( ((XMIN + ii*DX) < -0.15 ) || ((XMIN + ii*DX) > 0.15 ) )
            phase_reg = 0.;
        else
            phase_reg = M_PI;*/
#endif
        wf[ii] = make_cuDoubleComplex( mod_wf*cos(phase_reg) , mod_wf*sin(phase_reg) );
        
        ii += blockDim.x * gridDim.x;
    }
    
}

__global__ void ker_phase_imprint_1d(cuDoubleComplex* wf, double* phase) {
    uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
    
    // register variables <- quicker?
    double mod_wf;
    double phase_reg;
    
    while (ii < NX*NY*NZ) {
        mod_wf = cuCabs(wf[ii]);
        phase_reg = phase[ii];
        
        wf[ii] = make_cuDoubleComplex( mod_wf*cos(phase_reg) , mod_wf*sin(phase_reg) );
        
        ii += blockDim.x * gridDim.x;
    }
    
}

__global__ void ker_phase_init_1d(double* phase) {
    uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
    
    double phase_reg;
    
    while (ii < NX*NY*NZ) {
        
        if ( (XMIN + ii*DX) < 0. ) phase_reg = -M_PI;
        else phase_reg = M_PI;
        
        ii += blockDim.x * gridDim.x;
    }
}

/* ************************************************************************************************************************************* *
 * 																	 *
 * 							KERNELS TYPE ZZ									 *
 * 																	 *
 * ************************************************************************************************************************************* */

/*
 * Element-wise vector multiplication
 */
__global__ void ker_multiplyZZ(cuDoubleComplex* wf_momentum_dev, cuDoubleComplex* propagator_dev) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (ii < NX*NY*NZ) {
    // WYTESTOWAC CZY SZYBSZE NIE BEDZIE OBLICZANIE PROPAGATORA
    wf_momentum_dev[ii] = cuCmul( wf_momentum_dev[ii], propagator_dev[ii] );
    ii += blockDim.x * gridDim.x;
  }
}

/*
 * Element-wise multiplication
 * Multiply elements of first array by real parts of elements of second array and saves result to second array
 */
__global__ void ker_multiplyZReZ(cuDoubleComplex* complex_arr1_dev, cuDoubleComplex* complex_arr2_dev) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (ii < NX*NY*NZ) {
    double factor = complex_arr2_dev[ii].x;
    complex_arr2_dev[ii] = cuCmul( complex_arr1_dev[ii], factor );
    ii += blockDim.x * gridDim.x;
  }
}


__global__ void ker_T_wf(cuDoubleComplex* wf_momentum_dev, cuDoubleComplex* result_dev) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (ii < NX*NY*NZ) {
    result_dev[ii] = cuCmul(wf_momentum_dev[ii], kx_dev(ii)*kx_dev(ii));
    ii += blockDim.x * gridDim.x;
  }
  
}

__global__ void ker_Vext_wf(cuDoubleComplex* wf_dev, cuDoubleComplex* result_dev) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (ii < NX*NY*NZ) {
    result_dev[ii] = cuCmul(wf_dev[ii], (XMIN + ii*DX)*(XMIN + DX*ii) );
    ii += blockDim.x * gridDim.x;
  }
  
}


/* ************************************************************************************************************************************* *
 * 																	 *
 * 							KERNELS TYPE ZDZ									 *
 * 																	 *
 * ************************************************************************************************************************************* */

/*
 * This function counts: | gN|\psi|^2 \psi>
 */
__global__ void ker_Vcon_wf(cuDoubleComplex* wf_dev, double* density, cuDoubleComplex* result_dev) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (ii < NX*NY*NZ) {
    result_dev[ii] = cuCmul(wf_dev[ii], G_CONTACT*density[ii] );
    ii += blockDim.x * gridDim.x;
  }
  
}


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
//static __device__ cuDoubleComplex propagate_Vext();
//static __device__ cuDoubleComplex propagate_T();
//static __device__ cuDoubleComplex propagate_Vcon();
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


// tests with callbacks


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
  return make_cuDoubleComplex( sqrt(INV_SQRT_2PI/SIGMA)*exp(-(x0 + offset*dx)*(x0 + offset*dx)/4/(SIGMA*SIGMA)), 0. );
}



