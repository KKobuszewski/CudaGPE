#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <math.h>
#include <complex.h>
//#include <gsl/...> <- NAJLEPIEJ STWORZYC LINK DO TEJ BIBLIOTEKI I DODAC DO /usr/include, /usr/lib/ BO SIE ZLE LINKUJE!
#include <cufft.h>
#include <cufftXt.h>
#include <cublas_v2.h>
#include <cuda.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "global.h"
#include "dipolar.h"
#include "simulation.cuh"
#include "cudautils.cuh"
#include "kernels.cuh"


// global variables
extern Globals* global_stuff;
bool FLAG_RUN_SIMULATION = true;
extern const char* thread_names[];
extern const char* stream_names[];


// timing
cudaEvent_t start_t;
cudaEvent_t stop_t;

#ifdef IMAG_TIME
const uint64_t time_tot = 1e-02/DT;
//const uint64_t time_tot = 1000000;
#else
//const uint64_t time_tot = llround(0.318309886183791/DT); // no Vext revival time
const uint64_t time_tot = 2*llround((2*3.14159265358979323846/OMEGA)/DT); // harmonic potential revival time
#endif
const uint64_t frames_to_be_saved = 100;


// pthread managment
pthread_barrier_t barrier;


// cuda libs global variables
cublasHandle_t cublas_handle;
cufftHandle* cufft_plans;


// pointers to device functions
//dev_funcZ_ptr_t operator_T_h_ptr;
//dev_funcZ_ptr_t operator_Vext_h_ptr;

// variables on device only
//__constant__ double* norm_dev_con; // <- copy this variable to constant memory


/* ************************************************************************************************************************************* *
 * 																	 *
 * 							FUNC DEFINITIONS								 *
 * 																	 *
 * ************************************************************************************************************************************* */


void free_device();
void save_stats(uint64_t step_index);
void save_simulation_params();


/* ************************************************************************************************************************************* *
 * 																	 *
 * 							COMPUTIONAL FUNCTIONS								 *
 * 																	 *
 * ************************************************************************************************************************************* */

/*
 * This function counts chemical potential in time-independent Gross-Pitaevskii equation from relative norm change during ITE step.
 * ( Wavefunction is being normed every step, so the relative change is the value of norm in current step! )
 * 
 */
static inline long double chemical_potential_ite(long double norm) {
    return -logl(norm)/( (long double) DT);
}
/*
 * Less accurate but faster version of function above.
 */
static inline double chemical_potential_ite(double norm) {
    return -log(norm)/DT;
}


// interaction potential
static inline double interaction_potential(uint32_t ii) {
    return Vdd(kx(ii),1.,1.) + G_CONTACT; // dipolar + contact interactions
}



/* ************************************************************************************************************************************* *
 * 																	 *
 * 							SIM VARIABLES									 *
 * 																	 *
 * ************************************************************************************************************************************* */

// data structures on host (pinnable memory) TODO: Check how much of this could be allocated!
double complex* wf_host;
double complex* propagator_T_host;
double complex* propagator_Vext_host;
double complex* density_k_host;
double* density_r_host;
double* Vdd_host;

// data structures on device
cuDoubleComplex* complex_arr1_dev; // pointer on array holding wavefunction in device memory
cuDoubleComplex* complex_arr2_dev;
cuDoubleComplex* complex_arr3_dev;
double* double_arr1_dev;
cuDoubleComplex* propagator_T_dev; // array of constant factors e^-ik**2/2dt
cuDoubleComplex* propagator_Vext_dev; // array of constant factors e^-iVextdt
double* Vdd_dev; // array of costant factors <- count on host with spec funcs lib or use Abramowitz & Stegun approximation

// statistics
double norm_host;
double* norm_dev;

double chemical_potential_host;
double* chemical_potential_dev;

double mean_T_host;
double complex meanZ_T_host;
cuDoubleComplex* meanZ_T_dev;

double mean_Vext_host;
double complex meanZ_Vext_host;
cuDoubleComplex* meanZ_Vext_dev;

double mean_Vcon_host;
double complex meanZ_Vcon_host;
cuDoubleComplex* meanZ_Vcon_dev;

double mean_Vdip_host;
double complex meanZ_Vdip_host;
cuDoubleComplex* meanZ_Vdip_dev;

/* ************************************************************************************************************************************* *
 * 																	 *
 * 							SIM THREAD									 *
 * 																	 *
 * ************************************************************************************************************************************* *
 *
 * - allocation memory on host
 * - initialization of data
 * - main algorithm
 */
void* simulation_thread(void* passing_ptr) {
  
  
  
  //stick_this_thread_to_core(1); <- in cudautils, not used, include to header first
  pthread_barrier_wait (&barrier_global);
  printf("running %s thread.\n",thread_names[SIMULATION_THRD]);
  
  // allocate pinnable memory on host
  // arrays
  cudaHostAlloc((void**) &(wf_host), sizeof(double complex)*NX*NY*NZ, cudaHostAllocDefault); // pinnable memory <- check here for cudaMallocHost (could be faster)
  cudaHostAlloc((void**) &propagator_T_host, sizeof(double complex)*NX*NY*NZ, cudaHostAllocDefault); // pinnable memory <- check here for cudaMallocHost
  cudaHostAlloc((void**) &propagator_Vext_host, sizeof(double complex)*NX*NY*NZ, cudaHostAllocDefault); // pinnable memory <- check here for cudaMallocHost (could be faster)
  cudaHostAlloc((void**) &density_k_host, sizeof(double complex)*NX*NY*NZ, cudaHostAllocDefault); // for copying denstiy of wf in Fourier space
  cudaHostAlloc((void**) &density_r_host, sizeof(double)*NX*NY*NZ, cudaHostAllocDefault);
  cudaHostAlloc((void**) &Vdd_host, sizeof(double)*NX*NY*NZ, cudaHostAllocDefault);
  
  //scalars
  cudaHostAlloc((void**) &norm_host, sizeof(double), cudaHostAllocDefault); // pinnable memory <- check here for cudaMallocHost (could be faster)
  
  printf("allocated memory on host.\n");
  
  
#ifdef DEBUG
  printf("1st barrier reached by %s.\n",thread_names[SIMULATION_THRD]);
#endif
  pthread_barrier_wait (&barrier);
  // copy data async from host to device (if needed)
  if (global_stuff->init_wf_fd != -1) {
    // copy data from host to device (if needed) / cannot async because
    printf("copying initial wavefunction on device\n");
    HANDLE_ERROR( cudaMemcpy(complex_arr1_dev, init_wf_mmap, NX*NY*NZ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) ); // change to asynchronous!
  }
  else {
    
    printf("initating wavefunction on device.\n");
#ifndef V_CON
    call_kernel_Z_1d( ker_gauss_1d, complex_arr1_dev, (streams)[SIMULATION_STREAM] );
    //call_kernel_Z_1d( ker_const_1d, complex_arr1_dev, (streams)[SIMULATION_STREAM] );
#else
    call_kernel_Z_1d( ker_gauss_1d, complex_arr1_dev, (streams)[SIMULATION_STREAM] );
    //call_kernel_Z_1d( ker_const_1d, complex_arr1_dev, (streams)[SIMULATION_STREAM] );
#endif    
  }
  
  // copying device pointers to host memory (to make them callable in kernels)
  //cudaMemcpyFromSymbol(&operator_T_h_ptr, operator_T_dev_ptr, sizeof(dev_funcZ_ptr_t));
  //cudaMemcpyFromSymbol(&operator_Vext_h_ptr, operator_Vext_dev_ptr, sizeof(dev_funcZ_ptr_t));
  
  printf("creating propagators\n");
  /*
   * TODO: - do it parallel with OpenMP! (6 cores)
   * 
   * it is possioble to do it on device
   * call_kernel_Z_1d( ker_create_propagator_T, propagator_T_dev, (streams)[HELPER_STREAM] );
   */
  omp_set_num_threads(6);
  #pragma omp parallel for
  for( uint64_t ii=0; ii < NX; ii++ ) {
#ifdef REAL_TIME
    propagator_T_host[ii] = cexp(-I*kx(ii)*(0.5*kx(ii)*DT));
    propagator_Vext_host[ii] = cexp(-I*(0.5*OMEGA)*(OMEGA*(ii*DX+XMIN))*((ii*DX+XMIN)*DT)); // <- !!! KOLEJNOSC MNOZEMIA A DOKLADNOSC !!!
#endif
#ifdef IMAG_TIME   
    propagator_T_host[ii] = cexpl(-kx(ii)*0.5*kx(ii)*DT);
    propagator_Vext_host[ii] = cexpl(-(0.5*OMEGA*OMEGA*(ii*DX+XMIN)*(ii*DX+XMIN)*DT)); // <- !!! KOLEJNOSC MNOZEMIA A DOKLADNOSC !!!
#endif
    //printf("%.15f + %.15fj\n",creal(propagator_Vext_host[ii]), cimag(propagator_Vext_host[ii]) );
#ifdef V_DIP
    // test it using contact interactions only
    Vdd_host[ii] = Vdd(kx(ii),1.,1.) + G_CONTACT*SQRT_2PI; // dipolar + contact interactions
#endif
  }
  
  /*
   * CHECK IF IT IS DONE IN pipelining?
   * probably not...
   */
  
  // copying propag T to dev
  HANDLE_ERROR( cudaMemcpyAsync(propagator_T_dev, propagator_T_host,
				NX*NY*NZ*sizeof(cuDoubleComplex),
				cudaMemcpyHostToDevice,
				(streams)[HELPER_STREAM]) );
  
  // copying after initialization (in meantime on another stream)
  HANDLE_ERROR( cudaMemcpyAsync(wf_host, complex_arr1_dev,
				NX*NY*NZ*sizeof(cuDoubleComplex),
				cudaMemcpyDeviceToHost,
				(streams)[SIMULATION_STREAM]) );
  cudaDeviceSynchronize();
  
  
  // copying propag Vext to dev
  HANDLE_ERROR( cudaMemcpyAsync(propagator_Vext_dev, propagator_Vext_host,
				NX*NY*NZ*sizeof(cuDoubleComplex),
				cudaMemcpyHostToDevice,
				(streams)[HELPER_STREAM]) );
#ifdef V_DIP
  // copying propag Vext to dev
  HANDLE_ERROR( cudaMemcpyAsync(Vdd_dev, Vdd_host,
				NX*NY*NZ*sizeof(cuDoubleComplex),
				cudaMemcpyHostToDevice,
				(streams)[SIMULATION_STREAM]) );
#endif
  
#ifdef DEBUG
  /*
  HANDLE_ERROR( cudaMemcpyAsync(propagator_T_host, propagator_T_dev,
				NX*NY*NZ*sizeof(cuDoubleComplex),
				cudaMemcpyDeviceToHost,
				(streams)[HELPER_STREAM]) );
				*/
#endif
  
  // saving to file initial wavefuntion (1st frame) <- CZY TO JEST POTRZEBNE ???
  fwrite( wf_host, sizeof(double complex), NX*NY*NZ, (files[WF_FRAMES_FILE])->data );
  
  // saving to file propagators T, Vext, and F{ Vdd }
  /*
   * TODO: place this in helper thread on host
   */
  fprintf( (files[PROPAGATORS_FILE])->data, "x\t\t\tRe[e^-iVext(x)dt]\tIm[e^-iVext(x)dt]\tkx\t\t\tRe[e^-iT(kx)dt]\tIm[e^-iT(kx)dt]\tVdd\n"); // header
  for (uint64_t ii=0 ; ii < NX*NY*NZ; ii++) {
         fprintf( (files[PROPAGATORS_FILE])->data, "%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\n", XMIN + ii*DX, creal(propagator_Vext_host[ii]), cimag(propagator_Vext_host[ii]),
                                                                            kx(ii), creal(propagator_T_host[ii]), cimag(propagator_T_host[ii]),
                                                                            Vdd_host[ii]);
  }
//   for (uint64_t ii=0 ; ii < NX*NY*NZ/2; ii++)
//          fprintf( (files[PROPAGATORS_FILE])->data, "%.15f\t%.15f\t%.15f\n", ii, creal(propagator_Vext_host[ii]), cimag(propagator_Vext_host[ii]) );
  
#ifdef DEBUG
  
  printf("2nd barrier reached by %s.\n",thread_names[SIMULATION_THRD]);
  printf("FLAG_RUN_SIMULATION %u\n",FLAG_RUN_SIMULATION);
#endif
  cudaStreamSynchronize( (streams)[HELPER_STREAM] );
  pthread_barrier_wait (&barrier);
  
  
  
#ifdef DEBUG
  // checking total norm of propagator T (should be sqrt[Nx*|e^ia|^2] = sqrt[Nx], because Im[a] == 0 )
  CHECK_CUBLAS( cublasDznrm2( cublas_handle, NX*NY*NZ, propagator_T_dev, 1, norm_dev) );
  cudaDeviceSynchronize();
  HANDLE_ERROR( cudaMemcpyAsync(&norm_host, norm_dev,
				sizeof(double),
				cudaMemcpyDeviceToHost,
				(streams)[HELPER_STREAM]) );
  cudaDeviceSynchronize();
  fprintf( (files[PROPAGATORS_FILE])->data, "norm (cublas) propagator_T_dev: %.15f\n", norm_host );
#endif
  
  
  // checking norm of initial wavefunction
  CHECK_CUBLAS( cublasDznrm2( cublas_handle, NX*NY*NZ, complex_arr1_dev, 1, norm_dev) );
  cudaDeviceSynchronize();
  HANDLE_ERROR( cudaMemcpyAsync(&norm_host, norm_dev,
        			sizeof(double),
				cudaMemcpyDeviceToHost,
				(streams)[HELPER_STREAM]) );
  cudaStreamSynchronize( streams[HELPER_STREAM] );
  norm_host *= sqrt(DX);  
  fprintf( (files[STATS_FILE])->data, "norm of initial wf: %.15f\tdx: %.15f\tsqrt dx: %.15f\n\n", norm_host, DX, sqrt(DX) );
  
  // header of a file with statistics
  fprintf( (files[STATS_FILE])->data, "\n\nt [dt]:\tnorm:\t\t\tchemic. pot.\t\t<T>\t\t\t<Vext>\t\t\t<Vcon>\t\t\t<Vdip>\n" );
  
  
       /* *************************************************************************************************************************************** *
	* 																	  *
	* 							ALGORITHM LOOP									  *
	* 																	  *
	* *************************************************************************************************************************************** */
  uint64_t saving_steps = frames_to_be_saved;
  uint64_t timesteps_tot = time_tot/frames_to_be_saved;
  uint64_t timesteps;
  
  printf("\n\n");
  printf("dt: %e\n",DT);
  printf("time total (in dt): %lu\n", time_tot);
  printf("time between saving (in dt): %lu\n", timesteps_tot);
  printf("savings: %lu\n", frames_to_be_saved);
  printf("\n");
  
  uint32_t counter = 0;
  while( FLAG_RUN_SIMULATION ) { // simulation will be runing until the flag is set to false
#ifdef DEBUG
     timesteps = 2;
     saving_steps =1;
     printf("timesteps to be made: %lu\n", timesteps);
#else
     timesteps = timesteps_tot;
     printf("%lu. timesteps to be made: %lu\n", saving_steps, timesteps);
#endif
     saving_steps--;
     
     while(timesteps) {
       timesteps--;
       /* *************************************************************************************************************************************** *
	* 																	  *
	* 							ALGORITHM STEP									  *
	* 																	  *
	* *************************************************************************************************************************************** */
       // multiply by Vext propagator (do in callback load) !*
       
       // make copy of wavefunction
       CHECK_CUBLAS( cublasZcopy(cublas_handle, NX*NY*NZ, complex_arr1_dev, 1, complex_arr3_dev, 1) );
       cudaDeviceSynchronize();
       
       
       /*
        *       EVOLVE IN MOMENTUM SPACE
        */
       
       //printf("\ntransforming wavefunction to momentum space\n");
       CHECK_CUFFT( cufftExecZ2Z((cufft_plans)[FORWARD_PSI],
				 complex_arr1_dev,
				 complex_arr2_dev,
				 CUFFT_FORWARD) );
       
       // count |\psi|^2 array in meanwhile
       //cudaStreamSynchronize(streams[SIMULATION_STREAM]);
       call_kernel_ZD_1d( ker_modulus_pow2_wf_1d, complex_arr3_dev, double_arr1_dev, streams[HELPER_STREAM]);
       // it could be replaced with complex_arr3_dev <- maybe faster to copy array with cublas and do not synchronize streams 
       
       // multiply by T propagator (do in callback) <- J
       call_kernel_ZZ_1d( ker_multiplyZZ, complex_arr2_dev, propagator_T_dev, (streams)[SIMULATION_STREAM] );
       
       
       
       // go back to 'positions`'? space <- JAK JEST PO ANGIELSKU PRZESTRZEN POLOZEN ???
       CHECK_CUFFT( cufftExecZ2Z((cufft_plans)[BACKWARD_PSI],
				 complex_arr2_dev,
				 complex_arr1_dev,
				 CUFFT_INVERSE) );
       // run kernel to normalize aftter FFT
       call_kernel_Z_1d( ker_normalize_1d, complex_arr1_dev, (streams)[SIMULATION_STREAM] );
       
       
#ifdef V_DIP
       cudaStreamSynchronize(streams[HELPER_STREAM]); // ensure that density is counted 
       // count DFT of modulus of wavefunction (in positions` space).
       // CUFFT is done in HELPER_STREAM
       CHECK_CUFFT( cufftExecD2Z((cufft_plans)[FORWARD_DIPOLAR],
				 double_arr1_dev,
				 complex_arr3_dev) ); // double to complex must be forward, so no need to specify direction
       // now in double_arr1_dev we have density in positions space represantation
       // and in complex_arr3_dev we have Fourier transform of density
#endif       
       
       /*
        *       EVOLVE IN POSITIONS` SPACE
        */
       
       // evolve via external potential Vext (if defined)
#ifdef V_EXT
       call_kernel_ZZ_1d( ker_multiplyZZ, complex_arr1_dev, propagator_Vext_dev, (streams)[SIMULATION_STREAM] );
#endif
       
       // evolve via contact interactions potential
#ifdef V_CON
 #ifndef VDIP
       cudaStreamSynchronize(streams[HELPER_STREAM]); // make sure double_arr1_dev is filled with |\psi|^2
       call_kernel_ZD_1d( ker_propagate_Vcon_1d, complex_arr1_dev, double_arr1_dev,(streams)[SIMULATION_STREAM] );
 #endif
#endif
       
#ifdef VDIP
       // multiply Fourier Transform of density with Fourier transform of density
       call_kernel_ZD_1d( ker_multiplyZD, complex_arr3_dev, Vdd_dev, (streams)[HELPER_STREAM] ); // <- TODO: Do it in callback load !!!
       
       // count integral in potential of dipolar interactions - convolution (on complex_arr3_dev in place) 
       CHECK_CUFFT( cufftExecZ2Z((cufft_plans)[BACKWARD_DIPOLAR],
				 complex_arr3_dev,
				 complex_arr3_dev,
				 CUFFT_INVERSE) );
       // normalize (not in callback store)
       
       // now in complex_arr3_dev we have Vint(r)
       
       // create propagator of Vdip (in) / propagate Vdip
       call_kernel_ZZ_1d( ker_propagate_Vint_1d, complex_arr1_dev, complex_arr3_dev, (streams)[SIMULATION_STREAM] );
#endif
       
       
       
#ifdef IMAG_TIME
       cudaStreamSynchronize(streams[SIMULATION_STREAM]); // make sure evolution via interactions is completed
       // normalize wavefunction to |\psi|^2 = 1 (at every step!)
       CHECK_CUBLAS( cublasDznrm2( cublas_handle, NX*NY*NZ, complex_arr1_dev, 1, norm_dev) ); // count norm
       cudaDeviceSynchronize(); // ensure norm is from current step
       call_kernel_ZD_1d( ker_normalize_1d, complex_arr1_dev, norm_dev, (streams)[SIMULATION_STREAM] ); // normalize
#endif
       
       /* *************************************************************************************************************************************** *
	* 																	  *
	* 							END OF ALGORITHM STEP								  *
	* 																	  *
	* *************************************************************************************************************************************** */
       
       
       
       // compute and save statistics of a system (norm, energy, ... )
#ifndef DEBUG
#ifdef REAL_TIME
       if ( counter%500 == 0 ) {
#endif
#ifdef IMAG_TIME
       if ( counter%50 == 0 ) {
#endif
#endif
         save_stats(counter);
         
#ifndef DEBUG
       }
#endif
       counter++;
       
     }     
       // saving wavefunction
       HANDLE_ERROR( cudaMemcpy(wf_host, complex_arr1_dev, NX*NY*NZ*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );
       /*for (uint64_t ii=0 ; ii < NX*NY*NZ; ii++)
         fprintf( ((files[])->data), "%.15f\t%.15f\t%.15f\n", XMIN+DX*ii, creal((wf_host)[ii]), cimag((wf_host)[ii]) );*/
       fwrite( wf_host, sizeof(double complex), NX*NY*NZ, (files[WF_FRAMES_FILE])->data );
       
       //save_stats(timesteps_tot-timesteps)*(frames_to_be_saved-saving_steps)
#ifdef DEBUG     
       //call_kernel_Z_1d( ker_print_Z, complex_arr1_dev, (streams)[SIMULATION_STREAM] );
#endif
     if (!saving_steps) FLAG_RUN_SIMULATION = false;
  }
  
  // saving wavefunction to binary file
  HANDLE_ERROR( cudaMemcpy(wf_mmap, complex_arr1_dev, NX*NY*NZ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) ); // change to asynchronous!
  
#ifdef DEBUG
  printf("last barrier reached by %s.\n",thread_names[SIMULATION_THRD]);
#endif
  cudaDeviceSynchronize();
  pthread_barrier_wait (&barrier);
  
  // free memory on host
  HANDLE_ERROR( cudaFreeHost(wf_host) );
#ifdef DEBUG
  HANDLE_ERROR( cudaFreeHost(propagator_T_host) );
#endif
  
  free_device();
  /*
  HANDLE_ERROR( cudaFree(complex_arr1_dev) ); 	//
  HANDLE_ERROR( cudaFree(complex_arr2_dev) ); 	//
  //HANDLE_ERROR( cudaFree(double_arr1_dev)  ); 	//
  HANDLE_ERROR( cudaFree(propagator_T_dev) ); 	//
  //HANDLE_ERROR( cudaFree(propagator_Vext_dev) );	//
  //HANDLE_ERROR( cudaFree(Vdd_dev) );		//
  
  
  //HANDLE_ERROR( cudaFree(global_stuff->mean_T_dev) ); // result of integral with kinetic energy operator in momentum representaion
  //HANDLE_ERROR( cudaFree(mean_Vdip_dev) ); // result of integral with Vdip operator in positions' representation
  //HANDLE_ERROR( cudaFree(global_stuff->mean_Vext_dev) ); // result of integral with Vext operator in positions' representation
  //HANDLE_ERROR( cudaFree(global_stuff->mean_Vcon_dev) ); // result of integral with Vcon operator in positions' representation
  HANDLE_ERROR( cudaFree(norm_dev) ); //
  */
  pthread_barrier_wait (&barrier_global);
  pthread_exit(NULL);
}





/* ************************************************************************************************************************************* *
 * 																	 *
 * 							HELPER THREAD									 *
 * 																	 *
 * ************************************************************************************************************************************* */

/*
 * - allocation memory on device
 * - allocation plans
 * 
 */
void* helper_thread(void* passing_ptr) {
  
  //stick_this_thread_to_core(2);
  
  pthread_barrier_wait (&barrier_global);
  printf("running %s thread.\n",thread_names[HELPER_THRD]);
  
  // init memory on device
  // arrays for wavefunction
  HANDLE_ERROR( cudaMalloc((void**) &(complex_arr1_dev), sizeof(cuDoubleComplex) * NX*NY*NZ) ); 	//
  HANDLE_ERROR( cudaMalloc((void**) &(complex_arr2_dev), sizeof(cuDoubleComplex) * NX*NY*NZ) ); 	//
  HANDLE_ERROR( cudaMalloc((void**) &complex_arr3_dev, sizeof(cuDoubleComplex) * NX*NY*NZ) ); 	//
  HANDLE_ERROR( cudaMalloc((void**) &(double_arr1_dev), sizeof(double) * NX*NY*NZ) );		//
  
  // constant arrays
  HANDLE_ERROR( cudaMalloc((void**) &(propagator_T_dev), sizeof(cuDoubleComplex) * NX*NY*NZ) ); 	// array of constant factors e^-i*k**2/2*dt
  HANDLE_ERROR( cudaMalloc((void**) &(propagator_Vext_dev), sizeof(cuDoubleComplex) * NX*NY*NZ) );// array of constant factors e^-i*Vext*dt
  HANDLE_ERROR( cudaMalloc((void**) &(Vdd_dev), sizeof(double) * NX*NY*NZ) ); 			// array of costant factors <- count on host with spec funcs lib or use Abramowitz & Stegun approximation
  
  // scalar variables
  HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->mean_T_dev), sizeof(double))    ); // result of integral with kinetic energy operator in momentum representaion
  HANDLE_ERROR( cudaMalloc((void**) &meanZ_T_dev, sizeof(cuDoubleComplex))    ); // result of integral with kinetic energy operator in momentum representaion
  HANDLE_ERROR( cudaMalloc((void**) &meanZ_Vext_dev, sizeof(cuDoubleComplex))    ); // result of integral with kinetic energy operator in momentum representaion
  HANDLE_ERROR( cudaMalloc((void**) &meanZ_Vcon_dev, sizeof(cuDoubleComplex))    ); // result of integral with kinetic energy operator in momentum representaion
  HANDLE_ERROR( cudaMalloc((void**) &(meanZ_Vdip_dev), sizeof(cuDoubleComplex)) ); // result of integral with Vdip operator in positions' representation
  HANDLE_ERROR( cudaMalloc((void**) &(norm_dev), sizeof(double)) ); // variable to hold norm of wavefunction
  
  
#ifdef DEBUG
  printf("allocated memory on device.\n");
#endif
  
  for (uint8_t ii = 0; ii < num_streams; ii++)
    HANDLE_ERROR(	cudaStreamCreate( &(streams[ii]) )	);
  
#ifdef DEBUG
  printf("1st barrier reached by %s.\n",thread_names[HELPER_THRD]);
#endif
  //cudaDeviceSynchronize();
  pthread_barrier_wait (&barrier);
  
  // creating plans with callbacks
  cufft_plans = (cufftHandle*) malloc( (size_t) sizeof(cufftHandle)*num_plans );
#ifdef DEBUG
  printf("array of plans allocated.\n");
#endif
  for (uint8_t ii = 0; ii < num_plans; ii++) {
    CHECK_CUFFT(  cufftCreate( (cufft_plans)+ii )  ); // allocates expandable plans
    //printf("%d\n",(cufft_plans)[ii]);
  }
  
#ifdef DEBUG
  printf("expandable plans allocated.\n");
#endif
  
  size_t work_size; // CHYBA TO MUSI BYC TABLICA !!!
#if (DIM == 1)
  printf("creating CUFFT plans in 1d case.\n");
  // wavefunction forward
  CHECK_CUFFT( cufftMakePlan1d( (cufft_plans)[FORWARD_PSI], NX*NY*NZ, CUFFT_Z2Z, 1, &work_size ) 	);
  CHECK_CUFFT( cufftSetStream(  (cufft_plans)[FORWARD_PSI], (streams)[SIMULATION_STREAM] ) );
  //printf("%d\n",(cufft_plans)[FORWARD_PSI]);
  
  // wavefunction inverse
  //  printf("%p\n",(cufft_plans)+BACKWARD_PSI);
  CHECK_CUFFT( cufftMakePlan1d( (cufft_plans)[BACKWARD_PSI], NX*NY*NZ, CUFFT_Z2Z, 1, &work_size )	);
  CHECK_CUFFT( cufftSetStream(  (cufft_plans)[BACKWARD_PSI], (streams)[SIMULATION_STREAM]) );
  //printf("%d\n",(cufft_plans)[BACKWARD_PSI]);
  
  // modulus of wavefunction forward
  CHECK_CUFFT( cufftMakePlan1d( (cufft_plans)[FORWARD_DIPOLAR], NX*NY*NZ, CUFFT_D2Z, 1, &work_size )	);
  CHECK_CUFFT( cufftSetStream(  (cufft_plans)[FORWARD_DIPOLAR], (streams)[HELPER_STREAM] ) );
  //printf("%d\n",(cufft_plans)[FORWARD_DIPOLAR]);
  
  // integral in potential of dipolar inteaction
  CHECK_CUFFT( cufftMakePlan1d( (cufft_plans)[BACKWARD_DIPOLAR], NX*NY*NZ, CUFFT_Z2Z, 1, &work_size )	);
  CHECK_CUFFT( cufftSetStream(  (cufft_plans)[BACKWARD_DIPOLAR], (streams)[HELPER_STREAM]) ); // WLASCIWIE TUTAJ NIE WIADOMO W KTORYM STREAMIE?
  //printf("%d\n",(cufft_plans)[BACKWARD_DIPOLAR]);
  
#elif (DIM == 2)
  
#elif (DIM == 3)
  
#endif // case DIM for plan
  printf("\tplans created\n");
  
  // !!! SPRAWDZIC !!! funckje: <- co robia?
  //cufftResult cufftSetAutoAllocation(cufftHandle *plan, bool autoAllocate);
  //cufftSetCompatibilityMode() <- musi byc wywolana po create a przed make plan
  
  
  /* ************************************
   * 			CUBLAS		*
   * ************************************/
  
  CHECK_CUBLAS( cublasCreate(&cublas_handle) );
  CHECK_CUBLAS( cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE) ); // this means we can use only device pointers to scalars (required by CUBLAS routines)
  
  printf("CUBLAS initialized!\n");
  
  
  
  
  
  
#ifdef DEBUG
  printf("created FFT plans.\n");
#endif
#ifdef DEBUG
  printf("2nd barrier reached by %s.\n",thread_names[HELPER_THRD]);
#endif
  pthread_barrier_wait (&barrier);
  
  save_simulation_params();
  
  // start algorithm
  // dt =
  /*uint16_t timesteps;
  while( FLAG_RUN_SIMULATION ) { // simulation will be runing until the flag is set to false
#ifdef DEBUG
     timesteps = 1;
#else
     timesteps = 1000;
#endif
     while(timesteps) {
       timesteps--;
       
       //FLAG_RUN_SIMULATION = false;
     }
  }*/
  
  //cudaDeviceSynchronize();
  pthread_barrier_wait (&barrier);
#ifdef DEBUG
  printf("last barrier reached by %s.\n",thread_names[HELPER_THRD]);
#endif
  
  CHECK_CUBLAS( cublasDestroy(cublas_handle) );
  //CHECK_CUBLAS( cublasShutdown() );
  
  
  
  
  for (uint8_t ii = 0; ii < num_streams; ii++)
    HANDLE_ERROR(	cudaStreamDestroy( (streams[ii]) )	);
  
  pthread_barrier_wait (&barrier_global);
  pthread_exit(NULL);
}


void alloc_device(){
  
  
}


void alloc_host() {
  
  // must use
  
}

void free_device() {
  // free memory on device
  HANDLE_ERROR( cudaFree(complex_arr1_dev) ); 	//
  HANDLE_ERROR( cudaFree(complex_arr2_dev) ); 	//
  HANDLE_ERROR( cudaFree(complex_arr3_dev) ); 	//
  HANDLE_ERROR( cudaFree(double_arr1_dev)  ); 	//
  HANDLE_ERROR( cudaFree(propagator_T_dev) ); 	//
  HANDLE_ERROR( cudaFree(propagator_Vext_dev) );	//
  HANDLE_ERROR( cudaFree(Vdd_dev) );		//
  
  
  //HANDLE_ERROR( cudaFree(mean_T_dev) ); // result of integral with kinetic energy operator in momentum representaion
  //HANDLE_ERROR( cudaFree(mean_Vdip_dev) ); // result of integral with Vdip operator in positions' representation
  //HANDLE_ERROR( cudaFree(mean_Vext_dev) ); // result of integral with Vext operator in positions' representation
  //HANDLE_ERROR( cudaFree(global_stuff->mean_Vcon_dev) ); // result of integral with Vcon operator in positions' representation
  HANDLE_ERROR( cudaFree(norm_dev) ); //
  
}


void free_pinnable() {
    
}

/* ************************************************************************************************************************************* *
 * 																	 *
 * 							SAVING FUNCTIONS								 *
 * 																	 *
 * ************************************************************************************************************************************* */

/*
 * Saving statistics of the system, assuming that complex_arr1 is the wavefunction after step_index-th iteration.
 * uint64_t step_index - index of iteration (timestep)
 */
void save_stats(uint64_t step_index) {
  // saving stats in pipelining mode
  
  // count norm with cublas (it is already done in case of imaginary-time evolution)
#ifndef IMAG_TIME
  CHECK_CUBLAS( cublasDznrm2( cublas_handle, NX*NY*NZ, complex_arr1_dev, 1, norm_dev) );
#endif
  
  cudaDeviceSynchronize();
  
  // count <T> and copy norm in parallel
  CHECK_CUFFT( cufftExecZ2Z((cufft_plans)[FORWARD_PSI], complex_arr1_dev, complex_arr2_dev, CUFFT_FORWARD) ); // make sure we have copy of wavefunction in momentum space
  HANDLE_ERROR( cudaMemcpyAsync(&norm_host, norm_dev, sizeof(double), cudaMemcpyDeviceToHost, (streams)[HELPER_STREAM]) ); // copy norm to host in parallel
  cudaStreamSynchronize( (streams)[SIMULATION_STREAM] ); // make sure FFT done
  call_kernel_ZZ_1d( ker_T_wf, complex_arr2_dev, complex_arr3_dev, (streams)[SIMULATION_STREAM]); // make vector T|\psi>
  cudaStreamSynchronize( (streams)[SIMULATION_STREAM] ); // make sure FFT done
  CHECK_CUBLAS( cublasZdotc(cublas_handle, NX*NY*NZ, (complex_arr2_dev), 1, complex_arr3_dev, 1, meanZ_T_dev) ); // count <\psi||T \psi>
  cudaDeviceSynchronize();
  
  // count <Vext> and copy <T> in parallel
#ifdef V_EXT
  call_kernel_ZZ_1d( ker_Vext_wf, complex_arr1_dev, complex_arr3_dev, (streams)[SIMULATION_STREAM]); // make vector Vext|\psi>
#endif
  HANDLE_ERROR( cudaMemcpyAsync(&meanZ_T_host, meanZ_T_dev, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, (streams)[HELPER_STREAM]) ); // copy <T> in parallel
  cudaStreamSynchronize( (streams)[SIMULATION_STREAM] ); // make sure |Vext \psi> is done
#ifdef V_EXT
  CHECK_CUBLAS( cublasZdotc(cublas_handle, NX*NY*NZ, complex_arr1_dev, 1, complex_arr3_dev, 1, meanZ_Vext_dev) ); // count <\psi||T \psi>
#endif
  cudaDeviceSynchronize();
  
  // count <Vcon> and copy <Vext> in parallel
#ifdef V_CON
  call_kernel_ZDZ_1d( ker_Vcon_wf, complex_arr1_dev, double_arr1_dev, complex_arr3_dev, streams[SIMULATION_STREAM] );
#endif
#ifdef V_EXT
  HANDLE_ERROR( cudaMemcpyAsync(&meanZ_Vext_host, meanZ_Vext_dev, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, (streams)[HELPER_STREAM]) ); // copy <Vext> in parallel
#endif
  cudaStreamSynchronize( (streams)[SIMULATION_STREAM] );
#ifdef V_CON
  CHECK_CUBLAS( cublasZdotc(cublas_handle, NX*NY*NZ, complex_arr1_dev, 1, complex_arr3_dev, 1, meanZ_Vcon_dev) ); // count <\psi||T \psi>
#endif
  cudaDeviceSynchronize();
  
  // count <Vdip> and <Vcon> in parallel
#ifdef V_DIP
  
#endif
#ifdef V_CON
  HANDLE_ERROR( cudaMemcpyAsync(&meanZ_Vcon_host, meanZ_Vcon_dev, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, (streams)[HELPER_STREAM]) ); // copy <Vcon> in parallel
#endif
  cudaStreamSynchronize( (streams)[SIMULATION_STREAM] );
#ifdef V_DIP
  //CHECK_CUBLAS( cublasZdotc(cublas_handle, NX*NY*NZ, complex_arr1_dev, 1, complex_arr3_dev, 1, meanZ_Vdip_dev) ); // count <\psi||T \psi>
#endif  
  
  
  
  
  //cudaDeviceSynchronize(); <- not necessary
  norm_host *= sqrt(DX);
#ifdef IMAG_TIME
  chemical_potential_host = chemical_potential_ite(norm_host);
#else
  chemical_potential_host = 0.;
#endif
  //mean_T_host = creal(meanZ_T_host);
  meanZ_T_host *= (0.5*DX/NX) + I*0.;
#ifdef V_EXT
  meanZ_Vext_host *= (0.5*DX*OMEGA*OMEGA) + I*0.;
#else
  meanZ_Vext_host = 0. + I*0.;
#endif
#ifdef V_CON
  meanZ_Vcon_host *= (0.5*DX);
#else
  meanZ_Vcon_host = 0. + I*0.;
#endif
#ifdef V_DIP
  
#else
  meanZ_Vdip_host = 0. + I*0.;
#endif
  
  double Energy_tot = creal(meanZ_T_host) + creal(meanZ_Vext_host);
#ifdef DEBUG
  printf("T:\t%.15f + %.15fj\n",creal(meanZ_T_host), cimag(meanZ_T_host));
  printf("Vext:\t%.15f + %.15fj\n",creal(meanZ_Vext_host), cimag(meanZ_Vext_host));
  printf("Etot:\t%.15f\n",Energy_tot);
#endif
  
  fprintf( ((files[STATS_FILE])->data), "%lu.\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\n", step_index,
								    norm_host,
                                                                    chemical_potential_host,
								    creal(meanZ_T_host),
								    creal(meanZ_Vext_host),
                                                                    creal(meanZ_Vcon_host),
                                                                    creal(mean_Vdip_host));
}


/*
 * Saves simulations parameters to special file.
 * 
 * !!! CURRENTLY INVOKED IN HELPER THREAD AFTER 2ND BARRIER !!!
 */
void save_simulation_params() {
    char str_date[17];
    time_t t = time(NULL);
    strftime(str_date, sizeof(str_date), "%Y-%m-%d_%H:%M", localtime(&t)); 
    
    //fprintf( (files[SIM_PARAMS_FILE])->data,"\t\t\tSIMULATION GPE\n");
    fprintf( (files[SIM_PARAMS_FILE])->data,"date:\t%s\n",str_date);
    uint8_t dim = DIM;
    fprintf( (files[SIM_PARAMS_FILE])->data,"dim:\t%u\n", dim );
#ifdef IMAG_TIME
    fprintf( (files[SIM_PARAMS_FILE])->data, "evolution:\t%s\n","imaginary time" );
#else
    fprintf( (files[SIM_PARAMS_FILE])->data, "evolution:\t%s\n","real time" );
#endif
#ifdef V_EXT
    fprintf( (files[SIM_PARAMS_FILE])->data, "Vext:\t%s\n","yes" );
#else
    fprintf( (files[SIM_PARAMS_FILE])->data, "Vext:\t%s\n","no" );
#endif
#ifdef V_CON
    fprintf( (files[SIM_PARAMS_FILE])->data, "Vcon:\t%s\n","yes" );
#else
    fprintf( (files[SIM_PARAMS_FILE])->data, "Vcon:\t%s\n","no" );
#endif
#ifdef V_DIP
    fprintf( (files[SIM_PARAMS_FILE])->data, "Vdip:\t%s\n","yes" );
#else
    fprintf( (files[SIM_PARAMS_FILE])->data, "Vdip:\t%s\n","no" );
#endif
    fprintf( (files[SIM_PARAMS_FILE])->data,"timesteps made:\t%u\n", time_tot );
    fprintf( (files[SIM_PARAMS_FILE])->data,"dt:\t%.15f\n", DT );
    fprintf( (files[SIM_PARAMS_FILE])->data,"frames:\t%u\n", frames_to_be_saved );
    fprintf( (files[SIM_PARAMS_FILE])->data,"Nx:\t%u\n", NX );
    fprintf( (files[SIM_PARAMS_FILE])->data,"Ny:\t%u\n", NY );
    fprintf( (files[SIM_PARAMS_FILE])->data,"Nz:\t%u\n", NZ );
    fprintf( (files[SIM_PARAMS_FILE])->data,"N:\t%u\n", NX*NY*NZ );
    fprintf( (files[SIM_PARAMS_FILE])->data,"xmin:\t%.15f\n", XMIN );
    fprintf( (files[SIM_PARAMS_FILE])->data,"xmax:\t%.15f\n", XMAX );
    fprintf( (files[SIM_PARAMS_FILE])->data,"dx:\t%.15f\n",DX );
    fprintf( (files[SIM_PARAMS_FILE])->data,"ymin:\t%.15f\n", 0. );
    fprintf( (files[SIM_PARAMS_FILE])->data,"ymax:\t%.15f\n", 0. );
    fprintf( (files[SIM_PARAMS_FILE])->data,"dy: %.15f\n",0. );
    fprintf( (files[SIM_PARAMS_FILE])->data,"zmin:\t%.15f\n", 0. );
    fprintf( (files[SIM_PARAMS_FILE])->data,"zmax:\t%.15f\n", 0. );
    fprintf( (files[SIM_PARAMS_FILE])->data,"dz: %.15f\n",0. );
    fprintf( (files[SIM_PARAMS_FILE])->data,"kx_min:\t%.15f\n", KxMIN );
    fprintf( (files[SIM_PARAMS_FILE])->data,"kx_max:\t%.15f\n", KxMAX );
    fprintf( (files[SIM_PARAMS_FILE])->data,"dkx:\t%.15f\n",DKx );
    //printf("width of gauss in positions space (points on lattice): %.15f\n");
    //printf("width of gauss in positions space (points on lattice): %.15f\n");
#ifdef OMEGA
    fprintf( (files[SIM_PARAMS_FILE])->data,"harmonic potential angular freq.:\t%.15f\n", OMEGA );
#endif
#ifdef G_CONTACT
    fprintf( (files[SIM_PARAMS_FILE])->data,"contact interactions g factor:\t%.15f\n", G_CONTACT );
#endif
#ifdef G_DIPOLAR
    fprintf( (files[SIM_PARAMS_FILE])->data,"dipolar interactions g factor:\t%.15f\n", G_DIPOLAR );
#endif
    
    // here some more ...
}