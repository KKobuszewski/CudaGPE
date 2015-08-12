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

#include "global.h"
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
 * 							FUNC DEFINITIONS									 *
 * 																	 *
 * ************************************************************************************************************************************* */

void save_stats(uint64_t step_index);



/* ************************************************************************************************************************************* *
 * 																	 *
 * 							SIM VARIABLES									 *
 * 																	 *
 * ************************************************************************************************************************************* */

// data structures on host
double complex* wf_host;
double complex* propagator_T_host;
double complex* propagator_Vext_host;
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
  
  // allocate memory on host
  cudaHostAlloc((void**) &(wf_host), sizeof(double complex)*NX*NY*NZ, cudaHostAllocDefault); // pinnable memory <- check here for cudaMallocHost (could be faster)
  cudaHostAlloc((void**) &norm_host, sizeof(double), cudaHostAllocDefault); // pinnable memory <- check here for cudaMallocHost (could be faster)
  cudaHostAlloc((void**) &propagator_T_host, sizeof(double complex)*NX*NY*NZ, cudaHostAllocDefault); // pinnable memory <- check here for cudaMallocHost
  cudaHostAlloc((void**) &propagator_Vext_host, sizeof(double complex)*NX*NY*NZ, cudaHostAllocDefault); // pinnable memory <- check here for cudaMallocHost (could be faster)
  cudaHostAlloc((void**) &Vdd_host, sizeof(double)*NX*NY*NZ, cudaHostAllocDefault);
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
    call_kernel_Z_1d( ker_gauss_1d, complex_arr1_dev, (streams)[SIMULATION_STREAM] );
    
  }
  
  // copying device pointers to host memory (to make them callable in kernels)
  //cudaMemcpyFromSymbol(&operator_T_h_ptr, operator_T_dev_ptr, sizeof(dev_funcZ_ptr_t));
  //cudaMemcpyFromSymbol(&operator_Vext_h_ptr, operator_Vext_dev_ptr, sizeof(dev_funcZ_ptr_t));
  
  printf("creating propagator T\n");
  //call_kernel_Z_1d( ker_create_propagator_T, propagator_T_dev, (streams)[HELPER_STREAM] );
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
    //Vdd_host[ii] = 
  }
  
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
  
#ifdef DEBUG
  /*
  HANDLE_ERROR( cudaMemcpyAsync(propagator_T_host, propagator_T_dev,
				NX*NY*NZ*sizeof(cuDoubleComplex),
				cudaMemcpyDeviceToHost,
				(streams)[HELPER_STREAM]) );
				*/
#endif
  
  // saving to file initial wavefuntion (1st frame) <- CZY TO JEST POTRZEBNE ???
  /*for (uint64_t ii=0 ; ii < NX*NY*NZ; ii++)
         //fprintf( (files[WF_FRAMES_FILE])->data, "%.15f\t%.15f\t%.15f\n", XMIN+DX*ii, creal((wf_host)[ii]), cimag((wf_host)[ii]) );*/
  fwrite( wf_host, sizeof(double complex), NX*NY*NZ, (files[WF_FRAMES_FILE])->data );
  
  // saving to file propagators T, Vext, and F{ Vdd }
  for (uint64_t ii=0 ; ii < NX*NY*NZ; ii++) {
         fprintf( (files[PROPAGATORS_FILE])->data, "%.15f\t%.15f\t%.15f\n", XMIN + ii*DX, creal(propagator_Vext_host[ii]), cimag(propagator_Vext_host[ii]),
                                                                            kx(ii), creal(propagator_T_host[ii]), cimag(propagator_T_host[ii]),
                                                                                    Vdd_host);
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
  //call_kernel_Z_1d( ker_print_Z, complex_arr1_dev, (streams)[SIMULATION_STREAM] );
#endif
  // checking norm of initial wavefunction
  CHECK_CUBLAS( cublasDznrm2( cublas_handle, NX*NY*NZ, complex_arr1_dev, 1, norm_dev) );
  //call_kernel_ZD_1d( ker_count_norm_wf_1d, complex_arr1_dev, norm_dev,  (streams)[SIMULATION_STREAM], NX*NY*NZ*sizeof(double) );
  
  cudaDeviceSynchronize();
  printf("pointers: \n%p\n%p\n%p\n",&norm_host,norm_dev,(streams)+SIMULATION_STREAM);
  HANDLE_ERROR( cudaMemcpyAsync(&norm_host, norm_dev,
        			sizeof(double),
				cudaMemcpyDeviceToHost,
				(streams)[HELPER_STREAM]) );
  cudaDeviceSynchronize();
  norm_host *= sqrt(DX);
  fprintf( (files[STATS_FILE])->data, "norm of initial wf: %.15f\tdx: %.15f\tsqrt dx: %.15f\n\n", norm_host, DX, sqrt(DX) );
  
  // checking total norm of propagator T (should be sqrt[Nx*|e^ia|^2] = sqrt[Nx], because Im[a] == 0 )
  CHECK_CUBLAS( cublasDznrm2( cublas_handle, NX*NY*NZ, propagator_T_dev, 1, norm_dev) );
  cudaDeviceSynchronize();
  HANDLE_ERROR( cudaMemcpyAsync(&norm_host, norm_dev,
				sizeof(double),
				cudaMemcpyDeviceToHost,
				(streams)[HELPER_STREAM]) );
  cudaDeviceSynchronize();
  fprintf( (files[STATS_FILE])->data, "norm (cublas) propagator_T_dev: %.15f\n", norm_host );
       
  // header of a file <- DO FILEIO.C PRZENIESC!
  fprintf( (files[STATS_FILE])->data, "\nt [dt]:\tnorm:\t\t\tchemic. pot.\t\t<T>\t\t\t<Vext>\t\t\t<Vcon>\t\t\t<Vdip>\n" );
  
  
  // start algorithm
  // dt =
  //const uint64_t time_tot = llround(0.318309886183791/DT); // no Vext revival time
#ifdef IMAG_TIME
  const uint64_t time_tot = 100000;
#else
  const uint64_t time_tot = 10*llround((2*3.14159265358979323846/OMEGA)/DT); // harmonic potential revival time
#endif
  const uint64_t saving_tot = 20;
  uint64_t saving_steps = saving_tot;
  uint64_t timesteps_tot = time_tot/saving_tot;
  uint64_t timesteps;
  
  printf("\n\n");
  printf("dt: %e\n",DT);
  printf("time total (in dt): %lu\n", time_tot);
  printf("time between saving (in dt): %lu\n", timesteps_tot);
  printf("savings: %lu\n", saving_tot);
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
       //printf("main algorithm\n");
       /* *************************************************************************************************************************************** *
	* 																	  *
	* 							ALGORITHM STEP									  *
	* 																	  *
	* *************************************************************************************************************************************** */
       // multiply by Vext propagator (do in callback load) !*
       
       // make copy of wavefunction
       //CHECK_CUBLAS( cublasZcopy(cublas_handle, NX*NY*NZ, complex_arr1_dev, 1, complex_arr3_dev, 1) );
       
       
       
       /*
        *       EVOLVE IN MOMENTUM SPACE
        */
       
       //printf("\ntransforming wavefunction to momentum space\n");
       CHECK_CUFFT( cufftExecZ2Z((cufft_plans)[FORWARD_PSI],
				 complex_arr1_dev,
				 complex_arr2_dev,
				 CUFFT_FORWARD) );
       
       // count |\psi|^2 array in meanwhile
       cudaStreamSynchronize(streams[SIMULATION_STREAM]);
       call_kernel_ZD_1d( ker_modulus_pow2_wf_1d, complex_arr3_dev, double_arr1_dev, streams[HELPER_STREAM]);
       // it could be replaced with complex_arr3_dev <- maybe faster to copy array with cublas and do not synchronize streams 
       
            // multiply by T propagator (do in callback) <- J
       call_kernel_ZZ_1d( ker_propagate, complex_arr2_dev, propagator_T_dev, (streams)[SIMULATION_STREAM] );
       
       
#ifdef DEBUG
       
       // count norm using own function
       //call_kernel_ZD_1d( ker_count_norm_wf_1d, complex_arr2_dev, norm_dev,  (streams)[SIMULATION_STREAM], 1024*sizeof(cuDoubleComplex) );
       
       //count norm using CUBLAS       
       //CHECK_CUBLAS( cublasDznrm2( cublas_handle, NX*NY*NZ, propagator_T_dev, 1, norm_dev) );
       
       
       // saving after fft
       /*HANDLE_ERROR( cudaMemcpy(wf_host, complex_arr2_dev, NX*NY*NZ*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );
       
       for (uint64_t ii=0 ; ii < NX*NY*NZ/2; ii++)
         fprintf( (files[])->data[1], "%.15f\t%.15f\t%.15f\n", DKx*ii, creal((wf_host)[ii]), cimag((wf_host)[ii]) );
       for (uint64_t ii = NX*NY*NZ/2 ; ii < NX*NY*NZ; ii++)
         fprintf( (files[])->data[1], "%.15f\t%.15f\t%.15f\n", 2*KxMIN + DKx*ii, creal((wf_host)[ii]), cimag((wf_host)[ii]) );
       */
#endif
       
       
       // go back to 'positions`'? space <- JAK JEST PO ANGIELSKU PRZESTRZEN POLOZEN ???
       cudaStreamSynchronize(streams[HELPER_STREAM]); // ensure that complex 
       CHECK_CUFFT( cufftExecZ2Z((cufft_plans)[BACKWARD_PSI],
				 complex_arr2_dev,
				 complex_arr1_dev,
				 CUFFT_INVERSE) );
       // run kernel to normalize aftter FFT
       call_kernel_Z_1d( ker_normalize_1d, complex_arr1_dev, (streams)[SIMULATION_STREAM] );
       
       
       
       // count DFT of modulus of wavefunction (in positions` space)
       CHECK_CUFFT( cufftExecD2Z((cufft_plans)[FORWARD_DIPOLAR],
				 double_arr1_dev,
				 complex_arr3_dev) ); // double to complex must be forward, so no need to specify direction
       
       
       /*
        *       EVOLVE IN POSITIONS` SPACE
        */
       
       // evolve via external potential (if defined)
#ifdef V_EXT
       call_kernel_ZZ_1d( ker_propagate, complex_arr1_dev, propagator_Vext_dev, (streams)[SIMULATION_STREAM] );
#endif
       
       // evolve via contact interactions potential
       cudaStreamSynchronize(streams[HELPER_STREAM]);
       
       
       // count integral in potential of dipolar interactions <- in callback load 
       CHECK_CUFFT( cufftExecZ2Z((cufft_plans)[BACKWARD_DIPOLAR],
				 complex_arr3_dev,
				 complex_arr3_dev,
				 CUFFT_INVERSE) );
       // normalize (in callback store
       
       // create propagator of Vdip (in)
       
       
#ifdef IMAG_TIME
       CHECK_CUBLAS( cublasDznrm2( cublas_handle, NX*NY*NZ, complex_arr1_dev, 1, norm_dev) );
       cudaDeviceSynchronize();
       call_kernel_ZD_1d( ker_normalize_1d, complex_arr1_dev, norm_dev,  (streams)[SIMULATION_STREAM] );
#endif
       
       /* *************************************************************************************************************************************** *
	* 																	  *
	* 							END OF ALGORITHM STEP								  *
	* 																	  *
	* *************************************************************************************************************************************** */
       
       
       
       // compute and save statistics of a system (norm, energy, ... )
#ifndef DEBUG
       if ( (counter%500) == 0 ) {
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
       
       //save_stats(timesteps_tot-timesteps)*(saving_tot-saving_steps)
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
  /// free memory on device
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
  //HANDLE_ERROR( cudaMalloc((void**) &(mean_Vdip_dev), sizeof(double)) ); // result of integral with Vdip operator in positions' representation
  HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->mean_Vext_dev), sizeof(double)) ); // result of integral with Vext operator in positions' representation
  HANDLE_ERROR( cudaMalloc((void**) &meanZ_Vext_dev, sizeof(cuDoubleComplex))    ); // result of integral with kinetic energy operator in momentum representaion
  //HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->mean_Vcon_dev), sizeof(double)) ); // result of integral with Vcon operator in positions' representation
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




void save_stats(uint64_t step_index) {
  // saving stats in pipelining mode
  
  // count norm with cublas (it is already done in case of imaginary-time evolution)
#ifndef IMAG_TIME
  CHECK_CUBLAS( cublasDznrm2( cublas_handle, NX*NY*NZ, complex_arr1_dev, 1, norm_dev) );
#endif
  
  // count <T> and copy norm in parallel
  call_kernel_ZZ_1d( ker_T_wf, complex_arr2_dev, complex_arr3_dev, (streams)[SIMULATION_STREAM]);
  HANDLE_ERROR( cudaMemcpyAsync(&norm_host, norm_dev,
                                    sizeof(double),
	                            cudaMemcpyDeviceToHost,
	                            (streams)[HELPER_STREAM]) );
  cudaStreamSynchronize( (streams)[SIMULATION_STREAM] );
  CHECK_CUBLAS( cublasZdotc(cublas_handle, NX*NY*NZ, (complex_arr2_dev), 1, complex_arr3_dev, 1, meanZ_T_dev) );
  cudaDeviceSynchronize();
  
  // count <Vext> and copy <T> in parallel
  call_kernel_ZZ_1d( ker_Vext_wf, complex_arr1_dev, complex_arr3_dev, (streams)[SIMULATION_STREAM]);
  HANDLE_ERROR( cudaMemcpyAsync(&meanZ_T_host, meanZ_T_dev,
                                    sizeof(cuDoubleComplex),
	                            cudaMemcpyDeviceToHost,
	                            (streams)[HELPER_STREAM]) );
  cudaStreamSynchronize( (streams)[SIMULATION_STREAM] );
  CHECK_CUBLAS( cublasZdotc(cublas_handle, NX*NY*NZ, complex_arr1_dev, 1, complex_arr3_dev, 1, meanZ_Vext_dev) );
  cudaDeviceSynchronize();
  
  // copy <Vext>
  HANDLE_ERROR( cudaMemcpyAsync(&meanZ_Vext_host, meanZ_Vext_dev,
                                    sizeof(cuDoubleComplex),
	                            cudaMemcpyDeviceToHost,
	                            (streams)[HELPER_STREAM]) );
  
  cudaDeviceSynchronize();
  norm_host *= sqrt(DX);
  chemical_potential_host = 0.;
  //mean_T_host = creal(meanZ_T_host);
  meanZ_T_host *= (0.5*DX/NX) + I*0.;
  meanZ_Vext_host *= (0.5*DX*OMEGA*OMEGA) + I*0.;
  
  double Energy_tot = creal(meanZ_T_host) + creal(meanZ_Vext_host);
#ifdef DEBUG
  printf("T:\t%.15f + %.15fj\n",creal(meanZ_T_host), cimag(meanZ_T_host));
  printf("Vext:\t%.15f + %.15fj\n",creal(meanZ_Vext_host), cimag(meanZ_Vext_host));
  printf("Etot:\t%.15f\n",Energy_tot);
#endif
  
  fprintf( ((files[STATS_FILE])->data), "%lu.\t%.15f\t%.15f\t%.15f\t%.15f\n", step_index,
								    norm_host,
                                                                    chemical_potential_host,
								    creal(meanZ_T_host),
								    creal(meanZ_Vext_host) );
}