#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <math.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cublas_v2.h>
#include <cuda.h>

#include "global.h"
#include "simulation.cuh"
#include "cudautils.cuh"
#include "kernels.cuh"


// timing
cudaEvent_t start_t;
cudaEvent_t stop_t;


// global variables
extern Globals* global_stuff;
bool FLAG_RUN_SIMULATION = true;
extern const char* thread_names[];
extern const char* stream_names[];
//extern pthread_barrier_t barrier;


pthread_barrier_t barrier;
cublasHandle_t cublas_handle;

/*
 * 
 * !!! VERSION FOR 1 PTHREAD !!!
 * 
 */


/* ************************************************************************************************************************************* *
 * 																	 *
 * 							SIM THREAD									 *
 * 																	 *
 * ************************************************************************************************************************************* */

/*
 * - allocation memory on host
 * - initialization of data
 * - main algorithm
 */
void* simulation_thread(void* passing_ptr) {
  
  double complex* propagator_T_host;
  double complex* propagator_Vext_host;
  double norm_host;
  
  //stick_this_thread_to_core(1); <- in cudautils, not used, include to header first
  pthread_barrier_wait (&barrier_global);
  printf("running %s thread.\n",thread_names[SIMULATION_THRD]);
  
  // allocate memory on host
  cudaHostAlloc((void**) &(global_stuff->wf_host), sizeof(double complex)*NX*NY*NZ, cudaHostAllocDefault); // pinnable memory <- check here for cudaMallocHost (could be faster)
  cudaHostAlloc((void**) &norm_host, sizeof(double), cudaHostAllocDefault); // pinnable memory <- check here for cudaMallocHost (could be faster)
  cudaHostAlloc((void**) &propagator_T_host, sizeof(double complex)*NX*NY*NZ, cudaHostAllocDefault); // pinnable memory <- check here for cudaMallocHost
  cudaHostAlloc((void**) &propagator_Vext_host, sizeof(double complex)*NX*NY*NZ, cudaHostAllocDefault); // pinnable memory <- check here for cudaMallocHost (could be faster)
  printf("allocated memory on host.\n");
  
  
  // fill arrays on host & device
//   if (global_stuff->init_wf_fd != -1) {
//     for (uint64_t ii = 0; ii < NX*NY*NZ; ii++) {
//       global_stuff->wf_host[ii] = global_stuff->init_wf_map[ii];
//     }
//   }
  
  
#ifdef DEBUG
  printf("1st barrier reached by %s.\n",thread_names[SIMULATION_THRD]);
#endif
  pthread_barrier_wait (&barrier);
  // copy data async from host to device (if needed)
  if (global_stuff->init_wf_fd != -1) {
    // copy data from host to device (if needed) / cannot async because
    printf("copying initial wavefunction on device");
    HANDLE_ERROR( cudaMemcpy(global_stuff->complex_arr1_dev, global_stuff->init_wf_map, NX*NY*NZ * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice) ); // change to asynchronous!
  }
  else {
    
    printf("initating wavefunction on device.\n");
    call_kernel_Z_1d( ker_gauss_1d, global_stuff->complex_arr1_dev, (global_stuff->streams)[SIMULATION_STREAM] );
    
  }
  
  printf("creating propagator T\n");
  //call_kernel_Z_1d( ker_create_propagator_T, global_stuff->propagator_T_dev, (global_stuff->streams)[HELPER_STREAM] );
  for( uint64_t ii=0; ii < NX; ii++ ) {
    propagator_T_host[ii] = cexp(-I*kx(ii)*(0.5*kx(ii)*DT));
    propagator_Vext_host[ii] = cexp(-I*(0.5*OMEGA)*(OMEGA*(ii*DX+XMIN))*((ii*DX+XMIN)*DT)); // <- !!! KOLEJNOSC MNOZEMIA A DOKLADNOSC !!!
    //printf("%.15f + %.15fj\n",creal(propagator_Vext_host[ii]), cimag(propagator_Vext_host[ii]) );
  }
  
  // copying propag T to dev
  HANDLE_ERROR( cudaMemcpyAsync(global_stuff->propagator_T_dev, propagator_T_host,
				NX*NY*NZ*sizeof(cufftDoubleComplex),
				cudaMemcpyHostToDevice,
				(global_stuff->streams)[HELPER_STREAM]) );
  
  // copying after initialization (in meantime on another stream)
  HANDLE_ERROR( cudaMemcpyAsync(global_stuff->wf_host, global_stuff->complex_arr1_dev,
				NX*NY*NZ*sizeof(cufftDoubleComplex),
				cudaMemcpyDeviceToHost,
				(global_stuff->streams)[SIMULATION_STREAM]) );
  cudaDeviceSynchronize();
  
  
  // copying propag Vext to dev
  HANDLE_ERROR( cudaMemcpyAsync(global_stuff->propagator_Vext_dev, propagator_Vext_host,
				NX*NY*NZ*sizeof(cufftDoubleComplex),
				cudaMemcpyHostToDevice,
				(global_stuff->streams)[SIMULATION_STREAM]) );
  
#ifdef DEBUG
  HANDLE_ERROR( cudaMemcpyAsync(propagator_T_host, global_stuff->propagator_T_dev,
				NX*NY*NZ*sizeof(cufftDoubleComplex),
				cudaMemcpyDeviceToHost,
				(global_stuff->streams)[HELPER_STREAM]) );
  
  
#endif
  
  // saving to file after initialization
  for (uint64_t ii=0 ; ii < NX*NY*NZ; ii++)
         fprintf( (global_stuff->files)[0], "%.15f\t%.15f\t%.15f\n", XMIN+DX*ii, creal((global_stuff->wf_host)[ii]), cimag((global_stuff->wf_host)[ii]) );
  for (uint64_t ii=NX*NY*NZ/2 ; ii < NX*NY*NZ; ii++)
         fprintf( (global_stuff->files)[4], "%.15f\t%.15f\t%.15f\n", kx(ii), creal(propagator_T_host[ii]), cimag(propagator_T_host[ii]) );
  for (uint64_t ii=0 ; ii < NX*NY*NZ/2; ii++)
         fprintf( (global_stuff->files)[4], "%.15f\t%.15f\t%.15f\n", kx(ii), creal(propagator_T_host[ii]), cimag(propagator_T_host[ii]) );
  
#ifdef DEBUG
  
  printf("2nd barrier reached by %s.\n",thread_names[SIMULATION_THRD]);
  printf("FLAG_RUN_SIMULATION %u\n",FLAG_RUN_SIMULATION);
#endif
  cudaStreamSynchronize( (global_stuff->streams)[HELPER_STREAM] );
  pthread_barrier_wait (&barrier);
  
  
#ifdef DEBUG     
  call_kernel_Z_1d( ker_print_Z, global_stuff->complex_arr1_dev, (global_stuff->streams)[SIMULATION_STREAM] );
#endif
  // checking norm of initial wavefunction
  CHECK_CUBLAS( cublasDznrm2( cublas_handle, NX*NY*NZ, global_stuff->complex_arr1_dev, 1, global_stuff->norm_dev) );
  cudaDeviceSynchronize();
  
  HANDLE_ERROR( cudaMemcpyAsync(&norm_host, global_stuff->norm_dev,
        			sizeof(double),
				cudaMemcpyDeviceToHost,
				(global_stuff->streams)[HELPER_STREAM]) );
  cudaDeviceSynchronize();
  norm_host *= sqrt(DX);
  fprintf( (global_stuff->files)[3], "norm of initial wf: %.15f\tdx: %.15f\tsqrt dx: %.15f\n\n", norm_host, DX, sqrt(DX) );
  
  // checking total norm of propagator T (should be sqrt[Nx*|e^ia|^2] = sqrt[Nx], because Im[a] == 0 )
  CHECK_CUBLAS( cublasDznrm2( cublas_handle, NX*NY*NZ, global_stuff->propagator_T_dev, 1, global_stuff->norm_dev) );
  cudaDeviceSynchronize();
  HANDLE_ERROR( cudaMemcpyAsync(&norm_host, global_stuff->norm_dev,
				sizeof(double),
				cudaMemcpyDeviceToHost,
				(global_stuff->streams)[HELPER_STREAM]) );
  cudaDeviceSynchronize();
  fprintf( (global_stuff->files)[3], "norm (cublas) propagator_T_dev: %.15f\n", norm_host );
       
  // header of a file <- DO FILEIO.C PRZENIESC!
  fprintf( (global_stuff->files)[3], "\ntimestep:\tnorm after (cublas):\n" );
  
  
  // start algorithm
  // dt =
  //const uint64_t time_tot = llround(0.318309886183791/DT); // no Vext revival time
  const uint64_t time_tot = 10*llround((4*3.14159265358979323846/OMEGA)/DT); // harmonic potential revival time
  const uint64_t saving_tot = 20;
  uint64_t saving_steps = saving_tot;
  uint64_t timesteps_tot = time_tot/saving_tot;
  uint64_t timesteps;
  
  printf("\n\n");
  printf("time total (in dt): %lu\n", time_tot);
  printf("time between saving (in dt): %lu\n", timesteps_tot);
  printf("savings: %lu\n", saving_tot);
  printf("\n");
  
  while( FLAG_RUN_SIMULATION ) { // simulation will be runing until the flag is set to false
#ifdef DEBUG
     timesteps = 1;
     printf("timesteps to be made: %u\n", timesteps);
#else
     timesteps = timesteps_tot;
     printf("%lu. timesteps to be made: %lu\n", saving_steps, timesteps);
#endif
     uint32_t counter = 0;
     saving_steps--;
     
     while(timesteps) {
       timesteps--;
       //printf("main algorithm\n");
       /* *************************************************************************************************************************************** *
	* 																	  *
	* 							ALGORITHM STEP									  *
	* 																	  *
	* *************************************************************************************************************************************** */
       // multiply by Vext propagator (do in callback load) !
       
       // go to momentum space
       //printf("\ntransforming wavefunction to momentum space\n");
       CHECK_CUFFT( cufftExecZ2Z((global_stuff->plans)[FORWARD_PSI],
				 global_stuff->complex_arr1_dev,
				 global_stuff->complex_arr2_dev,
				 CUFFT_FORWARD) );
       
       // multiply by T propagator (do in callback) <- ALE KTORY store od FORWARD czy load od INVERSE
       call_kernel_ZZ_1d( ker_propagate, global_stuff->complex_arr2_dev, global_stuff->propagator_T_dev, (global_stuff->streams)[SIMULATION_STREAM] );
       
       
       // count norm using own function
       //call_kernel_ZD_1d( ker_count_norm_wf_1d, global_stuff->complex_arr2_dev, global_stuff->norm_dev,  (global_stuff->streams)[SIMULATION_STREAM], 1024*sizeof(cuDoubleComplex) );
       
       //count norm using CUBLAS       
       //CHECK_CUBLAS( cublasDznrm2( cublas_handle, NX*NY*NZ, global_stuff->propagator_T_dev, 1, global_stuff->norm_dev) );
       
       
#ifdef DEBUG
       // saving after fft
       HANDLE_ERROR( cudaMemcpy(global_stuff->wf_host, global_stuff->complex_arr2_dev, NX*NY*NZ*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost) );
       
       for (uint64_t ii=0 ; ii < NX*NY*NZ/2; ii++)
         fprintf( (global_stuff->files)[1], "%.15f\t%.15f\t%.15f\n", DKx*ii, creal((global_stuff->wf_host)[ii]), cimag((global_stuff->wf_host)[ii]) );
       for (uint64_t ii = NX*NY*NZ/2 ; ii < NX*NY*NZ; ii++)
         fprintf( (global_stuff->files)[1], "%.15f\t%.15f\t%.15f\n", 2*KxMIN + DKx*ii, creal((global_stuff->wf_host)[ii]), cimag((global_stuff->wf_host)[ii]) );
       
#endif
             
       
       // go back to 'positions`'? space <- JAK JEST PO ANGIELSKU PRZESTRZEN POLOZEN ???
       CHECK_CUFFT( cufftExecZ2Z((global_stuff->plans)[BACKWARD_PSI],
				 global_stuff->complex_arr2_dev,
				 global_stuff->complex_arr1_dev,
				 CUFFT_INVERSE) );
       
       // run kernel to normalize aftter FFT
       call_kernel_Z_1d( ker_normalize_1d, global_stuff->complex_arr1_dev, (global_stuff->streams)[SIMULATION_STREAM] );
       
       
       // evolve in positions` space
       call_kernel_ZZ_1d( ker_propagate, global_stuff->complex_arr1_dev, global_stuff->propagator_Vext_dev, (global_stuff->streams)[SIMULATION_STREAM] );
       
       
       /*
       // count DFT of modulus of wavefunction (in positions` space)
       CHECK_CUFFT( cufftExecD2Z((global_stuff->plans)[FORWARD_DIPOLAR],
				 global_stuff->double_arr1_dev,
				 global_stuff->complex_arr2_dev) ); // double to complex must be forward, so no need to specify direction
       
       
       
       // count integral in potential of dipolar interactions
       CHECK_CUFFT( cufftExecZ2Z((global_stuff->plans)[BACKWARD_DIPOLAR],
				 global_stuff->complex_arr2_dev,
				 global_stuff->complex_arr2_dev,
				 CUFFT_INVERSE) );
       // normalize (in callback store
       
       // create propagator of Vdip (in)
       */
       
       /* *************************************************************************************************************************************** *
	* 																	  *
	* 							END OF ALGORITHM STEP								  *
	* 																	  *
	* *************************************************************************************************************************************** */
       
       // check norm
       if ( counter == 10000 ) {
	  CHECK_CUBLAS( cublasDznrm2( cublas_handle, NX*NY*NZ, global_stuff->complex_arr1_dev, 1, global_stuff->norm_dev) );
	  cudaDeviceSynchronize();
	  
	  HANDLE_ERROR( cudaMemcpyAsync(&norm_host, global_stuff->norm_dev,
				    sizeof(double),
				    cudaMemcpyDeviceToHost,
				    (global_stuff->streams)[HELPER_STREAM]) );
	  cudaDeviceSynchronize();
	  norm_host *= sqrt(DX);
	  fprintf( (global_stuff->files)[3], "%lu.\t%.15f\n", (timesteps_tot-timesteps)*(saving_tot-saving_steps), norm_host );
	  counter = 0;
       }
       counter++;
       
     }
     
     
     
       // saving after ifft
       HANDLE_ERROR( cudaMemcpy(global_stuff->wf_host, global_stuff->complex_arr1_dev, NX*NY*NZ*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost) );
       for (uint64_t ii=0 ; ii < NX*NY*NZ; ii++)
         fprintf( (global_stuff->files)[2], "%.15f\t%.15f\t%.15f\n", XMIN+DX*ii, creal((global_stuff->wf_host)[ii]), cimag((global_stuff->wf_host)[ii]) );
#ifdef DEBUG     
       call_kernel_Z_1d( ker_print_Z, global_stuff->complex_arr1_dev, (global_stuff->streams)[SIMULATION_STREAM] );
#endif
     if (!saving_steps) FLAG_RUN_SIMULATION = false;
  }
//#ifdef DEBUG
//#endif
  
#ifdef DEBUG
  printf("last barrier reached by %s.\n",thread_names[SIMULATION_THRD]);
#endif
  cudaDeviceSynchronize();
  pthread_barrier_wait (&barrier);
  
  // free memory on host
  HANDLE_ERROR( cudaFreeHost(global_stuff->wf_host) );
#ifdef DEBUG
  HANDLE_ERROR( cudaFreeHost(propagator_T_host) );
#endif
  /*
  /// free memory on device
  HANDLE_ERROR( cudaFree(global_stuff->complex_arr1_dev) ); 	//
  HANDLE_ERROR( cudaFree(global_stuff->complex_arr2_dev) ); 	//
  //HANDLE_ERROR( cudaFree(global_stuff->double_arr1_dev)  ); 	//
  HANDLE_ERROR( cudaFree(global_stuff->propagator_T_dev) ); 	//
  //HANDLE_ERROR( cudaFree(global_stuff->propagator_Vext_dev) );	//
  //HANDLE_ERROR( cudaFree(global_stuff->Vdip_dev) );		//
  
  
  //HANDLE_ERROR( cudaFree(global_stuff->mean_T_dev) ); // result of integral with kinetic energy operator in momentum representaion
  //HANDLE_ERROR( cudaFree(global_stuff->mean_Vdip_dev) ); // result of integral with Vdip operator in positions' representation
  //HANDLE_ERROR( cudaFree(global_stuff->mean_Vext_dev) ); // result of integral with Vext operator in positions' representation
  //HANDLE_ERROR( cudaFree(global_stuff->mean_Vcon_dev) ); // result of integral with Vcon operator in positions' representation
  HANDLE_ERROR( cudaFree(global_stuff->norm_dev) ); //
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
  HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->complex_arr1_dev), sizeof(cuDoubleComplex) * NX*NY*NZ) ); 	//
  HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->complex_arr2_dev), sizeof(cuDoubleComplex) * NX*NY*NZ) ); 	//
  //HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->double_arr1_dev), sizeof(double) * NX*NY*NZ) );		//
  
  // constant arrays
  HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->propagator_T_dev), sizeof(cuDoubleComplex) * NX*NY*NZ) ); 	// array of constant factors e^-i*k**2/2*dt
  HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->propagator_Vext_dev), sizeof(cuDoubleComplex) * NX*NY*NZ) );// array of constant factors e^-i*Vext*dt
  //HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->Vdip_dev), sizeof(double) * NX*NY*NZ) ); 			// array of costant factors <- count on host with spec funcs lib or use Abramowitz & Stegun approximation
  
  // scalar variables
  //HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->mean_T_dev), sizeof(double))    ); // result of integral with kinetic energy operator in momentum representaion
  //HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->mean_Vdip_dev), sizeof(double)) ); // result of integral with Vdip operator in positions' representation
  //HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->mean_Vext_dev), sizeof(double)) ); // result of integral with Vext operator in positions' representation
  //HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->mean_Vcon_dev), sizeof(double)) ); // result of integral with Vcon operator in positions' representation
  HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->norm_dev), sizeof(double)) ); // variable to hold norm of wavefunction
  
  
#ifdef DEBUG
  printf("allocated memory on device.\n");
#endif
  
  for (uint8_t ii = 0; ii < num_streams; ii++)
    HANDLE_ERROR(	cudaStreamCreate( &(global_stuff->streams[ii]) )	);
  
#ifdef DEBUG
  printf("1st barrier reached by %s.\n",thread_names[HELPER_THRD]);
#endif
  //cudaDeviceSynchronize();
  pthread_barrier_wait (&barrier);
  
  // creating plans with callbacks
  global_stuff->plans = (cufftHandle*) malloc( (size_t) sizeof(cufftHandle)*num_plans );
#ifdef DEBUG
  printf("array of plans allocated.\n");
#endif
  for (uint8_t ii = 0; ii < num_plans; ii++) {
    CHECK_CUFFT(  cufftCreate( (global_stuff->plans)+ii )  ); // allocates expandable plans
    //printf("%d\n",(global_stuff->plans)[ii]);
  }
  
#ifdef DEBUG
  printf("expandable plans allocated.\n");
#endif
  
  size_t work_size; // CHYBA TO MUSI BYC TABLICA !!!
#if (DIM == 1)
  printf("creating CUFFT plans in 1d case.\n");
  // wavefunction forward
  // cufftMakePlan1d(plan, N, CUFFT_Z2Z, 1, &work_size);
  CHECK_CUFFT( cufftMakePlan1d( (global_stuff->plans)[FORWARD_PSI], NX*NY*NZ, CUFFT_Z2Z, 1, &work_size ) 	);
#ifdef DEBUG
  //pthread_barrier_wait (&barrier);
#endif
  // associate transform with specified stream
  
  // wavefunction forward
  CHECK_CUFFT( cufftSetStream(  (global_stuff->plans)[FORWARD_PSI], (global_stuff->streams)[SIMULATION_STREAM] ) );
  //printf("%d\n",(global_stuff->plans)[FORWARD_PSI]);
  
  // wavefunction inverse
  //  printf("%p\n",(global_stuff->plans)+BACKWARD_PSI);
  CHECK_CUFFT( cufftMakePlan1d( (global_stuff->plans)[BACKWARD_PSI], NX*NY*NZ, CUFFT_Z2Z, 1, &work_size )	);
  CHECK_CUFFT( cufftSetStream(  (global_stuff->plans)[BACKWARD_PSI], (global_stuff->streams)[SIMULATION_STREAM]) );
  //printf("%d\n",(global_stuff->plans)[BACKWARD_PSI]);
  
  // modulus of wavefunction forward
  CHECK_CUFFT( cufftMakePlan1d( (global_stuff->plans)[FORWARD_DIPOLAR], NX*NY*NZ, CUFFT_D2Z, 1, &work_size )	);
  CHECK_CUFFT( cufftSetStream(  (global_stuff->plans)[FORWARD_DIPOLAR], (global_stuff->streams)[HELPER_STREAM] ) );
  //printf("%d\n",(global_stuff->plans)[FORWARD_DIPOLAR]);
  
  // integral in potential of dipolar inteaction
  CHECK_CUFFT( cufftMakePlan1d( (global_stuff->plans)[BACKWARD_DIPOLAR], NX*NY*NZ, CUFFT_Z2Z, 1, &work_size )	);
  CHECK_CUFFT( cufftSetStream(  (global_stuff->plans)[BACKWARD_DIPOLAR], (global_stuff->streams)[HELPER_STREAM]) ); // WLASCIWIE TUTAJ NIE WIADOMO W KTORYM STREAMIE?
  //printf("%d\n",(global_stuff->plans)[BACKWARD_DIPOLAR]);
  
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
    HANDLE_ERROR(	cudaStreamDestroy( (global_stuff->streams[ii]) )	);
  
  pthread_barrier_wait (&barrier_global);
  pthread_exit(NULL);
}


void alloc_device(){
  
  
}


void alloc_host() {
  
  // must use
  
}
