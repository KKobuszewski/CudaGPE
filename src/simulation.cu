#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <math.h>
#include <cufft.h>
#include <cufftXt.h>
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
extern pthread_barrier_t barrier;


pthread_barrier_t barrier;

/*
 * 
 * !!! VERSION FOR 1 PTHREAD !!!
 * 
 */

void* simulation_thread(void* passing_ptr) {
  
  //stick_this_thread_to_core(1); <- in cudautils, not used, include to header first
  pthread_barrier_wait (&barrier_global);
  printf("running %s thread.\n",thread_names[SIMULATION_THRD]);
  
  // allocate memory on host
  cudaHostAlloc((void**) &(global_stuff->wf_host), sizeof(double complex)*NX*NY*NZ, cudaHostAllocDefault); // pinnable memory <- check here for cudaMallocHost (could be faster)
#ifdef DEBUG
  printf("allocated memory on device.\n");
#endif
  
  
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
    HANDLE_ERROR( cudaMemcpy(global_stuff->complex_arr1_dev, global_stuff->init_wf_map, NX*NY*NZ * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice) );
  }
  else {
    uint64_t threadsPerBlock;
    if (NX*NY*NZ >= 33554432)
      threadsPerBlock = 1024;
    else {
      threadsPerBlock = 128; // seems max grid size is ( 32768, ?, ? ) <- ????
    }
    dim3 dimBlock(threadsPerBlock,1,1);
    dim3 dimGrid( (NX*NY*NZ + threadsPerBlock - 1)/threadsPerBlock, 1, 1 ); // (numElements + threadsPerBlock - 1) / threadsPerBlock
#ifdef DEBUG
    printf("initating wavefunction on host. Kernel invocation:\n");
    printf("threads Per block: %lu\n", threadsPerBlock);
    printf("blocks: %lu\n",(NX*NY*NZ + threadsPerBlock - 1)/threadsPerBlock);
#endif
    // filling with data
    cudaGauss_1d<<<dimGrid,dimBlock,0,(global_stuff->streams)[SIMULATION_STREAM]>>>(global_stuff->complex_arr1_dev, NX*NY*NZ);
    HANDLE_ERROR( cudaGetLastError() );
  }
  
#ifdef DEBUG
  printf("2nd barrier reached by %s.\n",thread_names[SIMULATION_THRD]);
  printf("FLAG_RUN_SIMULATION %u\n",FLAG_RUN_SIMULATION);
#endif
  pthread_barrier_wait (&barrier);
  
  // start algorithm
  // dt =
  uint16_t timesteps;
  while( FLAG_RUN_SIMULATION ) { // simulation will be runing until the flag is set to false
#ifdef DEBUG
     timesteps = 10;
     printf("timesteps to be made: %u\n", timesteps);
#else
     timesteps = 1000;
#endif
     while(timesteps) {
       timesteps--;
       printf("main algorithm\n");
       // multiply by Vext propagator (do in callback load) !
       
       // go to momentum space
       CHECK_CUFFT( cufftExecZ2Z((global_stuff->plans)[FORWARD_PSI],
				 global_stuff->complex_arr1_dev,
				 global_stuff->complex_arr2_dev,
				 CUFFT_FORWARD) );
       
       // multiply by T propagator (do in callback) <- ALE KTORY store od FORWARD czy load od INVERSE
       
       // go back to 'positions`'? space <- JAK JEST PO ANGIELSKU PRZESTRZEN POLOZEN ???
       CHECK_CUFFT( cufftExecZ2Z((global_stuff->plans)[BACKWARD_PSI],
				 global_stuff->complex_arr2_dev,
				 global_stuff->complex_arr1_dev,
				 CUFFT_INVERSE) );
       
       
       
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
       
       
       FLAG_RUN_SIMULATION = false;
       
     }
     
     //FLAG_RUN_SIMULATION = false;
  }
  
  
#ifdef DEBUG
  printf("last barrier reached by %s.\n",thread_names[SIMULATION_THRD]);
#endif
  cudaDeviceSynchronize();
  pthread_barrier_wait (&barrier);
  
  // free memory on host
  HANDLE_ERROR( cudaFreeHost(global_stuff->wf_host) );
  
  
  pthread_barrier_wait (&barrier_global);
  pthread_exit(NULL);
}


void* helper_thread(void* passing_ptr) {
  
  //stick_this_thread_to_core(2);
  
  pthread_barrier_wait (&barrier_global);
  printf("running %s thread.\n",thread_names[HELPER_THRD]);
  
  // init memory on host & device
  HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->complex_arr1_dev), sizeof(cuDoubleComplex) * NX*NY*NZ) ); 	//
  HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->complex_arr2_dev), sizeof(cuDoubleComplex) * NX*NY*NZ) ); 	//
  HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->double_arr1_dev), sizeof(double) * NX*NY*NZ) );		//
  
  HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->propagator_T_dev), sizeof(cuDoubleComplex) * NX*NY*NZ) ); 	// array of constant factors e^-i*k**2/2*dt
  HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->propagator_Vext_dev), sizeof(cuDoubleComplex) * NX*NY*NZ) );// array of constant factors e^-i*Vext*dt
  HANDLE_ERROR( cudaMalloc((void**) &(global_stuff->Vdip_dev), sizeof(double) * NX*NY*NZ) ); 			// array of costant factors <- count on host with spec funcs lib or use Abramowitz & Stegun approximation
#ifdef DEBUG
  printf("allocated memory on device.\n");
#endif
  
  for (uint8_t ii = 0; ii < num_streams; ii++)
    HANDLE_ERROR(	cudaStreamCreate( &(global_stuff->streams[ii]) )	);
  
#ifdef DEBUG
  printf("1st barrier reached by %s.\n",thread_names[HELPER_THRD]);
#endif
  //cudaDeviceSynchronize();
#include <cufftXt.h>
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
  
#endif
  printf("\tplans created\n");
  
  // !!! SPRAWDZIC !!! funckje: <- co robia?
  //cufftResult cufftSetAutoAllocation(cufftHandle *plan, bool autoAllocate);
  //cufftSetCompatibilityMode() <- musi byc wywolana po create a przed make plan
  
#ifdef DEBUG
  printf("created FFT plans.\n");
#endif
#ifdef DEBUG
  printf("2nd barrier reached by %s.\n",thread_names[HELPER_THRD]);
#endif
  pthread_barrier_wait (&barrier);
  
  
  
  // start algorithm
  // dt =
  uint16_t timesteps;
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
  }
  
#ifdef DEBUG
  printf("last barrier reached by %s.\n",thread_names[HELPER_THRD]);
#endif
  cudaDeviceSynchronize();
  pthread_barrier_wait (&barrier);
  
  /// free memory on host & device
  HANDLE_ERROR( cudaFree( global_stuff->complex_arr1_dev ) ); 	//
  HANDLE_ERROR( cudaFree( global_stuff->complex_arr2_dev ) ); 	//
  HANDLE_ERROR( cudaFree( global_stuff->double_arr1_dev ) ); 	//
  HANDLE_ERROR( cudaFree( global_stuff->propagator_T_dev ) ); 	//
  HANDLE_ERROR( cudaFree( global_stuff->propagator_Vext_dev ) );//
  HANDLE_ERROR( cudaFree( global_stuff->Vdip_dev ) );		//
  
  for (uint8_t ii = 0; ii < num_streams; ii++)
    HANDLE_ERROR(	cudaStreamCreate( &(global_stuff->streams[ii]) )	);
  
  pthread_barrier_wait (&barrier_global);
  pthread_exit(NULL);
}


void alloc_device(){
  
  
}


void alloc_host() {
  
  // must use
  
}