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
    HANDLE_ERROR( cudaMemcpy(global_stuff->complex_arr1_dev, global_stuff->init_wf_map, NX*NY*NZ * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice) );
  }
  // make cufft plans - 4 needed and 8 callbacks
  // each cufft plan must be associated with specified stream
  // PLAN NOT EXECUTION IS ASSOCIATED WITH STREAM!
  
  
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
       
       FLAG_RUN_SIMULATION = false;
     }
  }
  
  
#ifdef DEBUG
  printf("last barrier reached by %s.\n",thread_names[HELPER_THRD]);
#endif
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
  
  // creating plans with callbacks
  const uint8_t num_plans = 4;
  cufftHandle plans[num_plans];
  for (uint8_t ii = 0; ii < num_plans; ii++)
    CHECK_CUFFT(	cufftCreate( &(plans[ii]) )	);
  
  size_t work_size;
  CHECK_CUFFT( cufftMakePlan1d(plans[FORWARD_PSI], NX*NY*NZ, CUFFT_Z2Z, 1, &work_size) );
  
  
  // !!! SPRAWDZIC !!! funckja: <- co robi?
  //cufftResult cufftSetAutoAllocation(cufftHandle *plan, bool autoAllocate);
  
#ifdef DEBUG
  printf("created FFT plans.\n");
#endif
  
#ifdef DEBUG
  printf("1st barrier reached by %s.\n",thread_names[HELPER_THRD]);
#endif
  pthread_barrier_wait (&barrier);
  
  
  
  
  // start algorithm
  // dt =
  uint16_t timesteps;
  while( FLAG_RUN_SIMULATION ) { // simulation will be runing until the flag is set to false
     timesteps = 1000;
     while(timesteps) {
       timesteps--;
       
       FLAG_RUN_SIMULATION = false;
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
  HANDLE_ERROR( cudaFree( global_stuff->Vdip_dev ) ); 		//
  
  
  pthread_barrier_wait (&barrier_global);
  pthread_exit(NULL);
}


void alloc_device(){
  
  
}


void alloc_host() {
  
  // must use
  
}