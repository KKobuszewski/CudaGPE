#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <math.h>
#include <complex.h>
#include <cuda.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cufftXt.h>

#include "global.h"
#include "simulation.cuh"
#include "cudautils.cuh"


// timing
cudaEvent_t start_t;
cudaEvent_t stop_t;


// global variables
extern GlobalSettings* global_stuff;
bool FLAG_RUN_SIMULATION = true;
extern const char* thread_names[];
extern const char* stream_names[];

/*
 * 
 * !!! VERSION FOR 1 PTHREAD !!!
 * 
 */

void* simulation_thread(void* passing_ptr) {
  
  //stick_this_thread_to_core(1); <- in cudautils, not used, include to header first
  pthread_barrier_wait (&barrier_global);
  printf("running %s thread.\n",thread_names[SIMULATION_THRD]);
  
  // allocate memory on host & device
  
  
  // fill arrays on host & device
  
  pthread_barrier_wait (&barrier);
  // copy data async from host to device (if needed)
  
  
  // make cufft plans - 4 needed and 8 callbacks
  // each cufft plan must be associated with specified stream
  // PLAN NOT EXECUTION IS ASSOCIATED WITH STREAM!
  
  
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
  
  
  // free memory on host & device
  
  pthread_exit(NULL);
}


void* helper_thread(void* passing_ptr) {
  
  //stick_this_thread_to_core(2);
  
  pthread_barrier_wait (&barrier_global);
  printf("running %s thread.\n",thread_names[HELPER_THRD]);
  
  // init memory on host & device
  
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
  
  pthread_exit(NULL);
}


void alloc_device(){
  
  
}


void alloc_host() {
  
  // must use
  
}