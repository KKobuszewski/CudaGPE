#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <math.h>
#include <pthread.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"

#include "cudautils.h"
#include "global.h"
#include "simulation.h"
#include "fileIO.h"

// sturctures definitions

/*
 * This structure provides general information and required data pointers to be shared between pthreads.
 */



// global variables declarations
Globals* global_stuff;
const char* thread_names[] = {"SIMULATION_THRD","HELPER_THRD"};
const char* stream_names[] = {"SIMULATION_STREAM","HELPER_STREAM"};


// set parameters - przepisac czesciowo na makra
const uint8_t num_streams = 2;
const uint8_t num_threads = 2; // except main thread


// threads
pthread_barrier_t barrier_global;
pthread_attr_t attr;
Array_of_thrd_functions thread_funcs = {simulation_thread, helper_thread};// type Array_of_thrd_functions is defined in global.h

// cuda streams
//cudaStream_t* streams;










/*
 * main function in main thread will only manage another threads - this allows having heterogenous and multistreamed application.
 * Application is dedicated only for single device architecture.
 * 
 * Command line arguments:
 * (for example path to a file with initial wavefunction - in case there is no initial wavefunction it will create new wavefunction)
 * 
 * return: EXIT_SUCCESS (if program ends with no bugs)
 */
int main(int argc, char* argv[]) {
    
#ifdef DEBUG
  printf(" ++++++++++++++++++++++++++++++++ DEBUG MODE ++++++++++++++++++++++++++++++++\n");
#endif
  
  // print device properties
  print_device();
  
  // clear and init device
  cudaDeviceReset(); // we want to be certain of proper behaviour of the device
  //cudaDeviceSynchronize();
  // look for some goods solutions for initializing device
  gpuDeviceInit(0);
  
  // make stucture to pass all variables in program
  global_stuff = (Globals*) malloc( (size_t) sizeof(Globals));
  // fill with known information
  global_stuff->init_wf_fd = -1;
  
  printf("\n");
  printf("Simulation params: \n");
  printf("dimensions: %u\n", DIM);
  printf("lattice points in direction x: %u\n", NX);
  printf("lattice points in direction y: %u\n", NY);
  printf("lattice points in direction z: %u\n", NZ);
  printf("total number of points in a lattice: %u, 2**%u\n", NX*NY*NZ, (uint32_t) ( log(NX*NY*NZ)/log(2) ) );
  printf("[xmin, xmax] : [%.15f, %.15f]\n", XMIN, XMAX);
  printf("dx: %.15f\n",DX);
  printf("[kxmin, kxmax] : [%.15f, %.15f]\n", KxMIN, KxMAX);
  printf("dkx: %.15f\n",DKx);
  printf("\n");
  
  
  // parse command line args
  printf("%d command line arguments:\n", argc);
  for (int ii = 0; ii < argc; ii++) {
    printf("%d. :\t%s", ii, argv[ii]);
    
    // open file with wavefunction to be read
    if (ii == 1) {
      printf("\tinitial wavefunction will be loaded from file %s", argv[ii]);
      global_stuff->init_wf_fd = mmap_create(argv[ii],					// in fileIO.c
					     (void**) &(global_stuff->init_wf_map),
					     NX*NY*NZ * sizeof(double complex),
					     PROT_READ, MAP_SHARED);
#ifdef DEBUG
      printf("\n\t\t\t\tsample of mmaped initial wavefunction: %lf + %lfj\n", creal(global_stuff->init_wf_map[1000]), cimag(global_stuff->init_wf_map[1000]));
#endif
      
    }
    // smth else ...
    // else if (ii == 2) {
    
    printf("\n");
  }
  
  // open files
  open_files();
  
  
  // create streams (CUDA)
  global_stuff->streams = (cudaStream_t*) malloc( (size_t) sizeof(cudaStream_t)*num_streams );
  
  // create threads (POSIX)
  pthread_t* threads = (pthread_t*) malloc( (size_t) sizeof(pthread_t)*num_threads );
  pthread_attr_init(&attr); // attr is declared in global variables
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); // threads will be managed by waiting for non-main threads to end
  pthread_barrier_init (&barrier, NULL, num_threads); // last number tells how many threads should be synchronized by this barier
  pthread_barrier_init (&barrier_global, NULL, num_threads+1);
  // this interface enable simple adding new threads
  for (uint8_t ii = 0; ii < num_threads; ii++) {
    // create threads
    printf("creating thread %s\n",thread_names[ii]);
    pthread_create(&threads[ii], &attr, thread_funcs[ii], (void*) global_stuff);
    
    // set affinity (every thread on its own core if possible)
    cpu_set_t cpu_core;
    CPU_ZERO(&cpu_core);
    CPU_SET(ii, &cpu_core);
    pthread_setaffinity_np(threads[ii], sizeof(cpu_set_t), &cpu_core);
    if (CPU_ISSET(ii, &cpu_core)) printf("affinity thread %s set successfully.\n",thread_names[ii]);
  }
  pthread_barrier_wait (&barrier_global); // global lock for threads
  
  // allocate memory ?
  
  
  
  // run threads
  
  
  // manage threads, use barriers to sync threads
  
  
  // join threads
  pthread_barrier_wait (&barrier_global); //maybe not necessary
  for (uint8_t ii = 0; ii < num_threads; ii++) {
    void* status;
    pthread_join(threads[ii], &status);
  }
  
  // save data
  
  // close files
  //if (backup_file) 	fclose(backup_file);
  //if (wf_file) 		fclose(wf_file);
  
  
  // close files
  mmap_destroy(global_stuff->init_wf_fd, global_stuff->init_wf_map, NX*NY*NZ * sizeof(double complex));
  close_files(global_stuff->files, global_stuff->num_files);
  
  
  /// free memory on device
  cudaFree(global_stuff->complex_arr1_dev ); 	//
  cudaFree(global_stuff->complex_arr2_dev ); 	//
  //HANDLE_ERROR( cudaFree(global_stuff->double_arr1_dev)  ); 	//
  cudaFree(global_stuff->propagator_T_dev ); 	//
  //HANDLE_ERROR( cudaFree(global_stuff->propagator_Vext_dev) );	//
  //HANDLE_ERROR( cudaFree(global_stuff->Vdip_dev) );		//
  
  
  //HANDLE_ERROR( cudaFree(global_stuff->mean_T_dev) ); // result of integral with kinetic energy operator in momentum representaion
  //HANDLE_ERROR( cudaFree(global_stuff->mean_Vdip_dev) ); // result of integral with Vdip operator in positions' representation
  //HANDLE_ERROR( cudaFree(global_stuff->mean_Vext_dev) ); // result of integral with Vext operator in positions' representation
  //HANDLE_ERROR( cudaFree(global_stuff->mean_Vcon_dev) ); // result of integral with Vcon operator in positions' representation
  cudaFree(global_stuff->norm_dev ); //
  
  
  // clear memory
  free(threads);
  
  
  free(global_stuff->streams);
  free(global_stuff);
  
  
  printf("Main: program completed. Exiting...\n");
  cudaThreadExit();
  cudaDeviceReset();
  return EXIT_SUCCESS;
}

