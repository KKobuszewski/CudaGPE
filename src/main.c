#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include "cudautils.h"
#include "global.h"
#include "simulation.h"

#define N 67108864


// sturctures definitions

/*
 * This structure provides general information and required data pointers to be shared between pthreads.
 */



// global variables
GlobalSettings* global_stuff;
const char* thread_names[] = {"SIMULATION_THRD","HELPER_THRD"};
const char* stream_names[] = {"SIMULATION_STREAM","MEMORY_STREAM"};


// set parameters - przepisac czesciowo na makra
const uint8_t num_streams = 2;
const uint8_t num_threads = 2; // except main thread
const uint8_t dim = 1;
  
const uint8_t filename_str_lenght = 128;
  
// threads
pthread_barrier_t barrier;
pthread_barrier_t barrier_global;
pthread_attr_t attr;
Array_of_thrd_functions thread_funcs = {simulation_thread, helper_thread};

// cuda streams
cudaStream_t* streams;




// functions' definitions
FILE** open_files();
void close_files(FILE** files, const uint8_t num_files);





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
  cudaDeviceSynchronize();
  // look for some goods solutions for initializing device
  
  
  // make stucture to pass all variables in program
  global_stuff = (GlobalSettings*) malloc( (size_t) sizeof(GlobalSettings));
  // fill with known information
  
  
  // parse command line args
  printf("command line arguments:\n");
  for (int ii = 0; ii < argc; ii++) {
    printf("%d. :\t%s\n", ii, argv[ii]);
    
    // 
    if (ii == 1) {
      //
    }
    // else if
  }
  
  

    
  
  
  
  
  // create streams
  streams = (cudaStream_t*) malloc( (size_t) sizeof(cudaStream_t)*num_streams);
  
  // create threads
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
  for (uint8_t ii; ii < num_threads; ii++) {
    void* status;
    pthread_join(threads[ii], &status);
  }
  
  // save data
  
  // close files
  //if (backup_file) 	fclose(backup_file);
  //if (wf_file) 		fclose(wf_file);
  
  // clear memory
  
  return EXIT_SUCCESS;
}


/*
 * In case to have transparent code - open files in special function and store pointers to the files in an array
 * PRZEMYSLEC ILE PLIKOW POTRZEBA -> WAVEFUNCTION, WCZYTYWANIE, BACKUP, ENERGIA, PRZEKROJE
 * CZYTAC/ZAPISYWAC WAVEFUNCTION DO PLIKOW BINARNYCH ZA POMOCA MMAP, A TIMING JAKOS INACZEJ (DO .TXT LUB nvprof UZYWAC)
 */
FILE** open_files(const uint8_t num_files) {
  
  FILE** files = (FILE**) malloc(num_files);
  //const uint8_t filename_str_lenght = 128;
  
  // move to creating files
  char backup_filename[filename_str_lenght];
  FILE* backup_file = NULL;
  sprintf(backup_filename,"./backup_dim%d_N%d.txt",dim,N );
  printf("backup save in: %s\n",backup_filename);
  backup_file = fopen(backup_filename,"w");
  if (!backup_file) printf("Error opening file %s!\n",backup_filename);
  
  files[0] = backup_file; // enum -> BACKUP_FILE
  
  
  char wf_filename[filename_str_lenght];
  FILE* wf_file = NULL;
  
  for (uint8_t ii=0; ii< num_files; ii++) {
    //files[ii] = fopen / mmap
    
  }
  
  
  return files;
}

/*
 * Closes files form array of pointers to files.
 * FILE** files - array of pointers to files
 *  const uint8_t num_files - number of files in the array
 */
void close_files(FILE** files, const uint8_t num_files) {
  for (uint8_t ii = 0; ii< num_files; ii++)
    if (files[ii]) fclose(files[ii]);
}


