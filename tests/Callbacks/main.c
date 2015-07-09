#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

#include "callbackdft.h"


/*typedef struct DataArray {
 double complex* data_r_host;
 double complex* data_k_host;
 cuDoubleComplex* data_r_dev;
 cuDoubleComplex* data_k_dev;
 uint64_t size;
} DataArray;*/



// global variables
// threads
pthread_barrier_t barrier;
//enum thread_id {KERNEL_THRD, MEMORY_THRD};

// cuda streams
cudaStream_t* streams_arr;
//enum stream_id {KERNEL_STREAM, MEMORY_STREAM};

DataArray* data_arr_ptr;



int main() {
  
  cudaDeviceReset();
  cudaDeviceSynchronize();
  
  // print device properties
  print_device();
    
  // create pointers to data
  const uint64_t size = N;
  DataArray* data_arr_ptr = (DataArray*) malloc((size_t) sizeof(DataArray)); // change to global variable <- easier to code
  
  // allocate memory for array of streams
  const uint8_t num_streams = 2; // rewrite on defines?
  streams_arr = (cudaStream_t*) malloc( (size_t) sizeof(cudaStream_t)*num_streams);
  
  // create threads
  const uint8_t num_threads = 2;
  printf("host thread id\t %u\ndevice thread id %u\n",KERNEL_THRD, MEMORY_THRD);
  
  pthread_t* thread_ptr_arr = (pthread_t*) malloc( (size_t) sizeof(pthread_t)*num_threads ); // alternatively pthread_t* thread_ptr_arr[num_threads];
  
  // init barier for threads
  pthread_barrier_init (&barrier, NULL, num_threads); // last number tells how many threads should be synchronized by this barier
  
  pthread_create(&thread_ptr_arr[KERNEL_THRD], NULL, host_thread, (void*) data_arr_ptr);
  pthread_create(&thread_ptr_arr[MEMORY_THRD], NULL, device_thread, (void*) data_arr_ptr);
  
  void* status;
  pthread_join(thread_ptr_arr[HOST_THRD], &status);
  pthread_join(thread_ptr_arr[DEVICE_THRD], &status);
  
  //printf("data visible in main thread:\n");  
  
  // Cleaning up
  free(thread_ptr_arr);
  free(streams_arr);
  free(data_arr_ptr);
  
  cudaThreadExit();
  cudaDeviceSynchronize();
  
  printf("Main: program completed. Exiting...\n");
  return EXIT_SUCCESS;
}
