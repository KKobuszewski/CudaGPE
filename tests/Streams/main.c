#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include "cudautils.h"

// global variables
// threads
pthread_barrier_t barrier;
enum thread_id {HOST_THRD, DEVICE_THRD};

// cuda streams
cudaStream_t* streams_arr;
enum stream_id {KERNEL_STREAM, MEMORY_STREAM};

typedef struct DataArray{
 double complex** data_r;
 double complex** data_k;
 uint64_t size;
} DataArray;

void create_data_arr(DataArray* data_arr,
		     double complex** data_r,
		     double complex** data_k,
		     cuDoubleComplex** data_r_dev,
		     cuDoubleComplex** data_k_dev,
		     const uint64_t size) {
  data_arr->data_r=data_r;
  data_arr->data_k=data_k;
  data_arr->size=size;
}

void free_data_arr(DataArray* data_arr) {
  free(*(data_arr->data_r));
  free(*(data_arr->data_k));
}

void alloc_data_host(DataArray* data_arr) {
  // when allocating data, we don't know which pointer will be ruturned, but we can use pointer to pointer
  // data_arr->data_r = complex**
  // *(complex**) = *complex
  // *(data_arr->data_r) means smth what is under the adress data_arr->data_r, so complex* double
  cudaHostAlloc((void**) data_arr->data_r, sizeof(double complex)*N, cudaHostAllocDefault); // pinnable memory <- check here for cudaMallocHost (could be faster)
  cudaHostAlloc((void**) data_arr->data_k, sizeof(double complex)*N, cudaHostAllocDefault); // pinnable memory
  
  // in case of pageable memory (slower/not asynchronous):
  //*(data_arr->data_r) = (double complex*) malloc( (size_t) sizeof(double complex)*data_arr->size );
  //*(data_arr->data_k) = (double complex*) malloc( (size_t) sizeof(double complex)*data_arr->size );
  
}

void alloc_data_device(DataArray* data_arr) {
  cudaMalloc((void**) data_arr->data_r_dev, sizeof(double complex)*N, cudaHostAllocDefault); // pinnable memory <- check here for cudaMallocHost (could be faster)
  cudaMalloc((void**) data_arr->data_k_dev, sizeof(double complex)*N, cudaHostAllocDefault); // pinnable memory
}

/*
 * Function to be called in thread managing host operations and invoking kernels
 */
void* host_thread(void* passing_ptr){
  DataArray* data_arr_ptr = (DataArray*) passing_ptr;
  
  alloc_data_host(data_arr_ptr);
  printf("data allocated by host thread\n");
  
  printf("data filling by host thread\n");
  for (uint64_t ii = 0; ii < data_arr_ptr->size; ii++) {
    (*(data_arr_ptr->data_r))[ii] = ii;
    (*(data_arr_ptr->data_k))[ii] = data_arr_ptr->size-ii;
  }
  printf("data filled by host thread\n");
  
  // synchronize with another thread
  pthread_barrier_wait (&barrier);
  
  printf("closing host thread\n");
  pthread_exit(NULL);
}


/*
 * Function to be called 
 */
void* device_thread(void* passing_ptr) {
  DataArray* data_arr_ptr = (DataArray*) passing_ptr;
  
  // init device, allocate suitable variables in gpu memory ...
  
  print_device();
  
  // synchronize with another thread
  pthread_barrier_wait (&barrier);
  
  printf("data visible in device thread:\n");
  for (uint64_t ii = 0; ii < data_arr_ptr->size; ii++) {
    printf("%lu.\t",ii);
    printf("%lf + %lfj\t", creal( (*(data_arr_ptr->data_r))[ii] ), cimag( (*(data_arr_ptr->data_r))[ii] ));
    printf("%lf + %lfj\n", creal( (*(data_arr_ptr->data_k))[ii] ), cimag( (*(data_arr_ptr->data_k))[ii] ));
  }
  
  
  printf("closing device thread\n");
  pthread_exit(NULL);  
}

/*
 * main should only control threads
 * 
 * the threads should be invoked on different cores:
 * http://stackoverflow.com/questions/1407786/how-to-set-cpu-affinity-of-a-particular-pthread
 * https://www.google.pl/search?client=ubuntu&channel=fs&q=how+to+schedule+pthreads+through+cores&ie=utf-8&oe=utf-8&gfe_rd=cr&ei=PSudVePFOqeA4AShra2AAQ
 */
int main() {
  
  // create pointers to data
  const uint64_t size = 50;
  double complex* data_r_host = NULL; // initializing with NULL for debuging purposes
  double complex* data_k_host = NULL; // initializing with NULL for debuging purposes
  DataArray* data_arr_ptr = (DataArray*) malloc((size_t) sizeof(DataArray)); // change to global variable <- easier to code
  create_data_arr(data_arr_ptr, &data_r_host, &data_k_host, size);
  
  // create threads
  const uint8_t num_threads = 2;
  printf("host thread id\t %u\ndevice thread id %u\n",HOST_THRD, DEVICE_THRD);
  
  pthread_t* thread_ptr_arr = (pthread_t*) malloc( (size_t) sizeof(pthread_t)*num_threads ); // alternatively pthread_t* thread_ptr_arr[num_threads];
  
  // init barier for threads
  pthread_barrier_init (&barrier, NULL, num_threads); // last number tells how many threads should be synchronized by this barier
  
  pthread_create(&thread_ptr_arr[HOST_THRD], NULL, host_thread, (void*) data_arr_ptr);
  pthread_create(&thread_ptr_arr[DEVICE_THRD], NULL, device_thread, (void*) data_arr_ptr);
  
//   for (uint8_t ii = 0; ii < num_threads; ii++) {
//     pthread_create(thread_ptr_arr[ii], NULL, host_thread, (void*) data_arr_ptr);
//   }
    
  //cudaStream_t stream1;
  //cudaStream_t stream2;
  //cudaStream_t* streams_arr[2] = {&stream1, &stream2};
  
  void* status;
  pthread_join(thread_ptr_arr[HOST_THRD], &status);
  pthread_join(thread_ptr_arr[DEVICE_THRD], &status);
  
  printf("data visible in main thread:\n");
  for (uint64_t ii=0; ii < data_arr_ptr->size; ii++) {
    printf( "%lu.\t",ii );
    printf( "%lf + %lf\t", creal(data_r_host[ii]), cimag(data_r_host[ii]) );
    printf( "%lf + %lf\n", creal(data_k_host[ii]), cimag(data_k_host[ii]) );
  }
  
  
  
  free(thread_ptr_arr);
  free_data_arr(data_arr_ptr);
  free(data_arr_ptr);
  
  
  
  printf("Main: program completed. Exiting...\n");
  return EXIT_SUCCESS;
}
