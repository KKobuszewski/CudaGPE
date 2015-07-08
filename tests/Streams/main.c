#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

#include "cudautils.h"

#define N 32

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
 cuDoubleComplex** data_r_dev;
 cuDoubleComplex** data_k_dev;
 uint64_t size;
} DataArray;

void create_data_arr(DataArray* data_arr,
		     double complex** data_r,
		     double complex** data_k,
		     /*cuDoubleComplex** data_r_dev,*/
		     /*cuDoubleComplex** data_k_dev,*/
		     const uint64_t size) {
  data_arr->data_r=data_r;
  data_arr->data_k=data_k;
  data_arr->size=size;
}

void free_data_arr(DataArray* data_arr) {
  cudaFreeHost(*(data_arr->data_r));
  cudaFreeHost(*(data_arr->data_k));
  cudaFree(*(data_arr->data_r_dev));
  cudaFree(*(data_arr->data_k_dev));
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
  cudaMalloc((void**) data_arr->data_r_dev, sizeof(double complex)*N); // pinnable memory <- check here for cudaMallocHost (could be faster)
  cudaMalloc((void**) data_arr->data_k_dev, sizeof(double complex)*N); // pinnable memory
}

/*
 * Function to be called in thread managing host operations and invoking kernels
 */
void* host_thread(void* passing_ptr) {
  DataArray* data_arr_ptr = (DataArray*) passing_ptr;
  
  alloc_data_host(data_arr_ptr);
  printf("data allocated by host thread\n");
  
  //printf("data filling by host thread\n");
  for (uint64_t ii = 0; ii < data_arr_ptr->size; ii++) {
    (*(data_arr_ptr->data_r))[ii] = ii;
    (*(data_arr_ptr->data_k))[ii] = data_arr_ptr->size-ii;
  }
  printf("data filled by host thread\n");
  
  // synchronize after allocating memory - streams should be created, mem on device ready for copying
  pthread_barrier_wait (&barrier);
  printf("1st barier host thread - allocating mem on cpu\n");
  
  
  
  
  
  //  here we can make cufft plan, for example
  cufftHandle plan_forward;
  cufftPlan1d(&plan_forward, N, CUFFT_Z2Z, 1);
  
  
  
  // synchornize after ... - data should be copyied on device
  pthread_barrier_wait (&barrier);
  printf("2nd barier host thread - \n");
  
  
  // run some computations
  cufftExecZ2Z(plan_forward, *(data_arr_ptr->data_r_dev), *(data_arr_ptr->data_k_dev), CUFFT_FORWARD);
  
  // synchornize after computations - 
  pthread_barrier_wait (&barrier);
  printf("3rd barier host thread - \n");
  
  
  
  // synchornize after computations - 
  pthread_barrier_wait (&barrier);
  printf("4th barier host thread - \n");
  
  printf("data visible in device thread:\n");
  for (uint64_t ii = 0; ii < data_arr_ptr->size; ii++) {
    printf("%lu.\t",ii);
    printf("%lf + %lfj\t", creal( (*(data_arr_ptr->data_r))[ii] ), cimag( (*(data_arr_ptr->data_r))[ii] ));
    printf("%lf + %lfj\n", creal( (*(data_arr_ptr->data_k))[ii] ), cimag( (*(data_arr_ptr->data_k))[ii] ));
  }
  
  printf("closing host thread\n");
  pthread_exit(NULL);
}


/*
 * Function to be called 
 */
void* device_thread(void* passing_ptr) {
  DataArray* data_arr_ptr = (DataArray*) passing_ptr; // casting passed pointer
  
  
  cuDoubleComplex* data_r_dev;
  cuDoubleComplex* data_k_dev;
  data_arr_ptr->data_r_dev = &data_r_dev; // in this way it would be easier to handle pointer to arrays
  data_arr_ptr->data_k_dev = &data_k_dev;
  
  // Each thread creates new stream ustomatically???
  // http://devblogs.nvidia.com/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
  cudaStreamCreate(streams_arr);
  cudaStreamCreate(streams_arr+1);
  
  // init device, allocate suitable variables in gpu memory ...
  alloc_data_device(data_arr_ptr);
  printf("data allocated by host thread\n");
  
  
  // synchronize after allocating memory - data on host should be allocated and ready for copying
  cudaDeviceSynchronize(); // CHECK IF THIS DO NOT CAUSE ERRORS! - should syncronize host and device irrespective on pthreads
  // cudaStreamSynchronize( <enum stream> ); // to synchronize only with stream !!!
  pthread_barrier_wait (&barrier);
  printf("1st barier device thread - allocating mem on gpu\n");
  
  
  
  
  //copying data
  cudaMemcpyAsync(data_arr_ptr->data_r_dev, data_arr_ptr->data_r, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, streams_arr[MEMORY_STREAM]);
  
  // synchronize after copying data
  cudaDeviceSynchronize(); // should be used on
  pthread_barrier_wait (&barrier);
  printf("2nd barier device thread - copying data on gpu\n");
  
  
  
  
  
  
  printf("data visible in device thread:\n");
  for (uint64_t ii = 0; ii < data_arr_ptr->size; ii++) {
    printf("%lu.\t",ii);
    printf("%lf + %lfj\t", creal( (*(data_arr_ptr->data_r))[ii] ), cimag( (*(data_arr_ptr->data_r))[ii] ));
    printf("%lf + %lfj\n", creal( (*(data_arr_ptr->data_k))[ii] ), cimag( (*(data_arr_ptr->data_k))[ii] ));
  }
  
  // synchronize after copying
  cudaDeviceSynchronize(); // should be used on
  pthread_barrier_wait (&barrier);
  printf("3rd barier device thread - \n");
  
  
  
  //copying data
  cudaMemcpyAsync(data_arr_ptr->data_r, data_arr_ptr->data_r_dev, N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, streams_arr[MEMORY_STREAM]);
  
  
  
  // synchronize after copying back data
  cudaDeviceSynchronize(); // should be used on
  pthread_barrier_wait (&barrier);
  printf("4th barier device thread - \n");
  
  
  
  
  
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
  
  // print device properties
  print_device();
    
  // create pointers to data
  const uint64_t size = N;
  double complex* data_r_host = NULL; // initializing with NULL for debuging purposes
  double complex* data_k_host = NULL; // initializing with NULL for debuging purposes
  DataArray* data_arr_ptr = (DataArray*) malloc((size_t) sizeof(DataArray)); // change to global variable <- easier to code
  create_data_arr(data_arr_ptr, &data_r_host, &data_k_host, size);
  
  // allocate memory for array of streams
  const uint8_t num_streams = 2; // rewrite on defines?
  streams_arr = (cudaStream_t*) malloc( (size_t) sizeof(cudaStream_t)*num_streams);
  
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
  free(streams_arr);
  free_data_arr(data_arr_ptr);
  free(data_arr_ptr);
  
  
  
  printf("Main: program completed. Exiting...\n");
  return EXIT_SUCCESS;
}
