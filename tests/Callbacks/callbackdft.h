#ifndef __CALLBACKDFT_H__
#define __CALLBACKDFT_H__

// macro constants
#define N 32//67108864

#include <cuComplex.h>
#include <cufft.h>

// type definitions

typedef struct DataArray {
 double complex* data_r_host;
 double complex* data_k_host;
 cuDoubleComplex* data_r_dev;
 cuDoubleComplex* data_k_dev;
 cufftHandle* plan_forward;
 uint64_t size;
} DataArray;


// global variables
extern DataArray* data_arr_ptr;

extern pthread_barrier_t barrier;
enum thread_id {KERNEL_THRD, MEMORY_THRD};

extern cudaStream_t* streams_arr;
enum stream_id {KERNEL_STREAM, MEMORY_STREAM};


// threads' functions
void* kernel_thread(void* passing_ptr)
void* memory_thread(void* passing_ptr);


#endif