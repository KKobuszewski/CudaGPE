#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

#include "callbackdft.cuh"
#include "cudautils.cuh"


// timing
cudaEvent_t start_t;
cudaEvent_t stop_t;

/*
 * Function to be called in thread managing host operations and invoking kernels
 */
void* kernel_thread(void* passing_ptr) {
   //= (DataArray*) passing_ptr;
  
  cuDoubleComplex* data_r_host;
  cuDoubleComplex* data_k_host;
  
  
  HANDLE_ERROR( cudaHostAlloc((void**) &data_r_host, sizeof(cuCoubleComplex)*N, cudaHostAllocDefault) ); // pinnable memory <- check here for cudaMallocHost (could be faster)
  HANDLE_ERROR( cudaHostAlloc((void**) &data_k_host, sizeof(cuCoubleComplex)*N, cudaHostAllocDefault) ); // pinnable memory
  
  data_arr_ptr->data_r_host = data_r_host;
  data_arr_ptr->data_k_host = data_k_host;
  
  printf("data on host allocated by kernel thread\n");
    
  // 1ST BARRIER: synchronize after allocating memory - streams should be created, mem on device ready for copying
  pthread_barrier_wait (&barrier);
  printf("1st barier host thread - allocating mem on cpu\n");
  printf("kernel thread - data_r_host:\t%p\n",data_arr_ptr->data_r_host);
  printf("kernel thread - data_r_dev:\t%p\n",data_arr_ptr->data_r_dev);
  printf("kernel thread - data_k_host:\t%p\n",data_arr_ptr->data_k_host);
  printf("kernel thread - data_k_dev:\t%p\n",data_arr_ptr->data_k_dev);
  
  // count parameters to run kernel
  uint64_t threadsPerBlock;
  if (N >= 33554432)
    threadsPerBlock = 1024;
  else
    threadsPerBlock = 512;
  dim3 dimBlock(threadsPerBlock,1,1);
  printf("threads Per block: %lu\n", threadsPerBlock);
  printf("blocks: %lu\n",(N + threadsPerBlock - 1)/threadsPerBlock);
  dim3 dimGrid( (N + threadsPerBlock - 1)/threadsPerBlock, 1, 1 ); // (numElements + threadsPerBlock - 1) / threadsPerBlock
  
  // execute kernel
  cudaGauss_1d<<<dimGrid,dimBlock, ,>>>(data_arr_ptr->data_r_dev, N);
  HANDLE_ERROR( cudaGetLastError() );
  
  printf("kernel thread initiates data on device\n");
  
  
  
  
  
  //2ND BARRIER: synchornize after ... - data should be copyied on device
  pthread_barrier_wait (&barrier);
  printf("2nd barier host thread - \n");
  /*
  
  // run some computations
  cufftExecZ2Z(plan_forward, *(data_arr_ptr->data_r_dev), *(data_arr_ptr->data_k_dev), CUFFT_FORWARD);
  printf("cufft done\n");
  
  // synchornize after computations - 
  
  cudaDeviceSynchronize(); // should be used on
  pthread_barrier_wait (&barrier);
  printf("3rd barier host thread - \n");
  
  
  
  // synchornize after computations - 
  pthread_barrier_wait (&barrier);
  printf("4th barier host thread - \n");
  */
  //printf("data visible in host thread:\n");
  /*for (uint64_t ii = 0; ii < (data_arr_ptr->size <= 32) ? data_arr_ptr->size : 32 ; ii++) {
    printf("%lu.\t",ii);
    printf("%lf + %lfj\t", creal( (*(data_arr_ptr->data_r))[ii] ), cimag( (*(data_arr_ptr->data_r))[ii] ));
    printf("%lf + %lfj\n", creal( (*(data_arr_ptr->data_k))[ii] ), cimag( (*(data_arr_ptr->data_k))[ii] ));
  }*/
  
  
  pthread_barrier_wait (&barrier);
  printf("kernel thread frees memory");
  
  HANDLE_ERROR( cudaFreeHost(data_r_host)) );
  printf("host r space freed\n");
  HANDLE_ERROR( cudaFreeHost(data_k_host)) );
  printf("host k space freed\n");
  
  printf("closing host thread\n");
  pthread_exit(NULL);
}


/*
 * Function to be called 
 */
void* memory_thread(void* passing_ptr) {
  //DataArray* data_arr_ptr = (DataArray*) passing_ptr; // casting passed pointer
  
  
  cuDoubleComplex* data_r_dev;
  cuDoubleComplex* data_k_dev;
  
  
  // init device, allocate suitable variables in gpu memory ...
  //alloc_data_device(data_arr_ptr);
  cudaMalloc((void**) &data_r_dev, sizeof(double complex)*N); // pinnable memory <- check here for cudaMallocHost (could be faster)
  cudaMalloc((void**) &data_k_dev, sizeof(double complex)*N); // pinnable memory
  
  // 
  data_arr_ptr->data_r_dev = data_r_dev; // in this way it would be easier to handle pointer to arrays
  data_arr_ptr->data_k_dev = data_k_dev;
  printf("data allocated by host thread\n");
  
  // Each thread creates new stream ustomatically???
  // http://devblogs.nvidia.com/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
  cudaStreamCreateWithFlags(streams_arr, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(streams_arr+1, cudaStreamNonBlocking);
  printf("streams created\n");
  
  //1ST BARRIER synchronize after allocating memory - data on host should be allocated and ready for copying
  pthread_barrier_wait (&barrier);
  cudaDeviceSynchronize(); // CHECK IF THIS DO NOT CAUSE ERRORS! - should syncronize host and device irrespective on pthreads
  printf("1st barier device thread - allocating mem on gpu\n");
  printf("1st barier host thread - allocating mem on cpu\n");
  printf("memory thread - data_r_host:\t%p\n",data_arr_ptr->data_r_host);
  printf("memory thread - data_r_dev:\t%p\n",data_arr_ptr->data_r_dev);
  printf("memory thread - data_k_host:\t%p\n",data_arr_ptr->data_k_host);
  printf("memory thread - data_k_dev:\t%p\n",data_arr_ptr->data_k_dev);
  
  
  //  here we can make cufft plan, for example
  cufftPlan1d(&plan_forward, N, CUFFT_Z2Z, 1);
  printf("memory thread creates cufft plan\n");
  
  //2ND BARRIER: synchronize after filling data & creating plan
  cudaDeviceSynchronize();
  pthread_barrier_wait (&barrier);
  printf("2nd barier device thread - \n");
  
  
  
  //copying data
  //cudaMemcpyAsync( *(data_arr_ptr->data_r), *(data_arr_ptr->data_r_dev), N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, streams_arr[MEMORY_STREAM] );
  cudaMemcpyAsync( *(data_arr_ptr->data_r), data_r_dev, N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, streams_arr[MEMORY_STREAM] );
  
  
  // synchronize after copying back data
  cudaDeviceSynchronize(); // should be used on
  pthread_barrier_wait (&barrier);
  printf("memory thread frees memory");
  
  
  cudaStreamDestroy(streams_arr[KERNEL_STREAM]);
  cudaStreamDestroy(streams_arr[MEMORY_STREAM]);
  
  cudaFree(data_r_dev);
  printf("device r space freed\n");
  cudaFree(data_k_dev);
  cudaDeviceSynchronize();
  printf("device k space freed\n");
  
  printf("closing device thread\n");
  pthread_exit(NULL);  
}