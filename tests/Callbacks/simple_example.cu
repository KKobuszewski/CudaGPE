#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>
#include <cuda.h>
#include <cufft.h>
#include <cufftXt.h>

#define N ((uint64_t) 1<<22)

#define M_2PI (6.283185307179586)
#define SQRT_2PI (2.5066282746310002)
#define INV_SQRT_2PI (0.3989422804014327)
#define SIGMA (1)

/*
 * compile: 	nvcc -dc -lm -g -G -Xptxas="-v" -m64 -O3 -o simple_example.o -c simple_example.cu
 * 		nvcc -m64 -arch=sm_35 -o simple_example simple_example.o -lcufft_static -lculibos
 * 
 *		//with store callback to normalize: 
 * 		nvcc -dc -lm -g -G -Xptxas="-v" -m64 -O3 -o simple_example_cb_store_normalize.o -c simple_example.cu -D STORE_CB_NORMALIZE
 * 		nvcc -m64 -arch=sm_35 -o simple_example_cb_store_normalize simple_example_cb_store_normalize.o -lcufft_static -lculibos
 * 
 * sources:
 * http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-use-cufft-callbacks-custom-data-processing/
 * + example in cuda samples
 */

/* CALLBACK TYPES

typedef enum cufftXtCallbackType_t {
    CUFFT_CB_LD_COMPLEX = 0x0,
    CUFFT_CB_LD_COMPLEX_DOUBLE = 0x1,
    CUFFT_CB_LD_REAL = 0x2,
    CUFFT_CB_LD_REAL_DOUBLE = 0x3,
    CUFFT_CB_ST_COMPLEX = 0x4,
    CUFFT_CB_ST_COMPLEX_DOUBLE = 0x5,
    CUFFT_CB_ST_REAL = 0x6,
    CUFFT_CB_ST_REAL_DOUBLE = 0x7,
    CUFFT_CB_UNDEFINED = 0x8
} cufftXtCallbackType;


Read more at: http://docs.nvidia.com/cuda/cufft/index.html#ixzz3fPgy3600
Follow us: @GPUComputing on Twitter | NVIDIA on Facebook

*/


static __device__ cufftDoubleComplex cufftSgn(void *dataIn, 
					      size_t offset, 
					      void *callerInfo, 
					      void *sharedPtr) 
{
    if (offset < N/2) {
      //((cufftDoubleComplex* ) dataIn)[offset] = make_cuDoubleComplex(-1.,0.);
      //printf("index: %lu\tvalue:%f",-1);
      return make_cuDoubleComplex(-1.,0.);
    }
    else {
      
      //printf("index: %lu\tvalue:%f",-1);
      return make_cuDoubleComplex(1.,0.);
    }
}

static __device__ cufftDoubleComplex cufftRect(void *dataIn, 
					      size_t offset, 
					      void *callerInfo, 
					      void *sharedPtr) 
{
    if ((offset < N/4) || offset > 3*N/4) {
      //((cufftDoubleComplex* ) dataIn)[offset] = make_cuDoubleComplex(-1.,0.);
      //printf("index: %lu\tvalue:%f",-1);
      return make_cuDoubleComplex(0.,0.);
    }
    else {
      
      //printf("index: %lu\tvalue:%f",-1);
      return make_cuDoubleComplex(1.,0.);
    }
}

static __device__ cufftDoubleComplex cudaGauss_1d(void *dataIn, 
					      size_t offset, 
					      void *callerInfo, 
					      void *sharedPtr) 
{
  // get the index of thread
  //uint64_t ii = offset;
  
  
  // allocate constants in shared memory
  const double x0 = (-5*SIGMA);
  const double dx = (10*SIGMA)/((double) N);
  
  //if (ii < N) {
  //  ((cufftDoubleComplex)* dataIn)[ii] = make_cuDoubleComplex( INV_SQRT_2PI*exp(-(x0 + ii*dx)*(x0 + ii*dx)/2/SIGMA)/SIGMA, 0. );
    
  //}
  return make_cuDoubleComplex( INV_SQRT_2PI*exp(-(x0 + offset*dx)*(x0 + offset*dx)/2/SIGMA)/SIGMA, 0. );
}

static __device__ void cufft_normalize(void *dataOut, 
				  size_t offset, 
				  cufftDoubleComplex element, 
				  void *callerInfo, 
				  void *sharedPtr) 
{
  ((cufftDoubleComplex*) dataOut)[offset] = make_cuDoubleComplex( cuCreal(element)/((double) N), cuCimag(element)/((double) N) );
}
// pointer to callback function (on device)
//__device__ cufftCallbackLoadZ d_loadCallbackPtr = cudaGauss_1d;
__device__ cufftCallbackLoadZ d_loadCallbackPtr = cufftRect;
__device__ cufftCallbackStoreZ d_storeCallbackPtr = cufft_normalize;

__global__ void kernel_normalize(cufftDoubleComplex* cufft_inverse_data) {
  uint64_t ii = blockIdx.x*blockDim.x + threadIdx.x;
  
  // in both kernel as well as callback we use predefined N to have comparable performance results
  
  if (ii < N) {
    cufft_inverse_data[ii] = make_cuDoubleComplex( cuCreal(cufft_inverse_data[ii])/((double) N), cuCimag(cufft_inverse_data[ii])/((double) N) );
  }
}

int main (){
  
  cudaDeviceReset();
  cudaDeviceSynchronize();
  
  const uint8_t filename_str_lenght = 128;
  const uint8_t dim = 1;
    
  char filename1d[filename_str_lenght];
  FILE *file1d;
  
#ifdef STORE_CB_NORMALIZE
  sprintf(filename1d,"cufft_%dd_N%lu_cb_store.bin",dim,N );
#else
  sprintf(filename1d,"cufft_%dd_N%lu.bin",dim,N );
#endif
  printf("1d cufft example save in: %s\n",filename1d);
  file1d = fopen(filename1d, "wb");
  if (file1d == NULL)
  {
      printf("Error opening file %s!\n",filename1d);
      exit(EXIT_FAILURE);
  }
  
  printf("N %lu\n",N);
  
  cufftDoubleComplex *data_dev;
  cudaMalloc((void**)&data_dev, sizeof(cufftDoubleComplex)*N);
  cudaDeviceSynchronize();
  
  cufftDoubleComplex* data_host;
  cudaHostAlloc((void**) &data_host, sizeof(cufftDoubleComplex)*N, cudaHostAllocDefault);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    exit(EXIT_FAILURE);
  }
  cudaDeviceSynchronize();
  
  
  // get host-usable pointer to callback functions
  cufftCallbackLoadZ h_loadCallbackPtr;
  cudaMemcpyFromSymbol(&h_loadCallbackPtr, 
                       d_loadCallbackPtr, 
                       sizeof(h_loadCallbackPtr) );
  cudaDeviceSynchronize();
  
  
  
  // creating plan with callback
  cufftHandle plan;
  cufftCreate(&plan);
  
  size_t work_size;
  cufftMakePlan1d(plan, N, CUFFT_Z2Z, 1, &work_size); //cufftMakePlan1d(cufftHandle *plan, int nx, cufftType type, int batch)
  
  /*if (cufftPlan1d(&plan, N, CUFFT_C2C, 1) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: Plan creation failed");
    exit(EXIT_FAILURE);
  }*/
  cudaDeviceSynchronize();
  cufftResult status = cufftXtSetCallback(plan,
		     (void **) &h_loadCallbackPtr,
                     CUFFT_CB_LD_COMPLEX_DOUBLE,
                     NULL ); //<- here can be added structure with data needed for callback execution!
  if (status == CUFFT_LICENSE_ERROR)
  {
        printf("This sample requires a valid license file.\n");
        printf("The file was either not found, out of date, or otherwise invalid.\n");
        exit(EXIT_FAILURE);
  }
  
  
  
  
  
  
  
  
  
  
  
  cufftCallbackStoreZ h_storeCallbackPtr;
  cudaMemcpyFromSymbol(&h_storeCallbackPtr, 
                       d_storeCallbackPtr, 
                       sizeof(h_storeCallbackPtr));
  
  
  // creating plan with callback
  cufftHandle plan_back;
  cufftCreate(&plan_back);
  
  size_t work_size2;
  cufftMakePlan1d(plan_back, N, CUFFT_Z2Z, 1, &work_size2); //cufftMakePlan1d(cufftHandle *plan, int nx, cufftType type, int batch)
  
  cudaDeviceSynchronize();
#ifdef STORE_CB_NORMALIZE
  status = cufftXtSetCallback(plan_back,
		     (void **) &h_storeCallbackPtr,
                     CUFFT_CB_ST_COMPLEX_DOUBLE,
                     NULL ); //<- here can be added structure with data needed for callback execution!
  if (status == CUFFT_LICENSE_ERROR)
  {
        printf("This sample requires a valid license file.\n");
        printf("The file was either not found, out of date, or otherwise invalid.\n");
        exit(EXIT_FAILURE);
  }
#endif
  cudaDeviceSynchronize();
  
  
  
  // EXECUTE TRANSFORM FORWARD AND BACKWADR
  
  if (cufftExecZ2Z(plan, data_dev, data_dev, CUFFT_FORWARD) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
    exit(EXIT_FAILURE);;
  }
  cudaDeviceSynchronize();
  
  
  // copy data
  cudaMemcpy(data_host, data_dev, N*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  
  printf("fft data forward:\n");
  for (uint64_t ii = 0; ii < N; ii++) {
    if (N <= 32) printf("%lf + %lfj\n", cuCreal(data_host[ii]), cuCimag(data_host[ii]));
    fwrite(data_host+ii, sizeof(cuDoubleComplex),1,file1d);
  }
  
  
  // execute back
  if (cufftExecZ2Z(plan_back, data_dev, data_dev, CUFFT_INVERSE) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
    exit(EXIT_FAILURE);;
  }
  cudaDeviceSynchronize();
#ifndef STORE_CB_NORMALIZE
  // run kernel to normalize
  uint64_t threadsPerBlock;
  if (N >= 33554432)
    threadsPerBlock = 1024;
  else {
    threadsPerBlock = 128; // seems max grid size is ( 32768, ?, ? ) <- ????
  }
  dim3 dimBlock(threadsPerBlock,1,1);
  dim3 dimGrid( (N + threadsPerBlock - 1)/threadsPerBlock, 1, 1 ); // (numElements + threadsPerBlock - 1) / threadsPerBlock
  printf("initating wavefunction on host. Kernel invocation:\n");
  printf("threads Per block: %lu\n", threadsPerBlock);
  printf("blocks: %lu\n",(N + threadsPerBlock - 1)/threadsPerBlock);
  // filling with data
  kernel_normalize<<<dimGrid,dimBlock>>>(data_dev);
  //HANDLE_ERROR( cudaGetLastError() );
#endif
  
  // copy data
  cudaMemcpy(data_host, data_dev, N*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  
  printf("fft data backward:\n");
  for (uint64_t ii = 0; ii < N; ii++) {
    if (N <= 32) printf("%lf + %lfj\n", cuCreal(data_host[ii]), cuCimag(data_host[ii]));
    fwrite(data_host+ii, sizeof(cuDoubleComplex),1,file1d);
  }
  
  
  fclose(file1d);
  
  cudaFree(data_dev);
  cudaFreeHost(data_host);
  cudaDeviceSynchronize();
  
  cudaThreadExit();
  cudaDeviceSynchronize();
  
  return EXIT_SUCCESS;
}