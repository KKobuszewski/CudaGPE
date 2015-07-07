
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cuComplex.h>

int main(){
  
  uint64_t N = 1<<26;
  printf("N=%lu\n",N);
  
  cudaDeviceReset();
  
  cuDoubleComplex* data_host;
  cuDoubleComplex* data_dev1;
  cuDoubleComplex* data_dev2;
  cuDoubleComplex* data_dev3;
  cuDoubleComplex* data_dev4;
  printf("sizeof cuDoubleComplex %lu\n",sizeof(cuDoubleComplex));
  printf("sizeof memory %lu\n",sizeof(cuDoubleComplex)*N);
  cudaHostAlloc((void**) &data_host, sizeof(cuDoubleComplex)*N, cudaHostAllocDefault);
  
  
  cudaMalloc((void**) &data_dev1, sizeof(cuDoubleComplex)*N);
  printf("1st array\n");
  /*cudaMemcpyAsync 	( 	void *  	dst,
		const void *  	src,
sizeof(cuDoubleComplex)*N,)*/
  cudaMalloc((void**) &data_dev2, sizeof(cuDoubleComplex)*N);
  printf("2nd array\n");
  cudaMalloc((void**) &data_dev3, sizeof(cuDoubleComplex)*N);
  printf("3rd array\n");
  cudaMalloc((void**) &data_dev4, sizeof(cuDoubleComplex)*N);
  printf("4th array\n");
  
   	
  
  
  cudaFreeHost(data_host);
  cudaFree(data_dev1);
  cudaFree(data_dev2);
  cudaFree(data_dev3);
  cudaFree(data_dev4);
  
  return EXIT_SUCCESS;
}