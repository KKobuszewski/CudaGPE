#include <stdio.h>
#include <cuda.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <stdint.h>


__global__ void staticReverse(cuDoubleComplex *d, uint64_t n)
{
  __shared__ cuDoubleComplex s[64];
  uint64_t t = threadIdx.x + blockIdx.x*blockDim.x;
  uint64_t tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

__global__ void dynamicReverse(cuDoubleComplex *d, uint64_t n)
{
  extern __shared__ cuDoubleComplex s[];
  uint64_t t = threadIdx.x + blockIdx.x*blockDim.x;
  uint64_t tr = n-t-1;
  printf("tr: %d",tr);
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

int main(void)
{
  const uint64_t n = 256*512; // maksymalnie 256 * 512
  cuDoubleComplex a[n], r[n], d[n];
  
  for (uint64_t i = 0; i < n; i++) {
    a[i] = make_cuDoubleComplex( (double) i, 0. );
    r[i] = make_cuDoubleComplex( (double) n-i-1, 0. );
    d[i] = make_cuDoubleComplex( 0. , 0.);
  }
  
  cuDoubleComplex *d_d;
  cudaMalloc(&d_d, n * sizeof(cuDoubleComplex)); 
  
  printf("static reverse\n");
  // run version with static shared memory
  cudaMemcpy(d_d, a, n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  staticReverse<<<512,n/512>>>(d_d, n);
  cudaMemcpy(d, d_d, n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  //for (uint64_t i = 0; i < n; i++) 
  //  if ( cuCreal(d[i]) != cuCreal(r[i]) ) printf("Error: d[%d]!=r[%d] (%lf, %lf)n", i, i,cuCreal(d[i]), cuCreal(r[i])); 
  
  printf("dynamic reverse\n");
  // run dynamic shared memory version
  cudaMemcpy(d_d, a, n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  
  // WYTESTOWAC odpalenie tego kernela z parametrami dimGrid(512,1,1)/dimBlocks(128,1,1)/3KB = 3*(1<<10) KB Shared Mem per thread !!!
  printf("sizeof cuDoubleComplex: %d\n",sizeof(cuDoubleComplex));
  printf("memory needed: %d\n",n*sizeof(cuDoubleComplex));
  printf("memory needed per block: %d\n",32*n*sizeof(cuDoubleComplex)/512);
  printf("shared memory : %d\n",96*1024);
  dynamicReverse<<<512,n/512,n*sizeof(cuDoubleComplex)/512>>>(d_d, n); 
  
  
  cudaMemcpy(d, d_d, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  //for (uint64_t i = 0; i < n; i++) 
  //  if ( cuCreal(d[i]) != cuCreal(r[i]) ) printf("Error: d[%d]!=r[%d] (%lf, %lf)n", i, i,cuCreal(d[i]), cuCreal(r[i]));
  return 0;
}
