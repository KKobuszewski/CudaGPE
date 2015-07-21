#include <stdio.h>
#include <cuda.h>
#include <cuComplex.h>
#include <cuda_runtime.h>


__global__ void staticReverse(cuDoubleComplex *d, int n)
{
  __shared__ cuDoubleComplex s[64];
  cuDoubleComplex t = make_cuDoubleComplex( (double)threadIdx.x, (double)(n-threadIdx.x-1) );
  cuDoubleComplex tr = make_cuDoubleComplex( (double)(n-threadIdx.x-1), (double)threadIdx.x );
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

__global__ void dynamicReverse(cuDoubleComplex *d, int n)
{
  extern __shared__ cuDoubleComplex s[];
  cuDoubleComplex t = make_cuDoubleComplex(threadIdx.x, n-threadIdx.x-1);
  cuDoubleComplex tr = make_cuDoubleComplex(n-t-1, threadIdx);
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

int main(void)
{
  const int n = 64;
  cuDoubleComplex a[n], r[n], d[n];
  
  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1;
    d[i] = 0;
  }
  
  cuDoubleComplex *d_d;
  cudaMalloc(&d_d, n * sizeof(cuDoubleComplex)); 
  
  // run version with static shared memory
  cudaMemcpy(d_d, a, n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  staticReverse<<<1,n>>>(d_d, n);
  cudaMemcpy(d, d_d, n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);
  
  // run dynamic shared memory version
  cudaMemcpy(d_d, a, n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  
  // WYTESTOWAC odpalenie tego kernela z parametrami dimGrid(512,1,1)/dimBlocks(128,1,1)/3KB = 3*(1<<10) KB Shared Mem per thread !!!
  printf("sizeof cuDoubleComplex: %d",sizeof(cuDoubleComplex));
  printf("memory needed: %d",n*sizeof(cuDoubleComplex));
  dynamicReverse<<<1,n,n*sizeof(cuDoubleComplex)>>>(d_d, n);
  
  
  
  cudaMemcpy(d, d_d, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);
}