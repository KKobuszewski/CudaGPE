#include <stdio.h>
#include <cuda.h>
#include <cuComplex.h>
#include <cuda_runtime.h>


__global__ void staticReverse(cuDoubleComplex *d, int n)
{
  __shared__ cuDoubleComplex s[64];
  int t = threadIdx.x + blockIdx.x*blockDim.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

__global__ void dynamicReverse(cuDoubleComplex *d, int n)
{
  extern __shared__ cuDoubleComplex s[];
  int t = threadIdx.x + blockIdx.x*blockDim.x;
  int tr = n-t-1;
  printf("tr: %d",tr);
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

int main(void)
{
  const int n = 64*512; // 2**15
  cuDoubleComplex a[n], r[n], d[n];
  
  for (int i = 0; i < n; i++) {
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
  for (int i = 0; i < n; i++) 
    if ( cuCreal(d[i]) != cuCreal(r[i]) ) printf("Error: d[%d]!=r[%d] (%lf, %lf)n", i, i,cuCreal(d[i]), cuCreal(r[i])); 
  
  printf("dynamic reverse\n");
  // run dynamic shared memory version
  cudaMemcpy(d_d, a, n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  
  // WYTESTOWAC odpalenie tego kernela z parametrami dimGrid(512,1,1)/dimBlocks(128,1,1)/3KB = 3*(1<<10) KB Shared Mem per thread !!!
  printf("sizeof cuDoubleComplex: %d",sizeof(cuDoubleComplex));
  printf("memory needed: %d",n*sizeof(cuDoubleComplex));
  dynamicReverse<<<512,n/512,n*sizeof(cuDoubleComplex)/512>>>(d_d, n); 
  
  
  cudaMemcpy(d, d_d, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if ( cuCreal(d[i]) != cuCreal(r[i]) ) printf("Error: d[%d]!=r[%d] (%lf, %lf)n", i, i,cuCreal(d[i]), cuCreal(r[i]));
}
