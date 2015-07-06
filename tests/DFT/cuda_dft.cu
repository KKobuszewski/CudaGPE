#include  <cufft.h> 


#define  NX  256 
#define  BATCH  10 



void cuda_dft_1d() {
  
  cufftHandle plan; 
  cufftComplex *data; 
  
  cudaMalloc((void**)&data,  sizeof(cufftComplex)*NX*BATCH );
  
  /* Create   a  1D  FFT  plan. */ 
  cufftPlan1d(&plan, NX, CUFFT C2C, BATCH);
  
  /* Use  the  CUFFT  plan  to  transform  the  signal  in place. */ 
  cufftExecC2C(plan, data, data, CUFFT FORWARD); 
  
  /* Destroy  the  CUFFT  plan. */ 
  cufftDestroy(plan); 
  cudaFree(data);
}

/*
#define NX 64
#define NY 64
#define NZ 128

cufftHandle plan;
cufftComplex *data1, *data2;
cudaMalloc((void**)&data1, sizeof(cufftComplex)*NX*NY*NZ);
cudaMalloc((void**)&data2, sizeof(cufftComplex)*NX*NY*NZ);
// Create a 3D FFT plan. 
cufftPlan3d(&plan, NX, NY, NZ, CUFFT_C2C);

// Transform the first signal in place.
cufftExecC2C(plan, data1, data1, CUFFT_FORWARD);

// Transform the second signal using the same plan.
cufftExecC2C(plan, data2, data2, CUFFT_FORWARD);

// Destroy the cuFFT plan.
cufftDestroy(plan);
cudaFree(data1); cudaFree(data2);
*/