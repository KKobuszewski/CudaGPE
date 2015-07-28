#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include <stdio.h>
#include <cuda.h>
#include <cuComplex.h>

#include "cudautils.cuh"

// kernels

/* ************************************************************************************************************************************* *
 * 																	 *
 * 							KERNELS TYPE Z									 *
 * 																	 *
 * ************************************************************************************************************************************* */
 // these kernels have in arguments only array of complex numbers!
__global__ void ker_gauss_1d(cufftDoubleComplex* data);
__global__ void ker_normalize_1d(cufftDoubleComplex* cufft_inverse_data);
__global__ void ker_create_propagator_T(cuDoubleComplex* propagator_T_dev);
__global__ void ker_print_Z(cuDoubleComplex* arr_dev);



/* ************************************************************************************************************************************* *
 * 																	 *
 * 							KERNELS TYPE ZD									 *
 * 																	 *
 * ************************************************************************************************************************************* */
// these kernels have in arguments only array of complex numbers and pointer to double data (scalar or array as well)
//<- on device we have probably only pointers, because we cannot handle device memory explicit
__global__ void ker_modulus_pow2_wf_1d(cuDoubleComplex* complex_arr_dev, double* double_arr_dev);
__global__ void ker_arg_wf_1d(cuDoubleComplex* complex_arr_dev, double* double_arr_dev);
__global__ void ker_count_norm_wf_1d(cuDoubleComplex* complex_arr_dev, double* norm_dev);





/* ************************************************************************************************************************************* *
 * 																	 *
 * 							KERNELS TYPE ZZ									 *
 * 																	 *
 * ************************************************************************************************************************************* */
// these kernels have in argument two arrays of complex numbers
__global__ void ker_propagate(cuDoubleComplex* wf_momentum_dev, cuDoubleComplex* popagator_T_dev);


// device-only functions and variables





/* ************************************************************************************************************************************* *
 * 																	 *
 * 							CALLING KERNELS									 *
 * 																	 *
 * ************************************************************************************************************************************* */


//#define AUTOCALL_KERNEL_Z(kernel,data,stream) call_kernel_Z_1d(kernel,data,stream,-1,-1,-1)


/*
 * Special function to call kernels - more transparent code
 * 
 * TUTAJ BEDZIETRZEBA ZROBIC KILKA FUNKCJI DLA ROZNYCH RODZAJOW ARGUMENTOW ALBO POROBIC WSKAZNIKI NA TABLICE GLOBALNIE WIDOCZNE NA DEVICE
 * PIERWSZE ROZWIAZANIE MA TA ZALETE, ZE W ZALEZNOSCI OD ARGUMENTOW FUNKCJI MOZNA ZMIENIC JAKIES PARAMETRY WYWOLANIA FUNCKJI
 * POPRAWIC JESZCZE TRZEBA TO BY nie bylo else/if
 */
static inline void call_kernel_Z_1d(void(*kernel)(cuDoubleComplex*), cuDoubleComplex* data, cudaStream_t stream, const uint32_t shared_mem=0, const int Nob=-1, const int BlkSz=-1) 
{
    if ( (Nob != -1) || (BlkSz != -1) ) {
      (*kernel)<<<Nob, BlkSz, shared_mem, stream>>>(data);
    }
    else
    {
      uint64_t threadsPerBlock;
      if (NX*NY*NZ >= 33554432) //     <-  CHYBA TRZEBA TO ZAŁATWIĆ MAKREM NA POCZATKU PROGRAMU 
	threadsPerBlock = 1024;
      else if (NX*NY*NZ >= 33554432/2)
	threadsPerBlock = 512;
      else if (NX*NY*NZ >= 33554432/4)
	threadsPerBlock = 256;
      else 
	threadsPerBlock = 128; // seems max grid size is ( 32768, ?, ? ) <- ????
      
      dim3 dimBlock(threadsPerBlock,1,1);
      dim3 dimGrid( (NX*NY*NZ + threadsPerBlock - 1)/threadsPerBlock, 1, 1 ); // (numElements + threadsPerBlock - 1) / threadsPerBlock
      //if (shared_mem > 0) shared_mem /= (dimGrid.x + dimGrid.y + dimGrid.z - 2);
#ifdef DEBUG
      printf("\nKernel invocation:\n");
      printf("threads per block: %lu\n", threadsPerBlock);
      printf("blocks: %lu\n",(NX*NY*NZ + threadsPerBlock - 1)/threadsPerBlock);
      printf("pointer to function: %p\n",kernel);
      
      if (shared_mem > 0) printf("shared_mem: %u\n",shared_mem);
#endif
      (*kernel)<<<dimGrid, dimBlock, shared_mem, stream>>>(data);
    }
    HANDLE_ERROR( cudaGetLastError() );
    
    // przemyslec jak to zreobic najlepiej
    //cudaStreamSynchronize(stream);

}

/*
 * Special function to call kernels - more transparent code
 * 
 * TUTAJ BEDZIETRZEBA ZROBIC KILKA FUNKCJI DLA ROZNYCH RODZAJOW ARGUMENTOW ALBO POROBIC WSKAZNIKI NA TABLICE GLOBALNIE WIDOCZNE NA DEVICE
 * PIERWSZE ROZWIAZANIE MA TA ZALETE, ZE W ZALEZNOSCI OD ARGUMENTOW FUNKCJI MOZNA ZMIENIC JAKIES PARAMETRY WYWOLANIA FUNCKJI
 * POPRAWIC JESZCZE TRZEBA TO BY nie bylo else/if
 */
static inline void call_kernel_ZD_1d(void(*kernel)(cuDoubleComplex*, double*), cuDoubleComplex* complex_data, double* double_data, cudaStream_t stream, const uint32_t shared_mem=0, const int Nob=-1, const int BlkSz=-1) 
{
    if ( (Nob != -1) || (BlkSz != -1) ) {
      (*kernel)<<<Nob, BlkSz, shared_mem, stream>>>(complex_data, double_data);
    }
    else
    {
      uint64_t threadsPerBlock;
      if (NX*NY*NZ >= 33554432) //     <-  CHYBA TRZEBA TO ZAŁATWIĆ MAKREM NA POCZATKU PROGRAMU 
	threadsPerBlock = 1024;
      else if (NX*NY*NZ >= 33554432/2)
	threadsPerBlock = 512;
      else if (NX*NY*NZ >= 33554432/4)
	threadsPerBlock = 256;
      else 
	threadsPerBlock = 128; // seems max grid size is ( 32768, ?, ? ) <- ????
      
      dim3 dimBlock(threadsPerBlock,1,1);
      dim3 dimGrid( (NX*NY*NZ + threadsPerBlock - 1)/threadsPerBlock, 1, 1 ); // (numElements + threadsPerBlock - 1) / threadsPerBlock
#ifdef DEBUG
      printf("\nKernel invocation:\n");
      printf("threads per block: %lu\n", threadsPerBlock);
      printf("blocks: %lu\n",(NX*NY*NZ + threadsPerBlock - 1)/threadsPerBlock);
      printf("pointer to function: %p\n",kernel);
      
      if (shared_mem > 0) printf("shared_mem: %u\n",shared_mem);
#endif
      (*kernel)<<<dimGrid, dimBlock, shared_mem, stream>>>(complex_data, double_data);
    }
    HANDLE_ERROR( cudaGetLastError() );
    
    // przemyslec jak to zreobic najlepiej
    //cudaStreamSynchronize(stream); // <- czy to tutaj w ogole moze byc??? przeciez wtedy nie da sie wykonac kerneli asynchronicznie w jednym watku

}

/*
 * Special function to call kernels - more transparent code
 * 
 * TUTAJ BEDZIETRZEBA ZROBIC KILKA FUNKCJI DLA ROZNYCH RODZAJOW ARGUMENTOW ALBO POROBIC WSKAZNIKI NA TABLICE GLOBALNIE WIDOCZNE NA DEVICE
 * PIERWSZE ROZWIAZANIE MA TA ZALETE, ZE W ZALEZNOSCI OD ARGUMENTOW FUNKCJI MOZNA ZMIENIC JAKIES PARAMETRY WYWOLANIA FUNCKJI
 * POPRAWIC JESZCZE TRZEBA TO BY nie bylo else/if
 */
static inline void call_kernel_ZZ_1d( void(*kernel)(cuDoubleComplex*, cuDoubleComplex*), cuDoubleComplex* complex_data1, cuDoubleComplex* complex_data2, cudaStream_t stream, const uint32_t shared_mem=0, const int Nob=-1, const int BlkSz=-1) 
{
    
    if ( (Nob != -1) || (BlkSz != -1) ) {
      printf("using function parameters when invoking kernel!\n");
      (*kernel)<<<Nob, BlkSz, shared_mem, stream>>>(complex_data1, complex_data2);
    }
    else
    {
      uint64_t threadsPerBlock;
      if (NX*NY*NZ >= 33554432) //     <-  CHYBA TRZEBA TO ZAŁATWIĆ MAKREM NA POCZATKU PROGRAMU 
	threadsPerBlock = 1024;
      else if (NX*NY*NZ >= 33554432/2)
	threadsPerBlock = 512;
      else if (NX*NY*NZ >= 33554432/4)
	threadsPerBlock = 256;
      else 
	threadsPerBlock = 128; // seems max grid size is ( 32768, ?, ? ) <- ????
      
      dim3 dimBlock(threadsPerBlock,1,1);
      dim3 dimGrid( (NX*NY*NZ + threadsPerBlock - 1)/threadsPerBlock, 1, 1 ); // (numElements + threadsPerBlock - 1) / threadsPerBlock
#ifdef DEBUG
      printf("\nKernel invocation:\n");
      printf("threads per block: %lu\n", threadsPerBlock);
      printf("blocks: %lu\n",(NX*NY*NZ + threadsPerBlock - 1)/threadsPerBlock);
      printf("pointer to function: %p\n",kernel);
#endif
      (*kernel)<<<dimGrid, dimBlock, shared_mem, stream>>>(complex_data1, complex_data2);
    }
    HANDLE_ERROR( cudaGetLastError() );
    
    // przemyslec jak to zreobic najlepiej
    //cudaStreamSynchronize(stream); // <- czy to tutaj w ogole moze byc??? przeciez wtedy nie da sie wykonac kerneli asynchronicznie w jednym watku

}


#endif