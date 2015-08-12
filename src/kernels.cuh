#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include <stdio.h>
#include <cuda.h>
#include <cuComplex.h>

#include "cudautils.cuh"
#include "cuda_complex_ext.cuh"

// functions from host can't be called on device (even inline), but the same name of funtion on device is not allowed
__device__ static __inline__ double kx_dev(const uint16_t index) {
  return (index < NX/2) ? index * DKx : KxMIN + (index - NX/2) * DKx;
}



// kernels

/* ************************************************************************************************************************************* *
 * 																	 *
 * 							KERNELS TYPE Z									 *
 * 																	 *
 * ************************************************************************************************************************************* */
 // these kernels have in arguments only array of complex numbers!
__global__ void ker_gauss_1d(cufftDoubleComplex* data);
__global__ void ker_normalize_1d(cufftDoubleComplex* cufft_inverse_data); // normalize after CUFFT (only)
__global__ void ker_create_propagator_T(cuDoubleComplex* propagator_T_dev);
__global__ void ker_print_Z(cuDoubleComplex* arr_dev);



/* ************************************************************************************************************************************* *
 * 																	 *
 * 							KERNELS TYPE ZD									 *
 * 																	 *
 * ************************************************************************************************************************************* */
// these kernels have in arguments only array of complex numbers and pointer to double data (scalar or array as well)
//<- on device we have probably only pointers, because we cannot handle device memory explicit
__global__ void ker_modulus_wf_1d(cuDoubleComplex* complex_arr_dev, double* double_arr_dev);
__global__ void ker_modulus_pow2_wf_1d(cuDoubleComplex* complex_arr_dev, double* double_arr_dev);
__global__ void ker_arg_wf_1d(cuDoubleComplex* complex_arr_dev, double* double_arr_dev);
__global__ void ker_count_norm_wf_1d(cuDoubleComplex* complex_arr_dev, double* norm_dev);
__global__ void ker_normalize_1d(cufftDoubleComplex* data, double* norm);

__global__ void ker_energy_T_1d(cuDoubleComplex* wf_k, double* T_mean);
__global__ void ker_energy_Vext_1d(cuDoubleComplex* wf_k, double* Vext_mean);





/* ************************************************************************************************************************************* *
 * 																	 *
 * 							KERNELS TYPE ZZ									 *
 * 																	 *
 * ************************************************************************************************************************************* */
// these kernels have in argument two arrays of complex numbers
__global__ void ker_propagate(cuDoubleComplex* wf_momentum_dev, cuDoubleComplex* popagator_T_dev);





/* ************************************************************************************************************************************* *
 * 																	 *
 * 						    KERNELS TYPE operatorZD								 *
 * 																	 *
 * ************************************************************************************************************************************* */

 // these kernels take pointer to device operator and counts integral of psi* x operator(psi)
__global__ void ker_operator_mean_1d( dev_funcZ_ptr_t func, cuDoubleComplex* wf, double* mean );
__global__ void ker_T_wf(cuDoubleComplex* wf_momentum_dev, cuDoubleComplex* result_dev);
__global__ void ker_Vext_wf(cuDoubleComplex* wf_dev, cuDoubleComplex* result_dev);







/* ************************************************************************************************************************************* *
 * 																	 *
 * 							DEVICE-ONLY FUNCTIONS								 *
 * 																	 *
 * ************************************************************************************************************************************* */
// device-only functions and variables



/*
 * This is function representing element-wise action of operator T on the wavefunction.
 * cuDoubleComplex wf_element - element of wavefunction to perform operation
 * uint64_t index - index of element in wavefunction array
 * 
 */
__device__ __inline__ double operator_T_dev(cuDoubleComplex wf_element, uint64_t index) {
  return cuCreal(  cuCmul( cuConj(wf_element), cuCmul(wf_element, kx_dev(index)*kx_dev(index)) )  );
}
// copyable pointer to this function
__device__ dev_funcZ_ptr_t operator_T_dev_ptr = operator_T_dev;


__device__  __inline__ double operator_Vext_dev(cuDoubleComplex wf_element, uint64_t index) {
  return cuCreal( cuCmul( 
		  cuConj(wf_element), cuCmul(wf_element, (XMIN + index*DX)*(XMIN + index*DX)) 
		  ) );
}
// copyable pointer to this function
__device__ dev_funcZ_ptr_t operator_Vext_dev_ptr = operator_T_dev;




/* ************************************************************************************************************************************* *
 * 																	 *
 * 							CALLING KERNELS									 *
 * 																	 *
 * ************************************************************************************************************************************* */


/*
 * Special function to call kernels - more transparent code
 * 
 * TUTAJ BEDZIETRZEBA ZROBIC KILKA FUNKCJI DLA ROZNYCH RODZAJOW ARGUMENTOW ALBO POROBIC WSKAZNIKI NA TABLICE GLOBALNIE WIDOCZNE NA DEVICE
 * PIERWSZE ROZWIAZANIE MA TA ZALETE, ZE W ZALEZNOSCI OD ARGUMENTOW FUNKCJI MOZNA ZMIENIC JAKIES PARAMETRY WYWOLANIA FUNCKJI
 * POPRAWIC JESZCZE TRZEBA TO BY nie bylo else/if
 */
static inline void call_kernel_Z_1d(void(*kernel)(cuDoubleComplex*), cuDoubleComplex* data, cudaStream_t stream, uint32_t shared_mem=0, const int Nob=-1, const int BlkSz=-1) 
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
      if (shared_mem > 0) shared_mem /= (dimGrid.x + dimGrid.y + dimGrid.z - 2);
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
static inline void call_kernel_ZD_1d(void(*kernel)(cuDoubleComplex*, double*), cuDoubleComplex* complex_data, double* double_data, cudaStream_t stream, uint32_t shared_mem=0, const int Nob=-1, const int BlkSz=-1) 
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
      if (shared_mem > 0) shared_mem /= (dimGrid.x + dimGrid.y + dimGrid.z - 2);
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
static inline void call_kernel_ZZ_1d( void(*kernel)(cuDoubleComplex*, cuDoubleComplex*), cuDoubleComplex* complex_data1, cuDoubleComplex* complex_data2, cudaStream_t stream, uint32_t shared_mem=0, const int Nob=-1, const int BlkSz=-1) 
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
      if (shared_mem > 0) shared_mem /= (dimGrid.x + dimGrid.y + dimGrid.z - 2);
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

/*
 * Special function to call kernels - more transparent code
 * 
 * TUTAJ BEDZIETRZEBA ZROBIC KILKA FUNKCJI DLA ROZNYCH RODZAJOW ARGUMENTOW ALBO POROBIC WSKAZNIKI NA TABLICE GLOBALNIE WIDOCZNE NA DEVICE
 * PIERWSZE ROZWIAZANIE MA TA ZALETE, ZE W ZALEZNOSCI OD ARGUMENTOW FUNKCJI MOZNA ZMIENIC JAKIES PARAMETRY WYWOLANIA FUNCKJI
 * POPRAWIC JESZCZE TRZEBA TO BY nie bylo else/if
 */
static inline void call_kernel_operatorZD_1d( void(*kernel)(dev_funcZ_ptr_t, cuDoubleComplex*, double*),
					      dev_funcZ_ptr_t func,
					      cuDoubleComplex* wf,
					      double* result,
					      cudaStream_t stream,
					      uint32_t shared_mem=0,
					      const int Nob=-1,
					      const int BlkSz=-1) 
{
    
    if ( (Nob != -1) || (BlkSz != -1) ) {
      printf("using function parameters when invoking kernel!\n");
      (*kernel)<<<Nob, BlkSz, shared_mem, stream>>>(func, wf, result);
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
      if (shared_mem > 0) shared_mem /= (dimGrid.x + dimGrid.y + dimGrid.z - 2);
#ifdef DEBUG
      printf("\nKernel invocation:\n");
      printf("threads per block: %lu\n", threadsPerBlock);
      printf("blocks: %lu\n",(NX*NY*NZ + threadsPerBlock - 1)/threadsPerBlock);
      printf("pointer to function: %p\n",kernel);
#endif
      (*kernel)<<<dimGrid, dimBlock, shared_mem, stream>>>(func, wf, result);
    }
    HANDLE_ERROR( cudaGetLastError() );
    
    // przemyslec jak to zreobic najlepiej
    //cudaStreamSynchronize(stream); // <- czy to tutaj w ogole moze byc??? przeciez wtedy nie da sie wykonac kerneli asynchronicznie w jednym watku

}

#endif