#ifndef __CUDAUTILS_CUH__
#define __CUDAUTILS_CUH__

//#include <stdio.h>
//#include <cuda.h>
#include <cublas_v2.h>
#include <cufft.h>

/*
 * This header contains definitions of functions and macros for handling operational requirements.
 */

#include "cudautils.h" // include definition of functions to be linked in gcc compiled files.


/* ************************************************************************************************************************************* *
 * 																	 *
 * 							TIME MEASUREMENT								 *
 * 																	 *
 * ************************************************************************************************************************************* */

// macros

// !!! these macros need cudaEvent_t start, stop; to be defined globally!!!
// UWAGA TO DZIALA TYLKO NA DEFAULT STREAM!!! <- do niczego / nauczyc sie nvvp/nvprof
#define CUDATIMEIT_START \
			  cudaEventCreate(&start_t); \
			  cudaEventCreate(&stop_t); \
			  cudaEventRecord(start_t,0)

#define CUDATIMEIT_STOP \
			  cudaEventRecord(stop_t,0); \
			  cudaEventSynchronize(stop_t)


// function definitions
double print_cudatimeit(const char* message);
double fprint_cudatimeit(FILE* file);


/* ************************************************************************************************************************************* *
 * 																	 *
 * 							ERROR HANDLING									 *
 * 																	 *
 * ************************************************************************************************************************************* */


// handling errors in simple way
static inline void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
} 

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}


/* 
 * This function enables simple handling of cufftResult (status of cufft-library operation)
 * 
 * DEVELOPE IT: if there is error, it should show what the error is !!!
 */
static inline void CheckCufft( cufftResult cufft_res,
                         const char *file,
                         int line ) {
    if (cufft_res != CUFFT_SUCCESS) {
        printf( "CUFFT error in %s at line %d\n", file, line );
        
        if (cufft_res == CUFFT_INVALID_PLAN) {printf("CUFFT_INVALID_PLAN\n");}
        else if (cufft_res == CUFFT_ALLOC_FAILED) {printf("CUFFT_ALLOC_FAILED\n");}
        else if (cufft_res == CUFFT_INVALID_TYPE) {printf("CUFFT_INVALID_TYPE\n");}
        else if (cufft_res == CUFFT_INVALID_VALUE) {printf("CUFFT_INVALID_VALUE\n");}
        else if (cufft_res == CUFFT_INTERNAL_ERROR) {printf("CUFFT_INTERNAL_ERROR\n");}
        else if (cufft_res == CUFFT_EXEC_FAILED) {printf("CUFFT_EXEC_FAILED\n");}
        else if (cufft_res == CUFFT_SETUP_FAILED) {printf("CUFFT_SETUP_FAILED\n");}
        else if (cufft_res == CUFFT_INVALID_SIZE) {printf("CUFFT_INVALID_SIZE\n");}
        else if (cufft_res == CUFFT_UNALIGNED_DATA) {printf("CUFFT_UNALIGNED_DATA\n");}
        else if (cufft_res == CUFFT_INCOMPLETE_PARAMETER_LIST) {printf("INCOMPLETE_PARAMETER_LIST\n");}
        else if (cufft_res == CUFFT_INVALID_DEVICE) {printf("CUFFT_INVALID_DEVICE\n");}
        else if (cufft_res == CUFFT_NO_WORKSPACE) {printf("CUFFT_NO_WORKSPACE\n");}
        else if (cufft_res == CUFFT_NOT_IMPLEMENTED) {printf("CUFFT_NOT_IMPLEMENTED\n");}
        else if (cufft_res == CUFFT_PARSE_ERROR) {printf("PARSE_ERROR\n");}
        else if (cufft_res == CUFFT_LICENSE_ERROR) {printf("CUFFT_LICENSE_ERROR\n");}
        
        exit( EXIT_FAILURE );
    }
} 

#define CHECK_CUFFT( cufft_res ) (CheckCufft(cufft_res, __FILE__, __LINE__)) 



/* 
 * This function enables simple handling of cublasStatus_t (status of cublas-library operation)
 * 
 * DEVELOPE IT: if there is error, it should show what the error is !!!
 */
static inline void CheckCublas( cublasStatus_t status,
                         const char *file,
                         int line ) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf( "CUBLAS error in %s at line %d\n", file, line );
        exit( EXIT_FAILURE );
    }
} 

#define CHECK_CUBLAS( status ) (CheckCublas(status, __FILE__, __LINE__)) 

#endif