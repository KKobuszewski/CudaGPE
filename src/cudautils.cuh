#ifndef __CUDAUTILS_CUH__
#define __CUDAUTILS_CUH__

//#include <stdio.h>
//#include <cuda.h>

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
// UWAGA TO DZIALA TYLKO NA DEFAULT STREAM!!! <- do niczego / nauczyc sie nvpp
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
static void HandleError( cudaError_t err,
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
  
  





#endif