#ifndef __CUDAUTILS_H__
#define __CUDAUTILS_H__


/*
 * This header contains definitions of CUDA functions that would be linked into gcc compiled files.
 */


#include <helper_functions.h> // destination dir: -I /usr/local/cuda/samples/common/inc
#include <helper_cuda.h> // destionation dir: -I /usr/local/cuda/samples/common/inc


/* ************************************************************************************************************************************* *
 * 																	 *
 * 							TIME MEASUREMENT								 *
 * 																	 *
 * ************************************************************************************************************************************* */

// macros

#define TIMEIT_START (start_t=clock())
#define TIMEIT_END(x) x = (clock()-start_t)

// functions (inline to be more efficient)
/*
 * Prints variable of type clock_t with user-defined message.
 * It's global inline function (works like marco, but is more programmer-friendly), so it must be defined in every compilation unit with specifier extern.
 */
inline void print_timeit(clock_t time_to_print, const char* message) {
  printf( "time of %-60s %lf s\n", message, time_to_print/((double)CLOCKS_PER_SEC) );
}

/*
 * Interface to save clock_t variable into an opened BINARY file. (!!! in seconds !!!)
 */
inline void fwrite_timeit(clock_t time, FILE* file) {
  double time_doub = time/((double)CLOCKS_PER_SEC);
  fwrite(&time_doub, (size_t) sizeof(double), 1, file);
}


/* ************************************************************************************************************************************* *
 * 																	 *
 * 							DEVICE PROPERTIES								 *
 * 																	 *
 * ************************************************************************************************************************************* */

// functions for checking device properties
void print_device();





#endif