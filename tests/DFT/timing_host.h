#ifndef __TIMING_HOST_H__
#define __TIMING_HOST_H__

// this header includes simplified interface to time the host functions

#define TIMEIT_START (start_t=clock())
#define TIMEIT_END(x) x = (clock()-start_t)

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

#endif