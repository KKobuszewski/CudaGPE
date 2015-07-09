#include <stdio.h>
#include <cuda.h>

#include "cudautils.cuh"

/* ************************************************************************************************************************************* *
 * 																	 *
 * 							DEVICE PROPERTIES								 *
 * 																	 *
 * ************************************************************************************************************************************* */

/*
 * Checks properies of device.
 */
void print_device() {
  
  cudaDeviceProp prop;
  int whichDevice;
  int SMversion;
  
  // get device information
  HANDLE_ERROR( cudaGetDevice(&whichDevice) );
  HANDLE_ERROR( cudaGetDeviceProperties(&prop, whichDevice) );
  
  // print device information
  if (!prop.deviceOverlap) {
    printf("Device will not handle overlaps!");
  }
  SMversion = prop.major << 4 + prop.minor;
  printf("GPU[%d] %s supports SM %d.%d", whichDevice, prop.name, prop.major, prop.minor);
  printf(", %s GPU Callback Functions\n", (SMversion >= 0x11) ? "capable" : "NOT capable");
}

/* ************************************************************************************************************************************* *
 * 																	 *
 * 							TIME MEASUREMENT								 *
 * 																	 *
 * ************************************************************************************************************************************* */

extern cudaEvent_t start_t;
extern cudaEvent_t stop_t;

double print_cudatimeit(const char* message) {
  float computationTime;
  cudaEventElapsedTime(&computationTime, start_t, stop_t);
  printf( "time of %-60s %lf s\n", message, (double) computationTime );
  return (double) computationTime;
}

double fprint_cudatimeit(FILE* file) {
  float computationTime;
  cudaEventElapsedTime(&computationTime, start_t, stop_t);
  fwrite(&computationTime, (size_t) sizeof(double), 1, file);
  return (double) computationTime;
}