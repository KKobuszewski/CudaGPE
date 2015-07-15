#include <stdio.h>
#include <sched.h>
#include <errno.h>
#include <unistd.h>

#include <cuda.h>

#include "cudautils.cuh"

/* ************************************************************************************************************************************* *
 * 																	 *
 * 							DEVICE PROPERTIES								 *
 * 																	 *
 * ************************************************************************************************************************************* */

int getNumberOfCpus()
{
    long nprocs       = -1;
    long nprocs_max   = -1;

# ifdef _SC_NPROCESSORS_ONLN
    nprocs = sysconf( _SC_NPROCESSORS_ONLN );
    if ( nprocs < 1 )
    {
        //std::cout << "Could not determine number of CPUs on line. Error is  " << strerror( errno ) << std::endl;
        return 0;
    }

    nprocs_max = sysconf( _SC_NPROCESSORS_CONF );

    if ( nprocs_max < 1 )
    {
        //std::cout << "Could not determine number of CPUs in host. Error is  " << strerror( errno ) << std::endl;
        return 0;
    }

    //std::cout << nprocs < " of " << nprocs_max << " online" << std::endl;
    return nprocs; 

#else
    //std::cout << "Could not determine number of CPUs" << std::endl;
    return 0;
#endif
}


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
  printf("CPU cores on host: %d\n",getNumberOfCpus());
}


// core_id = 0, 1, ... n-1, where n is the system's number of cores

int stick_this_thread_to_core(int core_id) {
   int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
   if (core_id < 0 || core_id >= num_cores)
      return EINVAL;

   cpu_set_t cpuset;
   CPU_ZERO(&cpuset);
   CPU_SET(core_id, &cpuset);

   pthread_t current_thread = pthread_self();    
   return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
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