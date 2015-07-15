#ifndef __GLOBAL_H__
#define __GLOBAL_H__

typedef void* (*Array_of_thrd_functions[])(void*);

typedef struct GlobalSettings {
  
  cudaStream_t* streams;
  pthread_barrier_t* barrier;
  
  // files
  char** filenames;
  FILE** files;
  
  // data structures on host
  double complex* psi_host;
  double statistics_host;
  // cross sections on host??
  
  // data structures on device
  cuDoubleComplex* complex_arr1_dev; // pointer on array holding wavefunction in device memory
  cuDoubleComplex* complex_arr2_dev;
  double* double_arr1_dev;
  cuDoubleComplex* propagator_T_dev; // array of constant factors e^-ik**2/2dt
  cuDoubleComplex* propagator_Vext_dev; // array of constant factors e^-iVextdt
  double* Vdip_dev; // array of costant factors <- count on host with spec funcs lib or use Abramowitz & Stegun approximation
  
  // cross sections on device
  
} GlobalSettings;

extern GlobalSettings* global_stuff;
extern pthread_barrier_t barrier;
extern pthread_barrier_t barrier_global;

// names
enum thread_id {SIMULATION_THRD, HELPER_THRD}; // enums in header!
enum stream_id {KERNEL_STREAM, MEMORY_STREAM};



#endif