#ifndef __GLOBAL_H__
#define __GLOBAL_H__


#include <cuda_runtime.h>

#include <stdint.h>
#include <complex.h>
#include <cuComplex.h>
#include <cufft.h>

// simulation parameters
#define NX ( (uint32_t) 1024 ) // type should be large enough to collect NX*NY*NZ
#define NY ( (uint32_t) 1 )
#define NZ ( (uint32_t) 1 )
#define DIM 1
#define REAL_TIME -1
#define IMAG_TIME -COMPLEX_I
// scalling the grid <- poprawic by nie bylo obliczen !!!
#define DX 1
#define XMAX (NX*DX/2)
#define XMIN (-XMAX)
#define DKx (6.283185307179586/(NX*DX))
#define KxMAX (6.283185307179586/(2*DX))
#define KxMIN (-6.283185307179586/(2*DX))


typedef void* (*Array_of_thrd_functions[])(void*);

typedef struct Globals {
  
  cudaStream_t* streams;
  pthread_barrier_t* barrier;
  cufftHandle* plans;
  
  // files
  char** filenames;
  FILE** files;
  uint8_t num_files;
  
  // memory maps
  double complex* init_wf_map; // map
  int init_wf_fd; // file descriptor
  
  // data structures on host
  double complex* wf_host;
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
  
  
  
  //scalars on host
  double mean_T_host;
  double mean_Vdip_host;
  double mean_Vext_host;
  double mean_Vcon_host;
  double norm_host;
  
  
  
  // scalars on device <- MOZE SKOPIOWAC TO NA DEVICE PRZEZ COPY FROM SYMBOL, BY BYLO LATWIEJ SIE TYM POSLUGIWAC ?!!!
  double* mean_T_dev;
  double* mean_Vdip_dev;
  double* mean_Vext_dev;
  double* mean_Vcon_dev;
  double* norm_dev;
  
  
} Globals;

extern Globals* global_stuff;
extern pthread_barrier_t barrier;// <- this barrier synchronizes only local threads that manage algorith and device
extern pthread_barrier_t barrier_global;
extern const uint8_t num_streams;
const uint8_t num_plans = 4;

// names
enum thread_id {SIMULATION_THRD, HELPER_THRD}; // enums in header!
enum stream_id {SIMULATION_STREAM, HELPER_STREAM};

enum plan_id {FORWARD_PSI, BACKWARD_PSI, FORWARD_DIPOLAR, BACKWARD_DIPOLAR};



#endif