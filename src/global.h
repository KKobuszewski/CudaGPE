#ifndef __GLOBAL_H__
#define __GLOBAL_H__


#include <cuda_runtime.h>

#include <stdint.h>
#include <complex.h>
#include <cuComplex.h>
#include <cufft.h>

#include "fileIO.h"



// simulation parameters
#define NX ( (uint32_t) 1024 ) // type should be large enough to collect NX*NY*NZ
#define NY ( (uint32_t) 1 )
#define NZ ( (uint32_t) 1 )
#define DIM 1

// evolution type
//#define REAL_TIME ( (double complex) -I)
//#define IMAG_TIME ( (double complex) -1)
//#define REAL_TIME -I
//#define IMAG_TIME -1

#ifndef EVOLUTION
//#define EVOLUTION REAL_TIME
#endif

// TIMESTEP LENGTH
#define DT ((double) 1e-6)

// scalling the grid <- poprawic by nie bylo obliczen !!!
#define XMAX ((double) .5)
#define XMIN (-XMAX)
#define DX ((double) (XMAX - XMIN)/(NX))
#define DKx ((double) 6.283185307179586/(XMAX-XMIN))
#define KxMAX ((double) 3.14159265358979323846/(DX))
#define KxMIN ((double) -3.14159265358979323846/(DX))
#define OMEGA ( 0.5*3.14159265358979323846*((double) NX)/ (2*XMAX*XMAX) )

#define G_CONTACT ((double) 1. )
#define G_DIPOLAR ((double) 1. )



#define M_2PI ((double) 6.283185307179586)
#define SQRT_2PI ((double) 2.5066282746310002)
#define INV_SQRT_2PI ((double) 0.3989422804014327)



/*
const uint64_t Nx, Ny, Nz;
const double dx, dkx;
const double xmin, xmax;
const double kxmin, kxmax;
const double dt;
*/


static inline double kx(const uint16_t index) {
  return (index < NX/2) ? index * DKx : KxMIN + (index - NX/2) * DKx;
}




// typedef for function pointers to device functions
typedef double (*dev_funcZ_ptr_t)(cuDoubleComplex, uint64_t); // data, index

typedef void* (*Array_of_thrd_functions[])(void*);

typedef struct Globals {
  
  cudaStream_t* streams;
  pthread_barrier_t* barrier;
  cufftHandle* plans;
  
  // files
  char** filenames;
  uint8_t num_files;
  
  // memory maps
  double complex* init_wf_map; // map
  int init_wf_fd; // file descriptor
  int wf_save_fd;
  
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


// global variables declarated in main.c
extern Globals* global_stuff;
extern pthread_barrier_t barrier;// <- this barrier synchronizes only local threads that manage algorithm and device
extern pthread_barrier_t barrier_global;
extern const uint8_t num_streams;
extern cudaStream_t* streams;
extern double complex* wf_mmap;
extern double complex* init_wf_mmap;
extern struct_file** files; // declared in main.c
const uint8_t num_plans = 4;

// enums
enum thread_id {SIMULATION_THRD, HELPER_THRD};
enum stream_id {SIMULATION_STREAM, HELPER_STREAM};
enum plan_id {FORWARD_PSI, BACKWARD_PSI, FORWARD_DIPOLAR, BACKWARD_DIPOLAR};
enum file_id {SIM_PARAMS_FILE, STATS_FILE, WF_FRAMES_FILE, PROPAGATORS_FILE};



#endif