#ifndef __GLOBAL_H__
#define __GLOBAL_H__


#include <cuda_runtime.h>

#include <stdint.h>
#include <complex.h>
#include <cuComplex.h>
#include <cufft.h>

#include "fileIO.h"



// simulation parameters
#define NX ( (uint32_t) 2048 ) // type should be large enough to collect NX*NY*NZ
#define NY ( (uint32_t) 1 )
#define NZ ( (uint32_t) 1 )
#define DIM 1

// evolution type




// scalling the grid <- poprawic by nie bylo obliczen !!!
// #ifdef V_CON
// #define OMEGA ((double) 1. )
// #define XMAX ((double) 5./sqrt(OMEGA) )
// #else
#define XMAX ((double) .5)
//#define OMEGA ( 0.5*3.14159265358979323846*((double) NX)/ (2*XMAX*XMAX) )
#define OMEGA ((double) 500. )
// #endif
#define XMIN (-XMAX)
#define DX ((double) (XMAX - XMIN)/(NX))
#define DKx ((double) 6.283185307179586/(XMAX-XMIN))
#define KxMAX ((double) 3.14159265358979323846/(DX))
#define KxMIN ((double) -3.14159265358979323846/(DX))



// TIMESTEP LENGTH
#ifdef IMAG_TIME
#define DT ((double) 1e-5/OMEGA)
#else
#define DT ((double) 1e-8)
#endif



#define G_CONTACT 100.*sqrt(OMEGA)/(XMAX - XMIN) // g contact in oscilatory units
#define G_DIPOLAR 0. // now it is equal to add



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
enum file_id {SIM_PARAMS_FILE, STATS_FILE, WF_FRAMES_FILE, PROPAGATORS_FILE,WF_K_FILE};



#endif