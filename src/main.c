#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <limits.h>
//#include <omp.h>

#include <cuda_runtime.h>
#include "helper_cuda.h"

#include "cudautils.h"
#include "global.h"
#include "simulation.h"
#include "fileIO.h"


// global variables declarations
Globals* global_stuff;
const char* thread_names[] = {"SIMULATION_THRD","HELPER_THRD"};
const char* stream_names[] = {"SIMULATION_STREAM","HELPER_STREAM"};


// set parameters - przepisac czesciowo na makra
const uint8_t num_streams = 2;
const uint8_t num_threads = 2; // except main thread


// threads
pthread_barrier_t barrier_global;
pthread_attr_t attr;
Array_of_thrd_functions thread_funcs = {simulation_thread, helper_thread};// type Array_of_thrd_functions is defined in global.h

// cuda streams
cudaStream_t* streams;



// mmap of wavefunction to be saved
double complex* wf_mmap;
double complex* init_wf_mmap;
struct_file** files;



void sig_handler(int signo)
{
  printf("\nPOSIX signal catched!\n");
  printf("received signal: ");
    if (signo == SIGINT) printf("%d SIGINT\n",signo);
    if (signo == SIGHUP) printf("%d SIGHUP\n",signo);
    if (signo == SIGQUIT) printf("%d SIGQUIT\n",signo);
    if (signo == SIGILL) printf("%d SIGILL\n",signo);
    if (signo == SIGABRT) printf("%d SIGABRT\n",signo);
    if (signo == SIGFPE) printf("%d SIGFPET\n",signo);
    if (signo == SIGKILL) printf("%d SIGKILL\n",signo);
    if (signo == SIGSEGV) printf("%d SIGSEGV\nSegmentation fault on HOST! (core dumped)\n",signo);
    if (signo == SIGSTOP) printf("%d SIGSTOP\n",signo);
  
  // close files
  mmap_destroy(global_stuff->init_wf_fd, init_wf_mmap, NX*NY*NZ * sizeof(double complex));
  mmap_destroy(global_stuff->wf_save_fd, wf_mmap, NX*NY*NZ * sizeof(double complex));
  close_struct_files(files, global_stuff->num_files);
  
  cudaThreadExit();
  cudaDeviceReset();
  
  printf("Main: program completed. Exiting...\n");
  exit(0);
}

/*
 * This function sets the same signal handler to all signals that can be catched
 * sig_handler must be a poiter to signal handler function (just a name of function)
 * (I think this is the most useful solition if we need to protect GPU memory)
 */
void set_signals_same() {
  printf("\nsetting signal actions.\n");
  for(int signo = 0; signo < 31; signo++) {
#ifdef DEBUG
    if (signo == SIGHUP) printf("%d SIGHUP\n",signo);
    if (signo == SIGINT) printf("%d SIGINT\n",signo);
    if (signo == SIGQUIT) printf("%d SIGQUIT\n",signo);
    if (signo == SIGILL) printf("%d SIGILL\n",signo);
    if (signo == SIGABRT) printf("%d SIGABRT\n",signo);
    if (signo == SIGFPE) printf("%d SIGFPET\n",signo);
    if (signo == SIGKILL) printf("__global__ void ker_energy_T_1d(cuDoubleComplex* wf_k, double* T_mean)%d SIGKILL\n",signo);
    if (signo == SIGSEGV) printf("%d SIGSEGV\n",signo);
    if (signo == SIGSTOP) printf("%d SIGSTOP\n",signo);
    if (signal(signo, sig_handler) == SIG_ERR) printf("\ncan't catch %d\n", signo);
#else
    signal(signo, sig_handler);
#endif
  }
  
}

/*
 * main function in main thread will only manage another threads - this allows having heterogenous and multistreamed application.
 * Application is dedicated only for single device architecture.
 * 
 * Command line arguments:
 * (for example path to a file with initial wavefunction - in case there is no initial wavefunction it will create new wavefunction)
 * 
 * return: EXIT_SUCCESS (if program ends with no bugs)
 */
int main(int argc, char* argv[]) {
    
#ifdef DEBUG
  printf(" ++++++++++++++++++++++++++++++++ DEBUG MODE ++++++++++++++++++++++++++++++++\n");
#endif
  
  //if (num_threads > PTHREAD_THREADS_MAX) { printf("to many threads!"); exit(0); } <- look for in limits.h ???
  
  // print device properties
  print_device();
  
  // clear and init device
  cudaDeviceReset(); // we want to be certain of proper behaviour of the device
  //cudaDeviceSynchronize();
  // look for some goods solutions for initializing device
  gpuDeviceInit(0);
  
  // make stucture to pass all variables in program
  global_stuff = (Globals*) malloc( (size_t) sizeof(Globals));
  // fill with known information
  global_stuff->init_wf_fd = -1;
  
  printf("\n");
#ifdef IMAG_TIME
  printf("****************************************************************************************************************\n");
  printf("*                                       IMAGINARY TIME EVOLUTION                                               *\n");
#ifdef IMPINT
  printf("*                                           PHASE IMPRINTING                                                   *\n");
#endif
  printf("****************************************************************************************************************\n");
#endif
  printf("Simulation params: \n");
  printf("dimensions: %u\n", DIM);
  printf("lattice points in direction x: %u\n", NX);
  printf("lattice points in direction y: %u\n", NY);
  printf("lattice points in direction z: %u\n", NZ);
  printf("total number of points in a lattice: %u, 2**%u\n", NX*NY*NZ, (uint32_t) ( log(NX*NY*NZ)/log(2) ) );
  printf("[xmin, xmax] : [%.15f, %.15f]\n", XMIN, XMAX);
  printf("dx: %.15f\n",DX);
  printf("[kxmin, kxmax] : [%.15f, %.15f]\n", KxMIN, KxMAX);
  printf("dkx: %.15f\n",DKx);
  //printf("width of gauss in positions space (points on lattice): %.15f\n");
  //printf("width of gauss in positions space (points on lattice): %.15f\n");
  printf("harmonic potential angular freq.: %.15f", OMEGA);
  printf("\n");
  
  
  // parse command line args
  printf("%d command line arguments:\n", argc);
  for (int ii = 0; ii < argc; ii++) {
    printf("%d. :\t%s", ii, argv[ii]);
    
    // open file with wavefunction to be read
    if (ii == 1) {
      printf("\tinitial wavefunction will be loaded from file %s", argv[ii]);
      global_stuff->init_wf_fd = mmap_create(argv[ii],					// in fileIO.c
					     (void**) &(init_wf_mmap),
					     NX*NY*NZ * sizeof(double complex),
					     PROT_READ, MAP_PRIVATE);
#ifdef DEBUG
      printf("\n\t\t\t\tsample of mmaped initial wavefunction: %lf + %lfj\n", creal(init_wf_mmap[1000]), cimag(init_wf_mmap[1000]));
#endif
      
    }
    
    /*
     * TODO:
     * specify directory to save file!
     */
    // else if (ii == 2) {
    
    // smth else ...
    // else if (ii == 2) {
    
    printf("\n");
  }
  
  
  // create threads (POSIX)
  pthread_t* threads = (pthread_t*) malloc( (size_t) sizeof(pthread_t)*num_threads );
  pthread_attr_init(&attr); // attr is declared in global variables
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); // threads will be managed by waiting for non-main threads to end
  pthread_barrier_init (&barrier, NULL, num_threads); // last number tells how many threads should be synchronized by this barier
  pthread_barrier_init (&barrier_global, NULL, num_threads+1);
  // this interface enable simple adding new threads
  for (uint8_t ii = 0; ii < num_threads; ii++) {
    // create threads
    printf("creating thread %s\n",thread_names[ii]);
    pthread_create(&threads[ii], &attr, thread_funcs[ii], (void*) global_stuff);
    
    // set affinity (every thread on its own core if possible)
    cpu_set_t cpu_core;
    CPU_ZERO(&cpu_core);
    CPU_SET(ii, &cpu_core);
    pthread_setaffinity_np(threads[ii], sizeof(cpu_set_t), &cpu_core);
    if (CPU_ISSET(ii, &cpu_core)) printf("affinity thread %s set successfully.\n",thread_names[ii]);
  }
  set_signals_same();
  files = open_struct_files(4);
  pthread_barrier_wait (&barrier_global); // global lock for threads
  
  // creating mmap to save wavefunction
  char wf_mmap_filepath[256], str_date[17];
  time_t t = time(NULL);
  strftime(str_date, sizeof(str_date), "%Y-%m-%d_%H:%M", localtime(&t));
  sprintf( wf_mmap_filepath,"./wavefunction_dim%d_N%d_%s.bin", DIM, NX*NY*NZ, str_date );
  printf("%s\n",wf_mmap_filepath);
  global_stuff->wf_save_fd = mmap_create(wf_mmap_filepath, (void**) &wf_mmap, NX*NY*NZ * sizeof(double complex), PROT_READ | PROT_WRITE, MAP_SHARED);
  
  // allocate memory ?
  
  
  
  // run threads
  
  
  // manage threads, use barriers to sync threads
  
  
  // join threads
  pthread_barrier_wait (&barrier_global); //maybe not necessary
  for (uint8_t ii = 0; ii < num_threads; ii++) {
    void* status;
    pthread_join(threads[ii], &status);
  }
  
  // save data
  
  // close files
  //if (backup_file) 	fclose(backup_file);
  //if (wf_file) 		fclose(wf_file);
  
  
  // close files
  mmap_destroy(global_stuff->init_wf_fd, init_wf_mmap, NX*NY*NZ * sizeof(double complex));
  mmap_destroy(global_stuff->wf_save_fd, wf_mmap, NX*NY*NZ * sizeof(double complex));
  //close_files(global_stuff->files, global_stuff->num_files);
  close_struct_files(files, global_stuff->num_files);
  
  
  
  
  
  // clear memory
  free(threads);
  
  
  free(streams);
  free(global_stuff);
  
  
  printf("Main: program completed. Exiting...\n");
  cudaThreadExit();
  cudaDeviceReset();
  
  // think about use atexit(<poiter to function that makes user-defined action just before main ends>)
  
  return EXIT_SUCCESS;
}

