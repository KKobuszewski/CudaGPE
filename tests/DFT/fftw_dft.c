#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <stdint.h>

#include "fftw_dft.h"

extern inline void print_timeit(clock_t time_to_print, const char* message);
extern inline void fwrite_timeit(clock_t time, FILE* file);

complex double gauss_1d(double x) {
  
  return INV_SQRT_2PI*exp(-x*x/2/SIGMA)/SIGMA + I*0.;//GAUSSIAN
}


void perform_fftw3_1d(const uint64_t N, FILE** array_timing) {
  
  // for timing purposes
  clock_t start_t=0;
  
  // initilizing files to save data
  const uint8_t filename_str_lenght = 128;
  const uint8_t dim = 1;
  
  char filename1d[filename_str_lenght];
  sprintf(filename1d,"./data/fftw_%dd_N%lu.bin",dim,N );
  printf("1d fftw3 example save in: %s\n",filename1d);
  FILE *file1d = fopen(filename1d, "wb");
  if (file1d == NULL)
  {
      printf("Error opening file %s!\n",filename1d);
      exit(EXIT_FAILURE);
  }
  
  
  // initializing 1d arrays of data
  fftw_complex* data_in_1d, *data_forward_1d, *data_backward_1d, *for_plan_1d;
  data_in_1d = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
  data_forward_1d = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
  data_backward_1d = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
  for_plan_1d = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
  
  const double x0 = -5*SIGMA;
  const double dx = (10*SIGMA)/((double) N);
  double x =0.;
  for (uint64_t ii=0; ii<N; ii++) {
    x = x0 + ii*dx;
    //printf("%lf\t",x);
    data_in_1d[ii] = gauss_1d(x);
    //printf("%lf + %lfj\n",creal(data_in_1d[ii]),cimag(data_in_1d[ii]));
  }
  
  
  // make plans <- check http://www.fftw.org/doc/Using-Plans.html for extensions!
  fftw_plan plan_forward, plan_backward, plan_in_place;
  
  TIMEIT_START;
  plan_forward = fftw_plan_dft_1d(N, data_in_1d, data_forward_1d, FFTW_FORWARD, FFTW_ESTIMATE); // jak robic FFTW_EXAHAUSTED? <- http://www.fftw.org/doc/Planner-Flags.html
  TIMEIT_END(clock_t plan_forward_t);
  fwrite_timeit(plan_forward_t,array_timing[0]);
  //fwrite(&plan_forward_t, (size_t) sizeof(double), 1, array_timing[0]);
  
  TIMEIT_START;
  plan_in_place = fftw_plan_dft_1d(N, data_in_1d, data_in_1d, FFTW_FORWARD, FFTW_ESTIMATE);
  TIMEIT_END(clock_t plan_in_place_t);
  fwrite_timeit(plan_in_place_t,array_timing[1]);
    
  TIMEIT_START;
  plan_backward = fftw_plan_dft_1d(N, data_forward_1d, data_backward_1d, FFTW_BACKWARD, FFTW_ESTIMATE);
  TIMEIT_END(clock_t plan_backward_t);
  fwrite_timeit(plan_backward_t,array_timing[2]);
  
  
  // make dft forward with two arrays
  TIMEIT_START;
  fftw_execute(plan_forward);
  TIMEIT_END(clock_t execute_forward_t);
  fwrite_timeit(execute_forward_t,array_timing[0]);
  
  for (uint16_t ii = 0; ii < N; ii++) {
    fwrite(data_forward_1d+ii, sizeof(fftw_complex),1,file1d);
  }
  
  //make dft forward in place
  TIMEIT_START;
  fftw_execute(plan_in_place);
  TIMEIT_END(clock_t execute_in_place_t);
  fwrite_timeit(execute_in_place_t,array_timing[1]);
  
  for (uint16_t ii = 0; ii < N; ii++) {
    fwrite(data_in_1d+ii, sizeof(fftw_complex),1,file1d);
  }
  
  //make dft data_backward_1d
  TIMEIT_START;
  fftw_execute(plan_backward);
  TIMEIT_END(clock_t execute_backward_t);
  fwrite_timeit(execute_backward_t,array_timing[2]);
  for (uint16_t ii = 0; ii < N; ii++) {
    fwrite(data_backward_1d+ii, sizeof(fftw_complex),1,file1d);
  }
  
  
  // cleaning up the mesh
  fftw_free(data_in_1d);
  fftw_free(data_forward_1d);
  fftw_free(data_backward_1d);
  fftw_free(for_plan_1d);
  
  fftw_cleanup(); // deallocates the planes
  fclose(file1d);
}