#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>


#include <cuda_runtime.h>

//#include "timing_host.h"
#include "fftw_dft.h"
#include "cuda_dft.h"

// defines
#define DIM_1D // to najlepiej by bylo przez Makefile zdefiniowac

/*
 * Opens files for timing different types of dft transforms.
 * 
 * returns array of pointers to files!
 */
FILE** init_fftw_files_for_timing(const uint8_t dim) {
  // buffer for files names
  const uint8_t filename_str_lenght = 128;
  char time_1d_forw_name[filename_str_lenght];
  char time_1d_in_place_name[filename_str_lenght];
  char time_1d_back_name[filename_str_lenght];
  
  sprintf( time_1d_forw_name, "./timing/fftw_forw_%dd.bin", dim );
  printf( "1d fftw3 example save in: %s\n", time_1d_forw_name );
  FILE *time_1d_forw = fopen(time_1d_forw_name, "wb");
  if (time_1d_forw == NULL)
  {
      printf("Error opening file %s!\n",time_1d_forw_name);
      exit(EXIT_FAILURE);
  }
  
  sprintf(time_1d_in_place_name,"./timing/fftw_in_place_%dd.bin",dim );
  printf("1d fftw3 example save in: %s\n",time_1d_in_place_name);
  FILE *time_1d_in_place = fopen(time_1d_in_place_name, "wb");
  if (time_1d_in_place == NULL)
  {
      printf("Error opening file %s!\n",time_1d_in_place_name);
     exit(EXIT_FAILURE);
  }
  
  sprintf(time_1d_back_name,"./timing/fftw_back_%dd.bin",dim );
  printf("1d fftw3 example save in: %s\n",time_1d_back_name);
  FILE *time_1d_back = fopen(time_1d_back_name, "wb");
  if (time_1d_back == NULL)
  {
      printf("Error opening file %s!\n",time_1d_back_name);
      exit(EXIT_FAILURE);
  }
  
  // allocating array of pointer to files
  FILE** array_files = (FILE**) malloc( (size_t) sizeof( FILE* )*3 );
  array_files[0] = time_1d_forw;
  array_files[1] = time_1d_in_place;
  array_files[2] = time_1d_back;
  
  return array_files;
}

FILE** init_cufft_files_for_timing(const uint8_t dim) {
  // buffer for files names
  const uint8_t filename_str_lenght = 128;
  char time_1d_forw_name[filename_str_lenght];
  char time_1d_in_place_name[filename_str_lenght];
  char time_1d_back_name[filename_str_lenght];
  
  sprintf( time_1d_forw_name, "./timing/cufft_forw_%dd.bin", dim );
  printf( "1d cufft example save in: %s\n", time_1d_forw_name );
  FILE *time_1d_forw = fopen(time_1d_forw_name, "wb");
  if (time_1d_forw == NULL)
  {
      printf("Error opening file %s!\n",time_1d_forw_name);
      exit(EXIT_FAILURE);
  }
  
  sprintf(time_1d_in_place_name,"./timing/cufft_in_place_%dd.bin",dim );
  printf("1d cufft example save in: %s\n",time_1d_in_place_name);
  FILE *time_1d_in_place = fopen(time_1d_in_place_name, "wb");
  if (time_1d_in_place == NULL)
  {
      printf("Error opening file %s!\n",time_1d_in_place_name);
     exit(EXIT_FAILURE);
  }
  
  sprintf(time_1d_back_name,"./timing/cufft_back_%dd.bin",dim );
  printf("1d cufft example save in: %s\n",time_1d_back_name);
  FILE *time_1d_back = fopen(time_1d_back_name, "wb");
  if (time_1d_back == NULL)
  {
      printf("Error opening file %s!\n",time_1d_back_name);
      exit(EXIT_FAILURE);
  }
  
  // allocating array of pointer to files
  FILE** array_files = (FILE**) malloc( (size_t) sizeof( FILE* )*3 );
  array_files[0] = time_1d_forw;
  array_files[1] = time_1d_in_place;
  array_files[2] = time_1d_back;
  
  return array_files;
}

/*
 * closes files in array
 */
void close_files_for_timing(FILE** array_files){
  fclose(array_files[0]);
  fclose(array_files[1]);
  fclose(array_files[2]);
}

int main(){
  
  // files to write timings
  FILE** file_timimng = NULL;
  FILE** file_timimng_cu = NULL;
#ifdef DIM_1D
  file_timimng =  init_fftw_files_for_timing(1);
  file_timimng_cu =  init_cufft_files_for_timing(1);
#endif
  if (!file_timimng) {
    printf("Error of allocating arrays of pointers to files for saving timing!"); // just to debug
    exit(EXIT_FAILURE);
  }
  
  cudaDeviceReset();
  uint64_t N = 0; // number of samples in dft
  
  for (uint64_t ii=5; ii <= 16; ii++) {
    N = 1<<ii;
#ifdef DIM_1D
    printf("\nN = %lu\n",N);
    //if (N < 65536)
    perform_fftw3_1d(N, file_timimng);
    perform_cufft_1d(N, file_timimng_cu);
#endif
  }
  
  close_files_for_timing(file_timimng);
  close_files_for_timing(file_timimng_cu);
  
  return 0;
}