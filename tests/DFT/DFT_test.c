#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <stdint.h>
//#include <cudaruntime.h>

#define M_PI (3.14159265358979323846)
#define M_2PI (6.283185307179586)
#define SQRT_2PI (2.5066282746310002)
#define INV_SQRT_2PI (0.3989422804014327)
#define SIGMA (1)


/*
 * compile: gcc -o DFT.exe DFT_test.c -lfftw3 -lm -O3 -Wall -std=c99
 * 
 */



/*
 * 
 */ 





int main() {
  
  uint64_t N = 0;
  
  for (uint16_t ii=5; ii < 6; ii++){
    N = 1<<ii;
    printf("\nN = %lu",N);
    perform_fftw3_1d(N);
    //perform_cufft_1d(N);
  }
  
  return 0;
}

/*
bool inplace = true; // true for in-place, false for out-of-place
int dim_size[] = {2,2};
int N[] = {2,2};
int data_length     = N[0]*(N[1]);      //  2 * (2)     = 4
int data_fft_length = N[0]*(N[1]/2+1);  //  2 * (2/2+1) = 4
float* h_data_r = nullptr;              //  fftw data array
fftwf_complex* h_data_c = nullptr;      //  fftw data array (only used in out-of-place tranforms)

//  allocate fftw memory
if(inplace) {
    h_data_r = (float*)fftwf_malloc(data_fft_length*sizeof(fftwf_complex));
    h_data_c = (fftwf_complex*)h_data_r;
} else {
    h_data_r = (float*)fftwf_malloc(data_length*sizeof(float));
    h_data_c = (fftwf_complex*)fftwf_malloc(data_fft_length*sizeof(fftwf_complex));
}

//  create plane
unsigned int flags = FFTW_MEASURE;
fftwf_plan m_plan = fftwf_plan_dft_r2c_2d(N[0],N[1],h_data_r,h_data_c,flags);

//  initialize data array
h_data_r[0] = 1;
h_data_r[1] = 1;
h_data_r[2] = 1;
h_data_r[3] = 1;

//  execute fft plan
fftwf_execute(m_plan);

std::cout << "result:" << std::endl;
for(int i = 0; i < data_fft_length; ++i)
    std::cout << "[" << i << "]: " << h_data_c[i][0] << " " << h_data_c[i][1]  << std::endl;
*/