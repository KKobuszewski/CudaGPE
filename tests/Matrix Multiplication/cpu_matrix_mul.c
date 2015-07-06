#include <stdio.h>
#include <stdint.h>

#include "cpu_matrix_mul.h"

//#define INDEX_1D(i_x,N) (i_x)
#define INDEX_2D(i_x,i_y,N) (i_x*N + i_y) // returns index of matrix a(i,j) of size NxN in 1d array of size N*N, normal matrix indexing convention

void mult_matrix(double* A_matrix, double* B_matrix, double* result, const uint64_t N){
  for (int ii = 0; ii < N; ii++) {
    for (int jj = 0; jj < N; jj++) {
      for (int kk = 0; kk < N; kk++)
	result[INDEX_2D(ii,jj,N)] += A_matrix[INDEX_2D(ii,kk,N)]*B_matrix[INDEX_2D(kk,jj,N)];
    }
  } 
}

void fill_matrix(double* matrix, const uint64_t N) {
  for (int ii = 0; ii < N; ii++) {
    for (int jj = 0; jj < N; jj++) {
      matrix[INDEX_2D(ii,jj,N)] = 1;//INDEX_2D(ii,jj,N);
    }
  }
}

void square_vec(double* matrix, const uint64_t N) {
  for (int ii=0; ii < N; ii++) {
    matrix[ii] *= matrix[ii];
  }
}

void square_matrix(double* matrix, const uint64_t N) {
  for (int ii=0; ii < N*N; ii++) {
    matrix[ii] *= matrix[ii];
  }
}

void print_matrix(double* matrix, const uint64_t N) {
  for (int ii = 0; ii < N; ii++) {
    for (int jj = 0; jj < N; jj++) {
      printf("%*lf",15,matrix[INDEX_2D(ii,jj,N)]);
    }
    printf("\n");
  }
}