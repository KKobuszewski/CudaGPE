#ifndef __CPU_MATRIX_MUL_H__
#define __CPU_MATRIX_MUL_H__

void mult_matrix(double* A_matrix, double* B_matrix, double* result, const uint64_t N);
void print_matrix(double* matrix, const uint64_t N);
void fill_matrix(double* matrix, const uint64_t N);

#endif