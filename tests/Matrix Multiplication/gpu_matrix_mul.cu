#include <stdio.h>
#include <cuda.h>

#include "gpu_matrix_mul.cuh"
#include "cudautils.cuh"


/* ********************************************** KERNELS ****************************************************************************** */

__global__ void cudaMultMatrix(Matrix A, Matrix X, Matrix Y)
{
	  int bx = blockIdx.x; 
          //int by = blockIdx.y;
	  int tx = threadIdx.x; 
          //int ty = threadIdx.y;
  // Calculate the row index of the Pd element and M
  int Row = bx * BLOCK_SIZE + tx;
  // Calculate the column idenx of Pd and N
  //int Col = bx * BLOCK_SIZE + tx;
  
  float Pvalue = 0;

   
  for (unsigned int k = 0; k < A.width; k++) 
    {
      if(Row < A.height)         
      Pvalue += A.elements[Row*A.width+k] * X.elements[k];
      //else
      //Pvalue += 0;

    }

  __syncthreads();
  
  if(Row < A.height)  		
    Y.elements[Row] = Pvalue;
  __syncthreads();
}


void