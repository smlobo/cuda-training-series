#include <stdio.h>
#include <math.h>
#include <float.h>
#include <assert.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const int DSIZE = 4096;
const int threads_per_block = 256;  // CUDA maximum is 1024
const int NUM_CHECK = 3;

// vector add kernel: C = A + B
__global__ void vadd(const float *A, const float *B, float *C, int ds){

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < ds)
    C[idx] = A[idx] + B[idx];
}

int main(){

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

  // allocate space for vectors in host memory
  h_A = new float[DSIZE];
  h_B = new float[DSIZE];
  h_C = new float[DSIZE];

  // initialize vectors in host memory
  for (int i = 0; i < DSIZE; i++) {
    h_A[i] = rand()/(float)RAND_MAX;
    h_B[i] = rand()/(float)RAND_MAX;
    h_C[i] = 0;
  }

  // allocate device space for vector A
  cudaMalloc(&d_A, DSIZE*sizeof(float));
  cudaMalloc(&d_B, DSIZE*sizeof(float));
  cudaMalloc(&d_C, DSIZE*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");

  // copy vector A to device:
  cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // cuda processing sequence step 1 is complete

  int blocks_per_grid = (DSIZE+threads_per_block-1)/threads_per_block;
  vadd<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");

  // cuda processing sequence step 2 is complete

  // copy vector C from device to host:
  cudaMemcpy(h_C, d_C, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

  // cuda processing sequence step 3 is complete
  for (int i = 0; i < NUM_CHECK; i++) {
    int randomIndex = rand() % DSIZE;
    printf("[%d] %.3f + %.3f = %.3f\n", randomIndex, h_A[randomIndex], 
      h_B[randomIndex], h_C[randomIndex]);
    assert(fabsf(h_A[randomIndex]+h_B[randomIndex]-h_C[randomIndex]) < FLT_EPSILON);
  }

  return 0;
}
  
