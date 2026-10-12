#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  // Blocks are laid out like in the later kernels: blockIdx.x walks the columns
  // of C, blockIdx.y the rows. Inside a block threadIdx.x deliberately stays on
  // the rows: a warp is 32 consecutive threadIdx.x, so its threads sit in 32
  // different rows and the loads from A can't be coalesced. Moving threadIdx.x
  // onto the columns would silently turn this kernel into kernel 2.
  const uint cRow = blockIdx.y * blockDim.x + threadIdx.x;
  const uint cCol = blockIdx.x * blockDim.y + threadIdx.y;

  // if statement is necessary to make things work under tile quantization
  if (cRow < M && cCol < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    // C = α*(A@B)+β*C
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}