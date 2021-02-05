#include <stdio.h>
#include "cudatest.h"
#include <cuda_runtime_api.h>
#include <stdint.h>

const int threads_per_block = 256;
const int blocks_per_grid = 32;

__global__
void fillPixel(uint8_t* frame, unsigned int width, unsigned int height) {
  unsigned int ptr = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int amount = width * height;

  unsigned int effective_ptr = ptr * 3;

  while (ptr < amount) {
    int i = ptr % width;
    int j = (ptr - i) / width;
    frame[effective_ptr] = (i+j)%255;
    frame[effective_ptr+1] = i%255;
    frame[effective_ptr+2] = j%255;
    ptr += blockDim.x * gridDim.x;
    effective_ptr = ptr * 3;
  }
}

extern "C" uint8_t* test_frame(const unsigned int width, const unsigned int height) {
  uint8_t* frame_cpu = (uint8_t*)malloc(3 * width * height);
  uint8_t* frame_gpu;
  unsigned int width_gpu, height_gpu;

  cudaMalloc((void**) &frame_gpu, 3 * width * height);
  cudaMalloc((void**) &width_gpu, sizeof(int));
  cudaMalloc((void**) &height_gpu, sizeof(int));

  cudaMemcpy(&width_gpu,&width,sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(&height_gpu,&height,sizeof(int),cudaMemcpyHostToDevice);

  fillPixel<<<blocks_per_grid,threads_per_block>>>(frame_gpu,width,height);

  cudaMemcpy(frame_cpu, frame_gpu, 3 * width * height, cudaMemcpyDeviceToHost);

  cudaFree(frame_gpu);
  cudaFree(&width_gpu);
  cudaFree(&height_gpu);

  return frame_cpu;
}


