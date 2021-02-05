#include <stdio.h>
#include "cudatest.h"
#include <cuda_runtime.h>

__global__
void addcu(int a, int b, int* c) {
  *c = a + b;
}

extern "C" void cudatest() {
  int c;
  int* dev_c;
  cudaMalloc((void**) &dev_c, sizeof(int));
  addcu<<<1, 1>>>(2, 7, dev_c);

  cudaMemcpy(&c,dev_c,sizeof(int),cudaMemcpyDeviceToHost);

  printf("2 + 7 = %d\n",c);

  cudaFree(dev_c);
}


