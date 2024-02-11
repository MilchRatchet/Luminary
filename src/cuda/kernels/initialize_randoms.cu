#include "utils.cuh"

__global__ void initialize_randoms() {
  device.ptrs.randoms[THREAD_ID] = 1;
}
