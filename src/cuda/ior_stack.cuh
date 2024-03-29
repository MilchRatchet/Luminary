#ifndef CU_IOR_STACK_H
#define CU_IOR_STACK_H

#include "utils.cuh"

enum IORStackMethod {
  IOR_STACK_METHOD_RESET         = 0,
  IOR_STACK_METHOD_PEEK_CURRENT  = 1,
  IOR_STACK_METHOD_PEEK_PREVIOUS = 2,
  IOR_STACK_METHOD_PUSH          = 3,
  IOR_STACK_METHOD_PULL          = 4
} typedef IORStackMethod;

__device__ float ior_stack_interact(const float ior, const uint32_t pixel, const IORStackMethod method) {
  uint32_t current_stack;
  if (method == IOR_STACK_METHOD_RESET) {
    current_stack = 0;
  }
  else {
    current_stack = device.ptrs.ior_stack[pixel];
  }

  float out_ior = ior;
  if (method == IOR_STACK_METHOD_PUSH || method == IOR_STACK_METHOD_RESET) {
    current_stack = current_stack << 8;

    const uint32_t compressed_ior = (__float_as_uint((0.5f * (ior - 1.0f)) + 1.0f) >> 15) & 0xFF;
    current_stack |= compressed_ior;
  }
  else {
    const uint32_t compressed_ior = (method == IOR_STACK_METHOD_PEEK_CURRENT) ? current_stack & 0xFF : (current_stack >> 8) & 0xFF;

    if (method == IOR_STACK_METHOD_PULL)
      current_stack = current_stack >> 8;

    out_ior = ((__uint_as_float(0x3F800000u | (compressed_ior << 15)) - 1.0f) * 2.0f) + 1.0f;
  }

  device.ptrs.ior_stack[pixel] = current_stack;

  return out_ior;
}

#endif /* CU_IOR_STACK_H */
