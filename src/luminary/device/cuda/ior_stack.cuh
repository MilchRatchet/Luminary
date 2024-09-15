#ifndef CU_IOR_STACK_H
#define CU_IOR_STACK_H

#include "utils.cuh"

enum IORStackMethod {
  /* Reset the stack and place an entry. */
  IOR_STACK_METHOD_RESET = 0,
  /* Return the top entry of the stack without modifying the stack. */
  IOR_STACK_METHOD_PEEK_CURRENT = 1,
  /* Return the second entry of the stack without modifying the stack. */
  IOR_STACK_METHOD_PEEK_PREVIOUS = 2,
  /* Add a new entry to the top of the stack. */
  IOR_STACK_METHOD_PUSH = 3,
  /* Remove the top entry of the stack and return it. */
  IOR_STACK_METHOD_PULL = 4
} typedef IORStackMethod;

// BSDF transparent fast path relies on this precision.
// If the precision is improved in the future we will run into INFs with surfaces with refractive index close to 1.
__device__ uint32_t ior_compress(const float ior) {
  return (__float_as_uint((0.5f * (ior - 1.0f)) + 1.0f) >> 15) & 0xFF;
}

__device__ float ior_decompress(const uint32_t compressed_ior) {
  return ((__uint_as_float(0x3F800000u | (compressed_ior << 15)) - 1.0f) * 2.0f) + 1.0f;
}

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

    const uint32_t compressed_ior = ior_compress(ior);
    current_stack |= compressed_ior;
  }
  else {
    const uint32_t compressed_ior = (method == IOR_STACK_METHOD_PEEK_PREVIOUS) ? (current_stack >> 8) & 0xFF : current_stack & 0xFF;

    if (method == IOR_STACK_METHOD_PULL)
      current_stack = current_stack >> 8;

    out_ior = ior_decompress(compressed_ior);
  }

  device.ptrs.ior_stack[pixel] = current_stack;

  return out_ior;
}

#endif /* CU_IOR_STACK_H */
