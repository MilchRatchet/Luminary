#ifndef CU_STATE_H
#define CU_STATE_H

#include "utils.cuh"

enum StateFlag { STATE_FLAG_ALBEDO = 0b00000001u } typedef StateFlag;

//
// Usage documentation:
//
// STATE_FLAG_ALBEDO: This flag is set for each pixel once a valid albedo value is found that can be written to the albedo buffer.
//                    The flag gets reset at the start of every frame.
//
//

__device__ bool state_consume(const int pixel, const StateFlag flag) {
  if (device.ptrs.state_buffer[pixel] & flag) {
    return false;
  }

  device.ptrs.state_buffer[pixel] |= flag;
  return true;
}

__device__ bool state_peek(const int pixel, const StateFlag flag) {
  return (device.ptrs.state_buffer[pixel] & flag) ? true : false;
}

__device__ void state_release(const int pixel, const StateFlag flag) {
  device.ptrs.state_buffer[pixel] &= ~flag;
}

#endif /* CU_STATE_H */
