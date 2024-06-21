#ifndef CU_STATE_H
#define CU_STATE_H

#include "utils.cuh"

enum StateFlag {
  STATE_FLAG_ALBEDO          = 0b00000001u,
  STATE_FLAG_BOUNCE_LIGHTING = 0b00000010u,
  STATE_FLAG_DELTA_PATH      = 0b00000100u
} typedef StateFlag;

//
// Usage documentation:
//
// STATE_FLAG_ALBEDO: This flag is set for each pixel once a valid albedo value is found that can be written to the albedo buffer.
//                    The flag gets reset at the start of every frame.
//
// STATE_FLAG_BOUNCE_LIGHTING: This flag is set for rays that are eligible to gather emission from light sources.
//                             If a ray is generated from a source that does not handle direct lighting, this flag is set.
//                             The flag gets reset when a new ray is spawned.
//
// STATE_FLAG_DELTA_PATH: This flag is set for paths whose path vertices' BSDFs are all delta distributions.
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
