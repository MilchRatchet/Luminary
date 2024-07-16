#ifndef CU_STATE_H
#define CU_STATE_H

#include "utils.cuh"

enum StateFlag {
  STATE_FLAG_ALBEDO           = 0b00000001u,
  STATE_FLAG_DELTA_PATH       = 0b00000010u,
  STATE_FLAG_CAMERA_DIRECTION = 0b00000100u
} typedef StateFlag;

//
// Usage documentation:
//
// STATE_FLAG_ALBEDO: This flag is set for each pixel once a valid albedo value is found that can be written to the albedo buffer.
//                    The flag gets reset at the start of every frame.
//
// STATE_FLAG_DELTA_PATH: This flag is set for paths whose vertices generated bounce rays only from delta (or near-delta) distributions.
//                        This flag is used for firefly clamping as it only applies to light gathered on path suffixes of non-delta paths.
//
// STATE_FLAG_CAMERA_DIRECTION: This flag is set while the current path is just a line along the original camera direction.
//                              This flag is used to allow light to be gathered through non-refractive transparencies when coming directly
//                              from the camera where no DL is executed.
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

__device__ void state_release(const int pixel, const uint32_t flag) {
  if (!flag)
    return;

  device.ptrs.state_buffer[pixel] &= ~flag;
}

#endif /* CU_STATE_H */
