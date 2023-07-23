#ifndef CU_STATE_H
#define CU_STATE_H

#include "utils.cuh"

enum StateFlag { STATE_FLAG_ALBEDO = 0b00000001u, STATE_FLAG_LIGHT_OCCUPIED = 0b00000010u } typedef StateFlag;

//
// Usage documentation:
//
// STATE_FLAG_ALBEDO: This flag is set for each pixel once a valid albedo value is found that can be written to the albedo buffer.
//                    The flag gets reset at the start of every frame.
//
// STATE_FLAG_LIGHT_OCCUPIED: This flag is set whenever a light ray is queued for the current pixel. This flag gets reset
//                            once the light ray has been processed and before the shading. A light ray may set this flag
//                            again for itself if it requires another iteration due to transparency.
//

__device__ bool state_consume(const int pixel, const StateFlag flag) {
  if (device.ptrs.state_buffer[pixel] & flag) {
    return false;
  }
  else {
    device.ptrs.state_buffer[pixel] |= flag;
    return true;
  }
}

__device__ void state_release(const int pixel, const StateFlag flag) {
  device.ptrs.state_buffer[pixel] &= ~flag;
}

#endif /* CU_STATE_H */
