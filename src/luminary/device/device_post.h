#ifndef LUMINARY_DEVICE_POST_H
#define LUMINARY_DEVICE_POST_H

#include "device_utils.h"

struct Device typedef Device;

struct DevicePost {
  uint32_t width;
  uint32_t height;

  bool glow_active;
  LuminaryGlowMode glow_mode;
  float glow_strength;
  LuminaryRGBF glow_color;
  float* DEVICE* glow_mips;
  uint32_t glow_mip_count;
} typedef DevicePost;

DEVICE_CTX_FUNC LuminaryResult device_post_create(DevicePost** post);
DEVICE_CTX_FUNC LuminaryResult device_post_allocate(DevicePost* post, uint32_t width, uint32_t height);
DEVICE_CTX_FUNC LuminaryResult device_post_update(DevicePost* post, const Camera* camera);
DEVICE_CTX_FUNC LuminaryResult device_post_apply(DevicePost* post, Device* device);
DEVICE_CTX_FUNC LuminaryResult device_post_destroy(DevicePost** post);

#endif /* LUMINARY_DEVICE_POST_H */
