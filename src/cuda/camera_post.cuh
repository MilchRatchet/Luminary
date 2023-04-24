#ifndef CU_CAMERA_POST_H
#define CU_CAMERA_POST_H

#include "camera_post_bloom.cuh"
#include "camera_post_lens_flare.cuh"
#include "utils.h"

extern "C" void device_camera_post_init(RaytraceInstance* instance) {
  if (instance->scene.camera.lens_flare) {
    device_lens_flare_init(instance);
  }

  if (instance->scene.camera.bloom) {
    device_bloom_init(instance);
  }
}

extern "C" void device_camera_post_apply(RaytraceInstance* instance, const RGBAhalf* src, RGBAhalf* dst) {
  if (instance->scene.camera.lens_flare) {
    device_lens_flare_apply(instance, src, dst);
  }

  if (instance->scene.camera.bloom) {
    device_bloom_apply(instance, src, dst);
  }
}

extern "C" void device_camera_post_clear(RaytraceInstance* instance) {
  if (instance->scene.camera.lens_flare) {
    device_lens_flare_clear(instance);
  }

  if (instance->scene.camera.bloom) {
    device_bloom_clear(instance);
  }
}

#endif /* CU_CAMERA_POST_H */
