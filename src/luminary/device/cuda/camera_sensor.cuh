#ifndef CU_LUMINARY_CAMERA_SENSOR_H
#define CU_LUMINARY_CAMERA_SENSOR_H

#include "camera_utils.cuh"
#include "utils.cuh"

__device__ vec3 camera_sensor_sample(const ushort2 pixel) {
  const float2 jitter = camera_get_jitter();

  const float step = 2.0f * (device.camera.fov / device.settings.width);
  const float vfov = step * device.settings.height * 0.5f;

#ifndef CAMERA_DEBUG_RENDER
  const ushort2 sensor_pixel = pixel;
#else
  const ushort2 sensor_pixel = make_ushort2(device.settings.width >> 1, device.settings.height >> 1);
#endif

  vec3 sensor_point;
  sensor_point.x = device.camera.fov - step * (sensor_pixel.x + jitter.x);
  sensor_point.y = -vfov + step * (sensor_pixel.y + jitter.y);
  sensor_point.z = camera_thin_lens_image_plane();

  return sensor_point;
}

#endif /* CU_LUMINARY_CAMERA_SENSOR_H */
