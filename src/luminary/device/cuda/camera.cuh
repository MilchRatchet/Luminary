#ifndef CU_CAMERA_H
#define CU_CAMERA_H

#include "camera_physical.cuh"
#include "camera_thin_lens.cuh"
#include "camera_utils.cuh"
#include "math.cuh"
#include "spectral.cuh"
#include "utils.cuh"

__device__ CameraSampleResult camera_sample(const ushort2 pixel) {
  CameraSampleResult result;
  if (device.camera.use_physical_camera) {
    const bool allow_reflections  = device.camera.allow_reflections;
    const bool spectral_rendering = device.camera.use_spectral_rendering;

    if (allow_reflections == true && spectral_rendering == true)
      result = camera_physical_sample<true, true>(pixel);
    else if (allow_reflections == true && spectral_rendering == false)
      result = camera_physical_sample<true, false>(pixel);
    else if (allow_reflections == false && spectral_rendering == true)
      result = camera_physical_sample<false, true>(pixel);
    else
      result = camera_physical_sample<false, false>(pixel);
  }
  else {
    result = camera_thin_lens_sample(pixel);
  }

  // Transform result to world space
  result.origin = quaternion_apply(device.camera.rotation, result.origin);
  result.origin = scale_vector(result.origin, device.camera.camera_scale * CAMERA_COMMON_SCALE);
  result.origin = add_vector(result.origin, device.camera.pos);

  result.ray = quaternion_apply(device.camera.rotation, result.ray);

  return result;
}

#endif /* CU_CAMERA_H */
