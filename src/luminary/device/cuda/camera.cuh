#ifndef CU_CAMERA_H
#define CU_CAMERA_H

#include "camera_lens.cuh"
#include "camera_sensor.cuh"
#include "camera_utils.cuh"
#include "math.cuh"
#include "spectral.cuh"
#include "utils.cuh"

__device__ CameraSimulationResult camera_sample(const ushort2 pixel) {

  float wavelength_pdf;
  const float wavelength = spectral_sample_wavelength(random_1D(RANDOM_TARGET_LENS_WAVELENGTH, pixel), wavelength_pdf);

  vec3 sensor_point = camera_sensor_sample(pixel);

  CameraSimulationResult result = camera_lens_sample(sensor_point, pixel);

  // Convert from spectral to RGB
  result.weight = mul_color(result.weight, spectral_wavelength_to_rgb(wavelength));
  result.weight = scale_color(result.weight, 1.0f / wavelength_pdf);

  // Transform result to world space
  result.origin = quaternion_apply(device.camera.rotation, result.origin);
  result.origin = scale_vector(result.origin, device.camera.camera_scale * CAMERA_COMMON_SCALE);
  result.origin = add_vector(result.origin, device.camera.pos);

  result.ray = quaternion_apply(device.camera.rotation, result.ray);

  return result;
}

#endif /* CU_CAMERA_H */
