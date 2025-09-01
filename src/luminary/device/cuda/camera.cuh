#ifndef CU_CAMERA_H
#define CU_CAMERA_H

#include "camera_lens.cuh"
#include "camera_sensor.cuh"
#include "camera_utils.cuh"
#include "math.cuh"
#include "spectral.cuh"
#include "utils.cuh"

template <bool ALLOW_REFLECTIONS, bool SPECTRAL_RENDERING>
__device__ CameraSimulationResult camera_sample_impl(const ushort2 pixel) {

  float wavelength_pdf;
  const float wavelength = spectral_sample_wavelength(random_1D(RANDOM_TARGET_LENS_WAVELENGTH, pixel), wavelength_pdf);

  vec3 sensor_point = camera_sensor_sample(pixel);

  CameraSimulationResult result = camera_lens_sample<ALLOW_REFLECTIONS, SPECTRAL_RENDERING>(sensor_point, wavelength, pixel);

  // Convert from spectral to RGB
  if constexpr (SPECTRAL_RENDERING) {
    result.weight = mul_color(result.weight, spectral_wavelength_to_rgb(wavelength));
    result.weight = scale_color(result.weight, 1.0f / wavelength_pdf);
  }

  // Transform result to world space
  result.origin = quaternion_apply(device.camera.rotation, result.origin);
  result.origin = scale_vector(result.origin, device.camera.camera_scale * CAMERA_COMMON_SCALE);
  result.origin = add_vector(result.origin, device.camera.pos);

  result.ray = quaternion_apply(device.camera.rotation, result.ray);

  return result;
}

__device__ CameraSimulationResult camera_sample(const ushort2 pixel) {
  const bool allow_reflections  = device.camera.allow_reflections;
  const bool spectral_rendering = device.camera.use_spectral_rendering;

  if (allow_reflections == true && spectral_rendering == true)
    return camera_sample_impl<true, true>(pixel);

  if (allow_reflections == true && spectral_rendering == false)
    return camera_sample_impl<true, false>(pixel);

  if (allow_reflections == false && spectral_rendering == true)
    return camera_sample_impl<false, true>(pixel);

  if (allow_reflections == false && spectral_rendering == false)
    return camera_sample_impl<false, false>(pixel);
}

#endif /* CU_CAMERA_H */
