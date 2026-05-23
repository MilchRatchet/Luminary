#ifndef CU_LUMINARY_CAMERA_UTILS_H
#define CU_LUMINARY_CAMERA_UTILS_H

#include "random.cuh"
#include "utils.cuh"

#define CAMERA_FRAUNHOFER_D_LINE (587.562f)
#define CAMERA_FRAUNHOFER_F_LINE (486.134f)
#define CAMERA_FRAUNHOFER_C_LINE (656.281f)

#define CAMERA_DESIGN_WAVELENGTH CAMERA_FRAUNHOFER_D_LINE

struct CameraSampleResult {
  vec3 origin;
  vec3 ray;
  RGBF weight;
} typedef CameraSampleResult;

LUMINARY_FUNCTION float2 camera_get_jitter(const PathID& path_id) {
  const uint32_t sample_id = path_id_get_sample_id(path_id);

  return random_2D_base_float(RANDOM_TARGET_CAMERA_JITTER, make_ushort2(0, 0), sample_id, 0);
}

template <bool IS_THIN_LENS>
LUMINARY_FUNCTION vec3 camera_sample_sensor(const PathID& path_id) {
  const float2 jitter = camera_get_jitter(path_id);

  const float aspect_ratio = (device.camera.use_aspect_ratio_from_resolution) ? ((float) device.settings.width / device.settings.height)
                                                                              : device.camera.sensor.aspect_ratio;

  float sensor_height;
  float sensor_distance;

  if constexpr (IS_THIN_LENS) {
    sensor_distance = 1.0f;
    sensor_height   = 2.0f * tanf(device.camera.thin_lens_fov * 0.5f) * sensor_distance;
  }
  else {
    sensor_distance = device.camera_aux.sensor_distance;
    sensor_height   = device.camera.sensor_diagonal_size / sqrtf(aspect_ratio * aspect_ratio + 1.0f);
  }

  const float sensor_width = aspect_ratio * sensor_height;

  const float step_x = sensor_width / device.settings.width;
  const float step_y = sensor_height / device.settings.height;

  const ushort2 sensor_pixel = path_id_get_pixel(path_id);

  vec3 sensor_point;
  sensor_point.x = 0.5f * sensor_width - step_x * (sensor_pixel.x + jitter.x);
  sensor_point.y = 0.5f * sensor_height - step_y * (sensor_pixel.y + jitter.y);
  sensor_point.z = -sensor_distance;

  // Flip vertically
  sensor_point.y *= -1.0f;

  return sensor_point;
}

////////////////////////////////////////////////////////////////////
// Dispersion
////////////////////////////////////////////////////////////////////

LUMINARY_FUNCTION float camera_ior_cauchy_approximation(const float nd, const float abbe, const float wavelength) {
  const float range_factor =
    (1.0f / (CAMERA_FRAUNHOFER_F_LINE * CAMERA_FRAUNHOFER_F_LINE)) - (1.0f / (CAMERA_FRAUNHOFER_C_LINE * CAMERA_FRAUNHOFER_C_LINE));

  const float b = (nd - 1.0f) / (abbe * range_factor);

  const float a = nd - b * (1.0f / (CAMERA_FRAUNHOFER_D_LINE * CAMERA_FRAUNHOFER_D_LINE));

  return a + b / (wavelength * wavelength);
}

template <bool SPECTRAL_RENDERING>
LUMINARY_FUNCTION float camera_medium_get_ior(const DeviceCameraMedium medium, const float wavelength) {
  if constexpr (SPECTRAL_RENDERING)
    return (medium.abbe != 0.0f) ? camera_ior_cauchy_approximation(medium.design_ior, medium.abbe, wavelength) : medium.design_ior;

  return medium.design_ior;
}

#endif /* CU_LUMINARY_CAMERA_UTILS_H */
