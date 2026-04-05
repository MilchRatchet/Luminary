#ifndef CU_LUMINARY_CAMERA_THIN_LENS_H
#define CU_LUMINARY_CAMERA_THIN_LENS_H

#include "camera_utils.cuh"
#include "math.cuh"
#include "utils.cuh"

// We force the weight to be 1, else the brightness of the image would depend on aperture size.
// That would be realistic but not practical.
LUMINARY_FUNCTION vec3 camera_thin_lens_sample_aperture(const PathID& path_id) {
  if (device.camera.lens.aperture_radius == 0.0f)
    return get_vector(0.0f, 0.0f, 0.0f);

  const float2 random = random_2D(RANDOM_TARGET_LENS, path_id);

  float2 sample;

  const float aperture_size = device.camera.lens.aperture_radius * CAMERA_COMMON_INV_SCALE;

  switch (device.camera.aperture_shape) {
    default:
    case LUMINARY_APERTURE_ROUND: {
      const float alpha = random.x * 2.0f * PI;
      const float beta  = sqrtf(random.y) * aperture_size;

      sample = make_float2(cosf(alpha) * beta, sinf(alpha) * beta);
    } break;
    case LUMINARY_APERTURE_BLADED: {
      const int blade   = random_1D(RANDOM_TARGET_LENS_BLADE, path_id) * device.camera.aperture_blade_count;
      const float alpha = sqrtf(random.x);
      const float beta  = random.y;

      const float u = 1.0f - alpha;
      const float v = alpha * beta;

      const float angle_step = (2.0f * PI) / device.camera.aperture_blade_count;

      const float angle1 = angle_step * blade;
      const float angle2 = angle_step * (blade + 1);

      sample.x = sinf(angle1) * u + sinf(angle2) * v;
      sample.y = cosf(angle1) * u + cosf(angle2) * v;

      sample.x *= aperture_size;
      sample.y *= aperture_size;
    } break;
  }

  return get_vector(sample.x, sample.y, 0.0f);
}

LUMINARY_FUNCTION CameraSampleResult camera_thin_lens_sample(const PathID& path_id) {
  const vec3 sensor_point = camera_sample_sensor(path_id);

  const vec3 sensor_to_focal_ray = normalize_vector(sub_vector(get_vector(0.0f, 0.0f, 0.0f), sensor_point));

  // The minus is because we are always looking in Z direction
  const vec3 focal_point = scale_vector(sensor_to_focal_ray, device.camera.lens.focal_length / sensor_to_focal_ray.z);

  const vec3 aperture_point = camera_thin_lens_sample_aperture(path_id);

  CameraSampleResult result;
  result.origin = aperture_point;
  result.ray    = normalize_vector(sub_vector(focal_point, aperture_point));
  result.weight = splat_color(1.0f);

  // Camera simulation is in +Z direction but Luminary uses -Z convention
  result.origin.z = -result.origin.z;
  result.ray.z    = -result.ray.z;

  return result;
}

#endif /* CU_LUMINARY_CAMERA_THIN_LENS_H */
