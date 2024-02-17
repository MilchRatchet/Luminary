#ifndef CU_CAMERA_H
#define CU_CAMERA_H

#include "math.cuh"
#include "utils.cuh"

__device__ vec3 camera_sample_aperture(const ushort2 pixel_coords) {
  if (device.scene.camera.aperture_size == 0.0f)
    return get_vector(0.0f, 0.0f, 0.0f);

  const float2 random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_LENS, pixel_coords);
  float2 sample;

  switch (device.scene.camera.aperture_shape) {
    default:
    case CAMERA_APERTURE_ROUND: {
      const float alpha = random.x * 2.0f * PI;
      const float beta  = sqrtf(random.y) * device.scene.camera.aperture_size;

      sample = make_float2(cosf(alpha) * beta, sinf(alpha) * beta);
    } break;
    case CAMERA_APERTURE_BLADED: {
      const int blade   = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_LENS_BLADE, pixel_coords) * device.scene.camera.aperture_blade_count;
      const float alpha = sqrtf(random.x);
      const float beta  = random.y;

      const float u = 1.0f - alpha;
      const float v = alpha * beta;

      const float angle_step = (2.0f * PI) / device.scene.camera.aperture_blade_count;

      const float angle1 = angle_step * blade;
      const float angle2 = angle_step * (blade + 1);

      sample.x = sinf(angle1) * u + sinf(angle2) * v;
      sample.y = cosf(angle1) * u + cosf(angle2) * v;

      sample.x *= device.scene.camera.aperture_size;
      sample.y *= device.scene.camera.aperture_size;
    } break;
  }

  return get_vector(sample.x, sample.y, 0.0f);
}

__device__ float2 camera_jitter(const float2 rand) {
  return make_float2(0.5f + tent_filter_importance_sample(rand.x), 0.5f + tent_filter_importance_sample(rand.y));
}

__device__ TraceTask camera_get_ray(TraceTask task, const uint32_t pixel) {
  const float2 jitter = (device.accum_mode != TEMPORAL_REPROJECTION)
                          ? camera_jitter(quasirandom_sequence_2D_global(QUASI_RANDOM_TARGET_CAMERA_JITTER))
                          : make_float2(device.emitter.jitter.x, device.emitter.jitter.y);

  vec3 film_point;
  film_point.x = device.scene.camera.fov - device.emitter.step * (task.index.x + jitter.x);
  film_point.y = -device.emitter.vfov + device.emitter.step * (task.index.y + jitter.y);
  film_point.z = 1.0f;

  vec3 film_to_focal_ray = normalize_vector(sub_vector(get_vector(0.0f, 0.0f, 0.0f), film_point));

  // The minus is because we are always looking in -Z direction
  vec3 focal_point = scale_vector(film_to_focal_ray, -device.scene.camera.focal_length / film_to_focal_ray.z);

  vec3 aperture_point = camera_sample_aperture(task.index);

  focal_point    = rotate_vector_by_quaternion(focal_point, device.emitter.camera_rotation);
  aperture_point = rotate_vector_by_quaternion(aperture_point, device.emitter.camera_rotation);

  task.ray    = normalize_vector(sub_vector(focal_point, aperture_point));
  task.origin = add_vector(device.scene.camera.pos, aperture_point);

  return task;
}

#endif /* CU_CAMERA_H */
