#ifndef CU_LUMINARY_CAMERA_LENS_H
#define CU_LUMINARY_CAMERA_LENS_H

#include "camera_simulation.cuh"
#include "camera_utils.cuh"
#include "math.cuh"
#include "utils.cuh"

// We force the weight to be 1, else the brightness of the image would depend on aperture size.
// That would be realistic but not practical.
__device__ vec3 camera_lens_sample_initial_direction(const vec3 sensor_point, const ushort2 pixel) {
  vec3 target_point = get_vector(0.0f, 0.0f, -device.camera.focal_length);

  if (device.camera.aperture_size != 0.0f) {
#ifndef CAMERA_DEBUG_RENDER
    const float2 random = random_2D(RANDOM_TARGET_LENS, pixel);
#else
    const float2 random = make_float2(((float) pixel_coords.x) / device.settings.width, ((float) pixel_coords.y) / device.settings.height);
#endif
    float2 sample;

    switch (device.camera.aperture_shape) {
      default:
      case LUMINARY_APERTURE_ROUND: {
        const float alpha = random.x * 2.0f * PI;
        const float beta  = sqrtf(random.y) * device.camera.aperture_size;

        sample = make_float2(cosf(alpha) * beta, sinf(alpha) * beta);
      } break;
      case LUMINARY_APERTURE_BLADED: {
        const int blade   = random_1D(RANDOM_TARGET_LENS_BLADE, pixel) * device.camera.aperture_blade_count;
        const float alpha = sqrtf(random.x);
        const float beta  = random.y;

        const float u = 1.0f - alpha;
        const float v = alpha * beta;

        const float angle_step = (2.0f * PI) / device.camera.aperture_blade_count;

        const float angle1 = angle_step * blade;
        const float angle2 = angle_step * (blade + 1);

        sample.x = sinf(angle1) * u + sinf(angle2) * v;
        sample.y = cosf(angle1) * u + cosf(angle2) * v;

        sample.x *= device.camera.aperture_size;
        sample.y *= device.camera.aperture_size;
      } break;
    }

    target_point = add_vector(target_point, get_vector(sample.x, sample.y, 0.0f));
  }

  return normalize_vector(sub_vector(target_point, sensor_point));
}

__device__ CameraSimulationResult camera_lens_sample(const vec3 sensor_point, const ushort2 pixel) {
  const vec3 initial_direction = camera_lens_sample_initial_direction(sensor_point, pixel);

  return camera_simulation_trace(sensor_point, initial_direction, pixel);
}

#endif /* CU_LUMINARY_CAMERA_LENS_H */
