#ifndef CU_LUMINARY_CAMERA_LENS_H
#define CU_LUMINARY_CAMERA_LENS_H

#include "camera_simulation.cuh"
#include "camera_utils.cuh"
#include "math.cuh"
#include "utils.cuh"

// We force the weight to be 1, else the brightness of the image would depend on aperture size.
// That would be realistic but not practical.
__device__ vec3 camera_lens_sample_initial_direction(const vec3 sensor_point, const ushort2 pixel, float& weight) {
  // Positioned at back vertex, which is the origin.
  vec3 target_point = get_vector(0.0f, 0.0f, 0.0f);

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

  const vec3 diff = sub_vector(target_point, sensor_point);

  const float dist        = get_length(diff);
  const float image_plane = camera_thin_lens_image_plane(camera_thin_lens_get_ior(CAMERA_DESIGN_WAVELENGTH));

  const vec3 ray = normalize_vector(diff);

  weight = fabsf(ray.z) * (image_plane * image_plane) / (dist * dist);

  return ray;
}

template <bool ALLOW_REFLECTIONS, bool SPECTRAL_RENDERING>
__device__ CameraSimulationResult camera_lens_sample(const vec3 sensor_point, const float wavelength, const ushort2 pixel) {
  float initial_weight;
  const vec3 initial_direction = camera_lens_sample_initial_direction(sensor_point, pixel, initial_weight);

  CameraSimulationResult result =
    camera_simulation_trace<ALLOW_REFLECTIONS, SPECTRAL_RENDERING>(sensor_point, initial_direction, wavelength, pixel);

  result.weight = scale_color(result.weight, initial_weight);

  return result;
}

#endif /* CU_LUMINARY_CAMERA_LENS_H */
