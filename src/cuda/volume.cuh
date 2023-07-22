#ifndef CU_VOLUME_H
#define CU_VOLUME_H

#include "math.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

//
// This implements a volume renderer. These volumes are homogenous and bound by a maximum height relative to the planet's
// surface. Closed form tracking is used to solve any light interaction with such a volume.
// While this implementation is mostly generic, some details are handled simply with the constraint that Luminary only has
// two types of volumes, fog and ocean water.
//

////////////////////////////////////////////////////////////////////
// Literature
////////////////////////////////////////////////////////////////////

// [FonWKH17]
// J. Fong, M. Wrenninge, C. Kulla, R. Habel, "Production Volume Rendering", SIGGRAPH 2017 Course, 2017
// https://graphics.pixar.com/library/ProductionVolumeRendering/

/*
 * Computes the start and length of a ray path through a volume.
 * The output path is only valid if the start is non-negative.
 *
 * @param origin Start point of ray in world space.
 * @param ray Direction of ray.
 * @param limit Maximum distance a ray may travel in world space.
 * @result Two floats:
 *                  - [x] = Start in world space.
 *                  - [y] = Distance through fog in world space.
 */
__device__ float2 volume_compute_path(const VolumeDescriptor volume, const vec3 origin, const vec3 ray, const float limit) {
  // Vertical intersection
  float start_y;
  float end_y;
  if (fabsf(ray.y) < eps) {
    if (origin.y >= OCEAN_MIN_HEIGHT && origin.y <= OCEAN_MAX_HEIGHT) {
      start_y = 0.0f;
      end_y   = FLT_MAX;
    }
    else {
      return make_float2(-FLT_MAX, 0.0f);
    }
  }
  else {
    const float dy1 = volume.min_height - origin.y;
    const float dy2 = volume.max_height - origin.y;

    const float sy1 = dy1 / ray.y;
    const float sy2 = dy2 / ray.y;

    start_y = fmaxf(fminf(sy1, sy2), 0.0f);
    end_y   = fmaxf(sy1, sy2);
  }

  // Horizontal intersection

  const float rn = 1.0f / sqrtf(ray.x * ray.x + ray.z * ray.z);
  const float rx = ray.x * rn;
  const float rz = ray.z * rn;

  const float dx = origin.x - device.scene.camera.pos.x;
  const float dz = origin.z - device.scene.camera.pos.z;

  const float dot = dx * rx + dz * rz;
  const float r2  = volume.dist * volume.dist;
  const float c   = (dx * dx + dz * dz) - r2;

  const float kx = dx - rx * dot;
  const float kz = dz - rz * dot;

  const float d = r2 - (kx * kx + kz * kz);

  if (d < 0.0f)
    return make_float2(-FLT_MAX, 0.0f);

  const float sd = sqrtf(d);
  const float q  = -dot - copysignf(sd, dot);

  const float t0 = fmaxf(0.0f, c / q);
  const float t1 = fmaxf(0.0f, q);

  const float start_xz = fminf(t0, t1);
  const float end_xz   = fmaxf(t0, t1);

  const float start = fmaxf(start_xz, start_y);
  const float dist  = fminf(fminf(end_xz, end_y) - start, limit - start);

  if (dist < 0.0f)
    return make_float2(-FLT_MAX, 0.0f);

  return make_float2(start, dist);
}

/*
 * Computes a random volume intersection point by perfectly importance sampling the transmittance
 * based pdf.
 *
 * @param volume VolumeDescriptor of the corresponding volume.
 * @param origin Origin of ray in world space.
 * @param ray Direction of ray.
 * @param start Start offset of ray.
 * @param max_dist Maximum dist ray may travel after start.
 * @result Distance of intersection point and origin in world space.
 */
__device__ float volume_sample_intersection(
  const VolumeDescriptor volume, const vec3 origin, const vec3 ray, const float start, const float max_dist) {
  // [FonWKH17] Equation 15
  const float t = (-logf(1.0f - white_noise())) / volume.avg_scattering;

  if (t > max_dist)
    return FLT_MAX;

  return start + t;
}

////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////

__global__ void volume_generate_g_buffer() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device.ptrs.task_counts[id * 6 + 4];
  const int task_offset = device.ptrs.task_offsets[id * 5 + 4];

  for (int i = 0; i < task_count; i++) {
    VolumeTask task = load_volume_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    vec3 ray;
    ray.x = cosf(task.ray_xz) * cosf(task.ray_y);
    ray.y = sinf(task.ray_y);
    ray.z = sinf(task.ray_xz) * cosf(task.ray_y);

    GBufferData data;
    data.hit_id    = task.volume_type;
    data.albedo    = RGBAF_set(0.0f, 0.0f, 0.0f, 0.0f);
    data.emission  = get_color(0.0f, 0.0f, 0.0f);
    data.flags     = G_BUFFER_REQUIRES_SAMPLING | G_BUFFER_VOLUME_HIT;
    data.normal    = get_vector(0.0f, 0.0f, 0.0f);
    data.position  = task.position;
    data.V         = scale_vector(ray, -1.0f);
    data.roughness = 1.0f;
    data.metallic  = 0.0f;

    store_g_buffer_data(data, pixel);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10) void volume_process_events() {
  const int task_count = device.trace_count[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device.trace_tasks + offset);
    const float2 result = __ldcs((float2*) (device.ptrs.trace_results + offset));

    float depth     = result.x;
    uint32_t hit_id = __float_as_uint(result.y);

    const int pixel = task.index.x + task.index.y * device.width;
    RGBF record     = load_RGBF(device.records + pixel);

    if (device.scene.fog.active) {
      const VolumeDescriptor volume = volume_get_descriptor_preset_fog();
      const float2 path             = volume_compute_path(volume, task.origin, task.ray, depth);

      if (path.x >= 0.0f) {
        if (device.iteration_type == TYPE_LIGHT) {
          record = scale_color(record, expf(-path.y * volume.avg_scattering));
        }
        else {
          const float volume_dist = volume_sample_intersection(volume, task.origin, task.ray, path.x, path.y);

          if (volume_dist < depth) {
            depth  = volume_dist;
            hit_id = VOLUME_FOG_HIT;
          }
        }
      }
    }

    if (device.scene.ocean.active) {
      const VolumeDescriptor volume = volume_get_descriptor_preset_ocean();
      const float2 path             = volume_compute_path(volume, task.origin, task.ray, depth);

      if (path.x >= 0.0f) {
        if (device.iteration_type == TYPE_LIGHT) {
          RGBF volume_transmittance = volume_get_transmittance(volume);

          record.r *= expf(-path.y * volume_transmittance.r);
          record.g *= expf(-path.y * volume_transmittance.g);
          record.b *= expf(-path.y * volume_transmittance.b);
        }
        else {
          const float volume_dist = volume_sample_intersection(volume, task.origin, task.ray, path.x, path.y);

          if (volume_dist < depth) {
            record.r *= (volume.scattering.r * expf(-volume_dist * volume.scattering.r))
                        / (volume.avg_scattering * expf(-volume_dist * volume.avg_scattering));
            record.g *= (volume.scattering.g * expf(-volume_dist * volume.scattering.g))
                        / (volume.avg_scattering * expf(-volume_dist * volume.avg_scattering));
            record.b *= (volume.scattering.b * expf(-volume_dist * volume.scattering.b))
                        / (volume.avg_scattering * expf(-volume_dist * volume.avg_scattering));

            depth  = volume_dist;
            hit_id = VOLUME_OCEAN_HIT;
          }

          record.r *= expf(-depth * volume.absorption.r);
          record.g *= expf(-depth * volume.absorption.g);
          record.b *= expf(-depth * volume.absorption.b);
        }
      }
    }

    __stcs((float2*) (device.ptrs.trace_results + offset), make_float2(depth, __uint_as_float(hit_id)));
    store_RGBF(device.records + pixel, record);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 9) void volume_process_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count   = device.ptrs.task_counts[id * 6 + 4];
  const int task_offset  = device.ptrs.task_offsets[id * 5 + 4];
  int light_trace_count  = device.ptrs.light_trace_count[id];
  int bounce_trace_count = device.ptrs.bounce_trace_count[id];

  for (int i = 0; i < task_count; i++) {
    VolumeTask task = load_volume_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    VolumeType volume_type = VOLUME_HIT_TYPE(task.volume_type);

    vec3 ray;
    ray.x = cosf(task.ray_xz) * cosf(task.ray_y);
    ray.y = sinf(task.ray_y);
    ray.z = sinf(task.ray_xz) * cosf(task.ray_y);

    const VolumeDescriptor volume = volume_get_descriptor_preset(volume_type);

    RGBF record = device.records[pixel];

    {
      // TODO: Importance sample phase function using inverse CDF LUT.
      const vec3 bounce_ray = angles_to_direction(white_noise() * PI, white_noise() * 2.0f * PI);
      const float cos_angle = dot_product(ray, bounce_ray);
      const float phase     = jendersie_eon_phase_function(cos_angle, volume.water_droplet_diameter);

      const float weight = 4.0f * PI * phase;

      device.ptrs.bounce_records[pixel] = scale_color(record, weight);

      TraceTask bounce_task;
      bounce_task.origin = task.position;
      bounce_task.ray    = bounce_ray;
      bounce_task.index  = task.index;

      store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), bounce_task);
    }

    const int light_occupied = (device.ptrs.state_buffer[pixel] & STATE_LIGHT_OCCUPIED);

    if (!light_occupied) {
      LightSample light = load_light_sample(device.ptrs.light_samples, pixel);

      uint32_t light_history_buffer_entry = LIGHT_ID_ANY;

      BRDFInstance brdf = brdf_get_instance_scattering();

      if (light.weight > 0.0f) {
        BRDFInstance brdf_sample = brdf_apply_sample_scattering(brdf, light, task.position, volume_type);

        device.ptrs.light_records[pixel] = mul_color(record, brdf_sample.term);
        light_history_buffer_entry       = light.id;

        TraceTask light_task;
        light_task.origin = task.position;
        light_task.ray    = brdf_sample.L;
        light_task.index  = task.index;

        store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), light_task);
      }

      device.ptrs.light_sample_history[pixel] = light_history_buffer_entry;
    }
  }

  device.ptrs.light_trace_count[id]  = light_trace_count;
  device.ptrs.bounce_trace_count[id] = bounce_trace_count;
}

#endif /* CU_VOLUME_H */
