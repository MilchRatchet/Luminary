#ifndef CU_VOLUME_H
#define CU_VOLUME_H

#include "math.cuh"
#include "ocean_utils.cuh"
#include "state.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

//
// This implements a volume renderer. These volumes are homogenous and bound by a disk-box (A horizontal disk with a width in the vertical
// axis). Closed form tracking is used to solve any light interaction with such a volume. While this implementation is mostly generic, some
// details are handled simply with the constraint that Luminary only has two types of volumes, fog and ocean water. Fog does not use
// absorption and only has scalar scattering values while ocean scattering and absorption is performed using three color channels.
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
  if (volume.max_height <= volume.min_height)
    return make_float2(-FLT_MAX, 0.0f);

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

  // Vertical intersection
  float start_y;
  float end_y;
  if (volume.type == VOLUME_TYPE_OCEAN) {
    const bool above_surface = ocean_get_relative_height(origin, OCEAN_ITERATIONS_INTERSECTION) > 0.0f;

    const float surface_intersect = ocean_intersection_distance(origin, ray, limit);

    if (above_surface) {
      start_y = surface_intersect;
      end_y   = FLT_MAX;
    }
    else {
      start_y = 0.0f;
      end_y   = surface_intersect;
    }
  }
  else {
    if (fabsf(ray.y) < 0.005f) {
      if (origin.y >= volume.min_height && origin.y <= volume.max_height) {
        start_y = 0.0f;
        end_y   = volume.dist;
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
  }

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
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_VOLUME];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_VOLUME];

  for (int i = 0; i < task_count; i++) {
    VolumeTask task = load_volume_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    uint32_t flags = (!state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) ? G_BUFFER_REQUIRES_SAMPLING : 0;
    flags |= G_BUFFER_VOLUME_HIT;

    GBufferData data;
    data.hit_id    = task.hit_id;
    data.albedo    = RGBAF_set(0.0f, 0.0f, 0.0f, 0.0f);
    data.emission  = get_color(0.0f, 0.0f, 0.0f);
    data.normal    = get_vector(0.0f, 0.0f, 0.0f);
    data.position  = task.position;
    data.V         = scale_vector(task.ray, -1.0f);
    data.roughness = 1.0f;
    data.metallic  = 0.0f;
    data.flags     = flags;

    store_g_buffer_data(data, pixel);
  }
}

__global__ void volume_process_events() {
  const int task_count = device.trace_count[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device.trace_tasks + offset);
    const float2 result = __ldcs((float2*) (device.ptrs.trace_results + offset));

    float depth     = result.x;
    uint32_t hit_id = __float_as_uint(result.y);

    if (device.scene.fog.active) {
      const VolumeDescriptor volume = volume_get_descriptor_preset_fog();
      const float2 path             = volume_compute_path(volume, task.origin, task.ray, depth);

      if (path.x >= 0.0f) {
        const float volume_dist = volume_sample_intersection(volume, task.origin, task.ray, path.x, path.y);

        if (volume_dist < depth) {
          depth  = volume_dist;
          hit_id = HIT_TYPE_VOLUME_FOG;
        }
      }
    }

    if (device.scene.ocean.active) {
      const VolumeDescriptor volume = volume_get_descriptor_preset_ocean();
      const float2 path             = volume_compute_path(volume, task.origin, task.ray, depth);

      if (path.x >= 0.0f) {
        const float volume_dist = volume_sample_intersection(volume, task.origin, task.ray, path.x, path.y);

        if (volume_dist < depth) {
          depth  = volume_dist;
          hit_id = HIT_TYPE_VOLUME_OCEAN;
        }
      }
    }

    __stcs((float2*) (device.ptrs.trace_results + offset), make_float2(depth, __uint_as_float(hit_id)));
  }
}

__global__ void volume_process_events_weight() {
  const int task_count = device.trace_count[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device.trace_tasks + offset);
    const float2 result = __ldcs((float2*) (device.ptrs.trace_results + offset));

    float depth     = result.x;
    uint32_t hit_id = __float_as_uint(result.y);

    const int pixel = task.index.x + task.index.y * device.width;
    RGBF record     = load_RGBF(device.records + pixel);

    if (device.scene.fog.active) {
      if (device.iteration_type == TYPE_LIGHT) {
        const VolumeDescriptor volume = volume_get_descriptor_preset_fog();
        const float2 path             = volume_compute_path(volume, task.origin, task.ray, depth);

        if (path.x >= 0.0f) {
          record = scale_color(record, expf(-path.y * volume.avg_scattering));
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
          if (hit_id == HIT_TYPE_VOLUME_OCEAN) {
            record.r *=
              (volume.scattering.r * expf(-path.y * volume.scattering.r)) / (volume.avg_scattering * expf(-path.y * volume.avg_scattering));
            record.g *=
              (volume.scattering.g * expf(-path.y * volume.scattering.g)) / (volume.avg_scattering * expf(-path.y * volume.avg_scattering));
            record.b *=
              (volume.scattering.b * expf(-path.y * volume.scattering.b)) / (volume.avg_scattering * expf(-path.y * volume.avg_scattering));
          }

          record.r *= expf(-path.y * volume.absorption.r);
          record.g *= expf(-path.y * volume.absorption.g);
          record.b *= expf(-path.y * volume.absorption.b);
        }
      }
    }

    store_RGBF(device.records + pixel, record);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 9) void volume_process_tasks() {
  const int task_count   = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_VOLUME];
  const int task_offset  = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_VOLUME];
  int light_trace_count  = device.ptrs.light_trace_count[THREAD_ID];
  int bounce_trace_count = device.ptrs.bounce_trace_count[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    VolumeTask task = load_volume_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    VolumeType volume_type = VOLUME_HIT_TYPE(task.hit_id);

    const VolumeDescriptor volume = volume_get_descriptor_preset(volume_type);

    RGBF record = device.records[pixel];

    write_albedo_buffer(get_color(0.0f, 0.0f, 0.0f), pixel);

    const vec3 bounce_ray = (volume.type == VOLUME_TYPE_FOG) ? jendersie_eon_phase_sample(task.ray, device.scene.fog.droplet_diameter)
                                                             : ocean_phase_sampling(task.ray);

    TraceTask bounce_task;
    bounce_task.origin = task.position;
    bounce_task.ray    = bounce_ray;
    bounce_task.index  = task.index;

    device.ptrs.mis_buffer[pixel] = 0.0f;
    store_RGBF(device.ptrs.bounce_records + pixel, record);
    store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), bounce_task);

    if (!state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
      LightSample light = load_light_sample(device.ptrs.light_samples, pixel);

      uint32_t light_history_buffer_entry = LIGHT_ID_ANY;

      BRDFInstance brdf = brdf_get_instance_scattering(scale_vector(task.ray, -1.0f));

      if (light.weight > 0.0f) {
        BRDFInstance brdf_sample = brdf_apply_sample_scattering(brdf, light, task.position, volume_type);

        const RGBF light_record = mul_color(record, brdf_sample.term);

        TraceTask light_task;
        light_task.origin = task.position;
        light_task.ray    = brdf_sample.L;
        light_task.index  = task.index;

        if (luminance(light_record) > 0.0f && state_consume(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
          store_RGBF(device.ptrs.light_records + pixel, light_record);
          light_history_buffer_entry = light.id;
          store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), light_task);
        }
      }

      device.ptrs.light_sample_history[pixel] = light_history_buffer_entry;
    }
  }

  device.ptrs.light_trace_count[THREAD_ID]  = light_trace_count;
  device.ptrs.bounce_trace_count[THREAD_ID] = bounce_trace_count;
}

#endif /* CU_VOLUME_H */
